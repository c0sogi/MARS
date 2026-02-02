import os
import glob
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, jaccard
from library.dataset import TweetDataset
from library.model import TweetModel
from library.engine import eval_fn
from library.inference import train_fold, generate_submission, get_best_start_end_idxs


def run():
    # 1. Initialization and Configuration
    seed_everything(Config.seed)

    # Clear any existing cache to ensure the new training data is processed correctly.
    # The cache is located in the working directory defined in Config.
    print("Clearing data cache...")
    cache_files = glob.glob(os.path.join(Config.working_dir, "cached_*"))
    for f in cache_files:
        try:
            os.remove(f)
        except OSError:
            pass

    # 2. Training Loop (5 Folds)
    print(f"Starting Training on {Config.original_train_path}...")
    for fold in range(Config.n_folds):
        # train_fold handles data loading, model init, training, and saving best checkpoint
        train_fold(fold)

    # 3. Out-Of-Fold (OOF) Evaluation
    print("Starting Out-Of-Fold (OOF) Evaluation...")

    # Load data needed for OOF
    from library.dataset import get_data
    from sklearn.model_selection import StratifiedKFold

    # Load filtered training data (pos/neg only)
    # get_data handles loading/caching. Since we cleared cache, it will re-process Config.original_train_path
    df_filtered, input_ids, attention_mask, _, _, offsets = get_data(
        load_cached_data=True
    )

    # Initialize OOF predictions container for the filtered dataset
    oof_preds = [""] * len(df_filtered)

    # Setup StratifiedKFold to match training split
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    # Iterate through folds to generate predictions for validation sets
    for fold, (_, val_idx) in enumerate(
        skf.split(df_filtered, df_filtered["sentiment"])
    ):
        print(f"Generating OOF predictions for Fold {fold}...")

        # Create Validation Dataset for this fold
        val_dataset = TweetDataset(
            input_ids=input_ids[val_idx],
            attention_mask=attention_mask[val_idx],
            texts=df_filtered.iloc[val_idx]["text"].values,
            offsets=offsets[val_idx],
            sentiments=df_filtered.iloc[val_idx]["sentiment"].values,
            is_test=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Load Model
        model_path = os.path.join(Config.working_dir, f"model_fold_{fold}.bin")
        model = TweetModel(Config)
        model.load_state_dict(torch.load(model_path, map_location=Config.device))
        model.to(Config.device)
        model.eval()

        # Inference
        _, (start_logits, end_logits) = eval_fn(val_loader, model, Config.device)

        # Decode
        for i, idx in enumerate(val_idx):
            text = val_dataset.texts[i]
            off = val_dataset.offsets[i]
            cleaned_text = " " + " ".join(str(text).split())

            pred = get_best_start_end_idxs(
                start_logits[i], end_logits[i], cleaned_text, off
            )
            oof_preds[idx] = pred

        del model
        torch.cuda.empty_cache()

    # 4. Global Metric Calculation
    # We need to combine OOF preds (pos/neg) with Neutral preds (identity)
    print("Calculating Global Jaccard Score...")

    df_full = pd.read_csv(Config.original_train_path)
    df_full.dropna(subset=["text", "selected_text", "sentiment"], inplace=True)

    # Create a mapping from textID to OOF prediction for pos/neg
    # df_filtered is a subset of df_full. We use textID to map back.
    df_filtered["oof_pred"] = oof_preds
    pred_map = dict(zip(df_filtered["textID"], df_filtered["oof_pred"]))

    final_predictions = []
    scores = []

    for i in range(len(df_full)):
        row = df_full.iloc[i]
        text_id = row["textID"]
        sentiment = row["sentiment"]
        text = str(row["text"])
        selected_text = str(row["selected_text"])

        if sentiment == "neutral":
            pred = text
        else:
            # Retrieve OOF prediction
            # If textID not in map (dropped during preprocessing), fallback to text (or empty)
            # Using text as fallback is safer for Jaccard than empty string
            pred = pred_map.get(text_id, text)

        score = jaccard(pred, selected_text)
        final_predictions.append(pred)
        scores.append(score)

    final_metric = np.mean(scores)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Create a temporary dataframe for analysis
    df_analysis = df_full.copy()
    df_analysis["jaccard"] = scores
    df_analysis["error"] = 1.0 - df_analysis["jaccard"]
    df_analysis["text_len"] = df_analysis["text"].apply(len)

    # Map sentiment to numeric for correlation
    sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}
    df_analysis["sentiment_code"] = df_analysis["sentiment"].map(sentiment_map)

    # Calculate correlations
    corr_len, _ = pearsonr(df_analysis["error"], df_analysis["text_len"])
    corr_sent, _ = pearsonr(df_analysis["error"], df_analysis["sentiment_code"])

    print(f"Correlation (Error vs Text Length): {corr_len:.4f}")
    print(f"Correlation (Error vs Sentiment): {corr_sent:.4f}")

    # 5. Submission Generation
    threshold = 0.7164761348654044
    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric:.5f}) > Threshold ({threshold:.5f}). Generating submission..."
        )
        # generate_submission uses the models we just trained to predict on test.csv
        generate_submission()
    else:
        print(
            f"\nMetric ({final_metric:.5f}) <= Threshold ({threshold:.5f}). Submission skipped."
        )


if __name__ == "__main__":
    run()
