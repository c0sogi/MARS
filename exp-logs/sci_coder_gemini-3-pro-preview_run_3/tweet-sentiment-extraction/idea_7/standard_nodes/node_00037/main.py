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


from sklearn.model_selection import StratifiedKFold
from library.dataset import get_data, get_loaders


def run():
    # 1. Initialization and Configuration
    seed_everything(Config.seed)

    # Use full dataset for Cross-Validation (Cite solution_lesson_node_00028)
    Config.original_train_path = "./input/train.csv"

    # Disable AWP and use standard epochs (Cite solution_lesson_node_00035)
    Config.use_awp = False
    Config.epochs = 3

    # Clear any existing cache to ensure the new training data is processed correctly.
    print("Clearing data cache...")
    cache_files = glob.glob(os.path.join(Config.working_dir, "cached_*"))
    for f in cache_files:
        try:
            os.remove(f)
        except OSError:
            pass

    # 2. Training Loop (5 Folds) + OOF Prediction
    print(f"Starting Training on {Config.original_train_path}...")

    # Load processed data (pos/neg only) to get indices
    df_filtered, _, _, _, _, _ = get_data(load_cached_data=False)

    # Setup StratifiedKFold to match get_loaders logic
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )
    splits = list(skf.split(df_filtered, df_filtered["sentiment"]))

    oof_preds = {}  # Store predictions by textID

    for fold in range(Config.n_folds):
        # Train
        train_fold(fold)

        # Inference on Validation Fold (OOF)
        print(f"Generating OOF predictions for Fold {fold}...")
        _, val_loader = get_loaders(fold)

        model_path = os.path.join(Config.working_dir, f"model_fold_{fold}.bin")
        model = TweetModel(Config)
        model.load_state_dict(torch.load(model_path, map_location=Config.device))
        model.to(Config.device)
        model.eval()

        _, (start_logits, end_logits) = eval_fn(val_loader, model, Config.device)

        # Decode
        val_idx = splits[fold][1]
        val_dataset = val_loader.dataset

        for i in range(len(val_dataset)):
            # Get original textID from the dataframe using the validation index
            text_id = df_filtered.iloc[val_idx[i]]["textID"]

            # Get metadata for decoding
            text = val_dataset.texts[i]
            offsets = val_dataset.offsets[i]
            cleaned_text = " " + " ".join(str(text).split())

            pred = get_best_start_end_idxs(
                start_logits[i], end_logits[i], cleaned_text, offsets
            )
            oof_preds[text_id] = pred

        del model
        torch.cuda.empty_cache()

    # 3. Compute Final Metric on Full Dataset (including Neutrals)
    print("Computing Final Validation Metric...")
    df_full = pd.read_csv(Config.original_train_path)
    df_full.dropna(subset=["text", "selected_text", "sentiment"], inplace=True)

    predictions = []
    scores = []

    for idx, row in df_full.iterrows():
        text_id = row["textID"]
        sentiment = row["sentiment"]
        text = str(row["text"])
        selected_text = str(row["selected_text"])

        # Deterministic Rule for Neutral (Cite solution_lesson_node_00006)
        if sentiment == "neutral":
            pred = text
        else:
            # Retrieve OOF prediction
            if text_id in oof_preds:
                pred = oof_preds[text_id]
            else:
                # Fallback (should not happen if alignment logic matches)
                pred = text

        score = jaccard(pred, selected_text)
        predictions.append(pred)
        scores.append(score)

    final_metric = np.mean(scores)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    df_full["jaccard"] = scores
    df_full["error"] = 1.0 - df_full["jaccard"]
    df_full["text_len"] = df_full["text"].apply(len)

    # Map sentiment to numeric for correlation
    sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}
    df_full["sentiment_code"] = df_full["sentiment"].map(sentiment_map)

    # Calculate correlations
    corr_len, _ = pearsonr(df_full["error"], df_full["text_len"])
    corr_sent, _ = pearsonr(df_full["error"], df_full["sentiment_code"])

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
