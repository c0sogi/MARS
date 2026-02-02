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

    # Override Config to use the metadata training split.
    # This ensures that metadata/val.csv is treated as a true hold-out set.
    Config.original_train_path = Config.train_metadata_path

    # Optimization for Fast Baseline: Reduce epochs to 2.
    # A100 is fast, but we want to ensure completion well within limits.
    Config.epochs = 2

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

    # 3. Holdout Validation Evaluation
    print("Starting Evaluation on Hold-out Validation Set...")

    # Load the hold-out validation set
    df_val = pd.read_csv(Config.val_metadata_path)

    # Preprocess validation data
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    input_ids_list = []
    attention_mask_list = []
    offsets_list = []
    texts = df_val["text"].values.astype(str)

    # Tokenize all validation texts
    for text in texts:
        # Normalize whitespace as done in training
        text = " " + " ".join(text.split())
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=Config.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )
        input_ids_list.append(encoded["input_ids"])
        attention_mask_list.append(encoded["attention_mask"])
        offsets_list.append(encoded["offset_mapping"])

    # Create Dataset and DataLoader for Validation
    # We pass selected_texts to allow for later analysis, though not used in __getitem__ for test mode
    val_dataset = TweetDataset(
        input_ids=np.array(input_ids_list),
        attention_mask=np.array(attention_mask_list),
        texts=texts,
        offsets=np.array(offsets_list),
        sentiments=df_val["sentiment"].values,
        selected_texts=df_val["selected_text"].values,
        is_test=True,  # Use test mode to skip start/end position requirements
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Ensemble Inference on Validation Set
    n_samples = len(df_val)
    final_start_logits = np.zeros((n_samples, Config.max_len))
    final_end_logits = np.zeros((n_samples, Config.max_len))

    print("Running Ensemble Inference...")
    for fold in range(Config.n_folds):
        model_path = os.path.join(Config.working_dir, f"model_fold_{fold}.bin")

        # Load Model
        model = TweetModel(Config)
        model.load_state_dict(torch.load(model_path, map_location=Config.device))
        model.to(Config.device)
        model.eval()

        # Get Logits
        _, (start_logits, end_logits) = eval_fn(val_loader, model, Config.device)

        # Accumulate
        final_start_logits += start_logits
        final_end_logits += end_logits

        # Cleanup
        del model
        torch.cuda.empty_cache()

    # Average Logits
    final_start_logits /= Config.n_folds
    final_end_logits /= Config.n_folds

    # Decode Predictions and Compute Metrics
    predictions = []
    scores = []

    for i in range(n_samples):
        sentiment = df_val.iloc[i]["sentiment"]
        text = texts[i]
        selected_text = str(df_val.iloc[i]["selected_text"])

        # Deterministic Rule for Neutral
        if sentiment == "neutral":
            pred = text
        else:
            # Model Prediction for Positive/Negative
            cleaned_text = " " + " ".join(text.split())
            offsets = val_dataset.offsets[i]
            pred = get_best_start_end_idxs(
                final_start_logits[i], final_end_logits[i], cleaned_text, offsets
            )

        score = jaccard(pred, selected_text)
        predictions.append(pred)
        scores.append(score)

    final_metric = np.mean(scores)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    df_val["jaccard"] = scores
    df_val["error"] = 1.0 - df_val["jaccard"]
    df_val["text_len"] = df_val["text"].apply(len)

    # Map sentiment to numeric for correlation
    sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}
    df_val["sentiment_code"] = df_val["sentiment"].map(sentiment_map)

    # Calculate correlations
    corr_len, _ = pearsonr(df_val["error"], df_val["text_len"])
    corr_sent, _ = pearsonr(df_val["error"], df_val["sentiment_code"])

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
