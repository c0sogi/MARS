import sys
import os
import pandas as pd
import numpy as np
import torch

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import Config, seed_everything
from library.dataset import TextNormalizationDataset
from library.trainer import Trainer
from library.model import predict_labels
from library.normalization_rules import Normalizer


def main():
    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    print("==== Starting Runfile ====")

    # 1. Load Data
    # We limit training data to ensure fast execution within the time limit (Fast Baseline).
    # 15,000 sentences provides a good balance between speed and performance.
    print("Loading Training Dataset...")
    train_ds = TextNormalizationDataset(split="train", debug_size=15000)

    # We must use the full validation set for the metric calculation as per requirements.
    print("Loading Validation Dataset...")
    val_ds = TextNormalizationDataset(split="val", debug_size=None)

    # 2. Training
    print("Initializing Trainer...")
    trainer = Trainer()

    # Train for 1 epoch to ensure completion within the 2-hour limit.
    print("Starting Training (1 Epoch)...")
    trainer.train(train_ds, val_ds, epochs=1)

    # 3. Validation Inference & Metric Calculation
    print("Performing Validation Inference...")
    # predict_labels returns list of lists (sentences) of predicted class labels
    # The model is automatically moved to GPU in the library functions.
    preds_by_sentence = predict_labels(trainer.model, val_ds)

    # Flatten predictions to token level to match the metadata dataframe
    flat_preds = [label for sent in preds_by_sentence for label in sent]

    # Load validation metadata to compare against ground truth
    print("Loading Validation Metadata...")
    df_val = pd.read_csv(
        Config.VAL_DATA_PATH,
        keep_default_na=False,
        dtype={"sentence_id": int, "token_id": int},
    )

    # Ensure strict alignment: Sort by sentence_id, token_id
    # The dataset class processes sentences in sorted order of sentence_id
    df_val = df_val.sort_values(["sentence_id", "token_id"])

    # Verify alignment
    if len(flat_preds) != len(df_val):
        print(
            f"Warning: Prediction count {len(flat_preds)} does not match Validation Data count {len(df_val)}."
        )
        # Truncate to match in case of minor discrepancies, though alignment should be exact
        min_len = min(len(flat_preds), len(df_val))
        flat_preds = flat_preds[:min_len]
        df_val = df_val.iloc[:min_len]

    df_val["class_pred"] = flat_preds

    # Apply Normalization Rules
    print("Applying Normalization Rules to Validation Set...")
    norm = Normalizer()

    # We use the predicted class to normalize the 'before' text
    # This matches the inference logic used for submission
    df_val["after_pred"] = df_val.apply(
        lambda row: norm.normalize(row["before"], row["class_pred"]), axis=1
    )

    # Calculate Accuracy
    # Exact string match required
    df_val["is_correct"] = df_val["after"] == df_val["after_pred"]
    final_metric = df_val["is_correct"].mean()

    # Print Required Metric Format
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n==== Failure Analysis ====")

    # Correlation between error and input length
    # Error = 1 (incorrect), 0 (correct)
    df_val["error_flag"] = (~df_val["is_correct"]).astype(int)
    df_val["len_before"] = df_val["before"].str.len()

    # Calculate correlation
    corr = df_val["len_before"].corr(df_val["error_flag"])
    print(f"Correlation (Token Length vs Error Magnitude): {corr}")

    # Print top error classes to identify systematic patterns
    print("\nAccuracy by Class (Lowest 5):")
    class_acc = df_val.groupby("class")["is_correct"].mean().sort_values()
    print(class_acc.head(5))

    # 5. Submission Generation
    THRESHOLD = 0.8801040350013403

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating Submission..."
        )

        # Load Test Data
        print("Loading Test Dataset...")
        test_ds = TextNormalizationDataset(split="test", debug_size=None)

        # Generate Submission
        # This function handles prediction, normalization, and saving to CSV
        trainer.generate_submission(test_ds)
        print("Submission generated successfully.")
    else:
        print(
            f"\nMetric ({final_metric}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
