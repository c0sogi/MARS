import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
from scipy.stats import spearmanr
from transformers import AutoTokenizer

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.dataset import get_dataloaders
from library.trainer import Trainer, set_seed
from library.metrics import compute_spearmanr


def main():
    # 1. Setup and Configuration
    warnings.filterwarnings("ignore")

    # Modify Config for Fast Baseline
    # The dataset is small (~4.4k train), so we can run a few epochs quickly on A100.
    Config.EPOCHS = 3
    Config.TRAIN_BATCH_SIZE = 8
    Config.VALID_BATCH_SIZE = 16

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loading
    print("Loading data...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # We need val_loader explicitly for final validation and failure analysis
    # trainer.fit() returns test_loader, but we get all of them here to be sure
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer, load_cached_data=True
    )

    # 3. Training
    print("Starting training process...")
    trainer = Trainer()

    # fit() trains the model and loads the best checkpoint at the end
    # It returns the test_loader, but we already have it from get_dataloaders
    _ = trainer.fit()

    # 4. Final Validation & Metric
    print("Performing final validation...")
    trainer.model.eval()
    val_preds_list = []
    val_labels_list = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                "input_ids_q": batch["input_ids_q"].to(device),
                "attention_mask_q": batch["attention_mask_q"].to(device),
                "input_ids_a": batch["input_ids_a"].to(device),
                "attention_mask_a": batch["attention_mask_a"].to(device),
            }

            if "token_type_ids_q" in batch:
                inputs["token_type_ids_q"] = batch["token_type_ids_q"].to(device)
            if "token_type_ids_a" in batch:
                inputs["token_type_ids_a"] = batch["token_type_ids_a"].to(device)

            labels = batch["labels"].to(device)

            logits = trainer.model(**inputs)
            probs = torch.sigmoid(logits)

            val_preds_list.append(probs.cpu().numpy())
            val_labels_list.append(labels.cpu().numpy())

    val_preds = np.concatenate(val_preds_list, axis=0)
    val_targets = np.concatenate(val_labels_list, axis=0)

    final_metric = compute_spearmanr(val_targets, val_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate Mean Absolute Error per sample (averaged across 30 targets)
    # Shape: (N_samples,)
    mae_per_sample = np.mean(np.abs(val_targets - val_preds), axis=1)

    # Load validation metadata to get text features
    val_df = pd.read_csv(Config.VAL_PATH)

    # Ensure alignment: The DataLoader does not shuffle validation, so indices align.
    # However, let's double check lengths
    if len(val_df) != len(mae_per_sample):
        print(
            f"Warning: Metadata length ({len(val_df)}) != Predictions length ({len(mae_per_sample)})"
        )
        # Truncate to minimum length just in case of drop_last mismatch (though val shouldn't drop last)
        min_len = min(len(val_df), len(mae_per_sample))
        val_df = val_df.iloc[:min_len]
        mae_per_sample = mae_per_sample[:min_len]

    # Extract features
    val_df["q_body_len"] = val_df["question_body"].fillna("").astype(str).apply(len)
    val_df["a_len"] = val_df["answer"].fillna("").astype(str).apply(len)

    # Calculate correlations
    corr_q_len, _ = spearmanr(mae_per_sample, val_df["q_body_len"])
    corr_a_len, _ = spearmanr(mae_per_sample, val_df["a_len"])

    print(f"Correlation (Error vs Question Body Length): {corr_q_len}")
    print(f"Correlation (Error vs Answer Length): {corr_a_len}")

    # 6. Submission
    THRESHOLD = 0.38439426044249936

    if final_metric > THRESHOLD:
        print(
            f"\nValidation metric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        qa_ids, preds = trainer.predict(test_loader)
        trainer.generate_submission(qa_ids, preds)
    else:
        print(
            f"\nValidation metric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
