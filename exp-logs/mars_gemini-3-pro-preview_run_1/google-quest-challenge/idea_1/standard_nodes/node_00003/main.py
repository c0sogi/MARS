import sys
import os
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.trainer import Trainer


def main():
    # ==========================================
    # 1. Setup & Initialization
    # ==========================================
    print("Initializing configuration and seeds...")
    Config.setup()
    seed_everything(Config.SEED)

    # ==========================================
    # 2. Model Training
    # ==========================================
    print("\nInitializing Trainer and starting training...")
    trainer = Trainer()

    # The fit method handles:
    # - Data loading (and caching)
    # - Training loop with Early Stopping
    # - Saving and reloading the best model
    trainer.fit()

    # ==========================================
    # 3. Validation Assessment
    # ==========================================
    print("\nPerforming final validation assessment...")
    # Calculate metric on the full validation set using the best model
    val_loss, val_score = trainer.validate()

    # Print the required metric
    print(f"Final Validation Metric: {val_score}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\nPerforming failure analysis...")

    # Ensure model is in eval mode
    trainer.model.eval()

    all_preds = []
    all_targets = []

    # Use the validation loader from the trainer
    if trainer.val_loader is None:
        raise RuntimeError("Validation loader not found in trainer instance.")

    device = trainer.device

    # Generate predictions on validation set
    with torch.no_grad():
        for q_seq, a_seq, targets in trainer.val_loader:
            q_seq = q_seq.to(device)
            a_seq = a_seq.to(device)

            outputs = trainer.model(q_seq, a_seq)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate results
    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)

    # Calculate Mean Absolute Error (MAE) per sample across all 30 targets
    # Shape: (n_samples,)
    sample_errors = np.mean(np.abs(preds_arr - targets_arr), axis=1)

    # Load validation metadata to get input features
    # We must respect the Config.DEBUG flag to ensure alignment with the data loader
    val_df = pd.read_csv(Config.VAL_PATH)
    if Config.DEBUG:
        val_df = val_df.iloc[: Config.DEBUG_SIZE]

    # Verify alignment between dataframe and predictions
    if len(val_df) != len(sample_errors):
        print(
            f"Warning: Validation DataFrame length ({len(val_df)}) differs from prediction count ({len(sample_errors)}). Truncating to match."
        )
        min_len = min(len(val_df), len(sample_errors))
        val_df = val_df.iloc[:min_len]
        sample_errors = sample_errors[:min_len]

    # Extract features: Character lengths of Question and Answer
    # Question = Title + Body
    q_text = (
        val_df["question_title"].fillna("") + " " + val_df["question_body"].fillna("")
    ).astype(str)
    val_df["q_char_len"] = q_text.str.len()

    a_text = val_df["answer"].fillna("").astype(str)
    val_df["a_char_len"] = a_text.str.len()

    # Calculate correlations
    corr_q, _ = pearsonr(val_df["q_char_len"], sample_errors)
    corr_a, _ = pearsonr(val_df["a_char_len"], sample_errors)

    print(f"Correlation between Error and Question Length: {corr_q}")
    print(f"Correlation between Error and Answer Length: {corr_a}")

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    print("\nGenerating submission file...")
    trainer.predict()
    print("Process completed successfully.")


if __name__ == "__main__":
    main()
