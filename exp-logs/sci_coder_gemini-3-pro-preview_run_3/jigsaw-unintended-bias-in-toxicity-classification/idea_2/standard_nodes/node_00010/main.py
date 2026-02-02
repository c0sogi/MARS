import os
import sys
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config, seed_everything
from library.trainer import Trainer
from library.utils import JigsawEvaluator


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Create working directory
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data Preparation
    # Using full dataset as configured in Config (no subsampling)
    print(f"Using full training data from {Config.TRAIN_PATH}")

    # 3. Training
    print("Initializing Trainer...")
    # debug=False ensures we use the file at Config.TRAIN_PATH (our subset)
    trainer = Trainer(debug=False)

    print("Starting training...")
    trainer.train()

    # 4. Validation & Metrics
    print("Performing validation inference...")
    trainer.model.eval()

    val_preds = []
    val_targets = []

    # Use the trainer's validation loader
    val_loader = trainer.val_loader

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(Config.DEVICE)
            attention_mask = batch["attention_mask"].to(Config.DEVICE)
            targets = batch["target"].to(Config.DEVICE)

            logits = trainer.model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)

            val_preds.append(probs[:, 0].cpu().numpy())
            val_targets.append(targets[:, 0].cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)

    # Load validation metadata (Trainer already loads it)
    val_df = trainer.val_df

    # Calculate Final Metric
    evaluator = JigsawEvaluator(val_targets, val_preds, val_df)
    final_score, metrics = evaluator.get_final_metric()

    # Print required metric format
    print(f"Final Validation Metric: {final_score}")
    print(f"Metrics Breakdown: {metrics}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(val_targets - val_preds)

    # Correlation with text length
    text_lengths = val_df[Config.TEXT_COL].astype(str).str.len()
    corr_len, _ = pearsonr(errors, text_lengths)
    print(f"Correlation (Error vs Text Length): {corr_len:.6f}")

    # Correlation with Identity Attributes
    print("Correlation (Error vs Identity Presence):")
    for identity_col in Config.IDENTITY_COLUMNS:
        if identity_col in val_df.columns:
            # Fill NaNs with 0 for correlation calculation
            id_values = val_df[identity_col].fillna(0.0)
            corr_id, _ = pearsonr(errors, id_values)
            print(f"  {identity_col}: {corr_id:.6f}")

    # 6. Submission
    THRESHOLD = 0.9368921743738194

    if final_score > THRESHOLD:
        print(
            f"\nMetric ({final_score}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.predict_and_submit()
    else:
        print(
            f"\nMetric ({final_score}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
