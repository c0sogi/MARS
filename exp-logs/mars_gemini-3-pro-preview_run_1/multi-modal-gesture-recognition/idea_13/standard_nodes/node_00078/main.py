import os
import sys
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config, set_seed
from library.data_loader import get_dataloaders
from library.engine import Trainer, evaluate
from library.utils import get_logger, levenshtein_distance


def main():
    # 1. Setup & Configuration
    # Set seed for reproducibility
    set_seed(Config.SEED)
    logger = get_logger(__name__)

    # Override Config for a Fast Baseline Execution
    # We limit epochs to ensure the run completes quickly within the time limit.
    # The dataset is small (~300 samples), so we use the full dataset but fewer epochs.
    Config.EPOCHS = 40
    Config.EARLY_STOPPING_PATIENCE = 10

    logger.info("Configuration configured for fast baseline.")

    # 2. Data Loading
    # Load dataloaders with caching enabled
    train_loader, val_loader, test_loader = get_dataloaders(debug=False)

    # 3. Model Training
    trainer = Trainer()
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # 4. Validation & Metrics
    # Load the best checkpoint saved during training
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
    else:
        logger.warning("No checkpoint found. Using last model state.")

    # Evaluate on the full validation set
    # Note: evaluate returns (avg_loss, lev_score, all_preds, all_sample_ids)
    # It does not return targets, so we will reconstruct them for failure analysis.
    val_loss, val_lev, val_preds, val_ids = evaluate(
        trainer.model, val_loader, trainer.criterion, trainer.device
    )

    # Print the required metric string
    print(f"Final Validation Metric: {val_lev}")

    # 5. Failure Analysis
    logger.info("Performing failure analysis...")

    # Reconstruct ground truth map from validation dataframe
    val_df = val_loader.dataset.df
    id_to_labels = {}
    id_to_length = {}

    for _, row in val_df.iterrows():
        sid = row["sample_id"]
        # Parse labels string to list of ints
        seq_str = str(row["labels"]) if pd.notna(row["labels"]) else ""
        if seq_str and seq_str.lower() != "nan":
            labels = [int(x) for x in seq_str.split(",")]
        else:
            labels = []

        id_to_labels[sid] = labels
        id_to_length[sid] = row["num_frames"]

    # Calculate per-sample Levenshtein distance and collect features
    errors = []
    lengths = []

    for pid, pred_seq in zip(val_ids, val_preds):
        target_seq = id_to_labels.get(pid, [])

        # Calculate raw Levenshtein distance for this sample
        dist = levenshtein_distance(pred_seq, target_seq)

        errors.append(dist)
        lengths.append(id_to_length.get(pid, 0))

    # Calculate correlation
    if len(errors) > 1:
        corr, _ = pearsonr(errors, lengths)
        print(f"Correlation (Error vs Sequence Length): {corr:.4f}")
    else:
        print("Insufficient data for correlation analysis.")

    # 6. Submission Generation
    # Threshold defined in the task description
    THRESHOLD = 0.0824829931972789

    if val_lev < THRESHOLD:
        logger.info(
            f"Validation metric {val_lev} < {THRESHOLD}. Generating submission..."
        )
        trainer.predict(test_loader, best_model_path)
    else:
        logger.info(f"Validation metric {val_lev} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
