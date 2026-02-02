import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, calculate_auc
from library.data_loader import get_dataloaders
from library.trainer import Trainer

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    # Load metadata dataframes
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    # Initialize DataLoaders (handles caching of peak intensity indices)
    # debug=False ensures we use the full provided dataset (which is small enough)
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, debug=False
    )

    # --------------------------------------------------------------------------
    # 3. Model Training
    # --------------------------------------------------------------------------
    trainer = Trainer()

    best_auc = -1.0

    # Explicit training loop
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        # Train one epoch
        train_loss = trainer.train_one_epoch(train_loader, epoch, Config.NUM_EPOCHS)

        # Validate one epoch
        val_loss, val_auc = trainer.validate_one_epoch(val_loader)

        # Checkpoint logic
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(trainer.model.state_dict(), Config.CHECKPOINT_PATH)

    # --------------------------------------------------------------------------
    # 4. Final Evaluation & Failure Analysis
    # --------------------------------------------------------------------------
    # Load the best model for analysis
    if os.path.exists(Config.CHECKPOINT_PATH):
        trainer.model.load_state_dict(
            torch.load(Config.CHECKPOINT_PATH, map_location=trainer.device)
        )

    trainer.model.eval()

    # Collect predictions and targets for the validation set
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for view_a, targets in val_loader:
            view_a = view_a.to(trainer.device)

            # Forward pass
            logits = trainer.model(view_a)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(targets.numpy().flatten())

    # Calculate Final Metric
    final_auc = calculate_auc(all_targets, all_preds)
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis
    # Access the dataframe from the dataset to get metadata features
    val_ds_df = val_loader.dataset.df.copy()

    # Ensure lengths match (val_loader is not shuffled, so direct assignment works)
    val_ds_df["pred"] = all_preds
    val_ds_df["target"] = all_targets
    val_ds_df["error"] = (val_ds_df["target"] - val_ds_df["pred"]).abs()

    print("Failure Analysis (Correlation with Error):")
    features_to_analyze = ["flair_slice_count", "flair_peak_index"]

    for feat in features_to_analyze:
        if feat in val_ds_df.columns:
            # Calculate Pearson correlation
            corr = val_ds_df["error"].corr(val_ds_df[feat])
            print(f"Correlation between Error and {feat}: {corr}")

    # --------------------------------------------------------------------------
    # 5. Conditional Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.6254545454545455

    if final_auc > THRESHOLD:
        trainer.predict_and_submit(test_loader)
    else:
        print(
            f"Validation metric {final_auc} is not greater than {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
