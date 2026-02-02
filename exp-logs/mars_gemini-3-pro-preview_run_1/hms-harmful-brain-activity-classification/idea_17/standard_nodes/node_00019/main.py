import sys
import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import get_dataloaders
from library.models import HarmfulBrainActivityModel
from library.engine import fit, validate


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.make_dirs()
    logger = get_logger("runfile")
    logger.info("Starting runfile execution...")

    # 2. Data Loading
    logger.info("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # 3. Model Initialization
    logger.info("Initializing Model...")
    model = HarmfulBrainActivityModel(config=Config)
    model.to(Config.DEVICE)

    # 4. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
    )

    # 5. Training Loop
    logger.info("Starting Training...")
    fit(model, train_loader, val_loader, optimizer, scheduler, Config)

    # 6. Final Validation & Failure Analysis
    logger.info("Loading best model for validation...")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
    else:
        logger.warning("Best model checkpoint not found. Using current model state.")

    # Run validation
    val_loss, val_metric, val_preds = validate(model, val_loader, Config.DEVICE)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {val_metric}")

    # Failure Analysis
    logger.info("Performing Failure Analysis...")

    # Retrieve targets from validation dataset
    # Note: val_loader is not shuffled, so order matches
    val_df = val_loader.dataset.df
    target_cols = [c for c in val_df.columns if c.endswith("_prob")]
    y_true = val_df[target_cols].values

    # Calculate per-sample KL Divergence
    epsilon = 1e-15
    y_pred_clipped = np.clip(val_preds, epsilon, 1 - epsilon)

    # Term 1: sum(y_true * log(y_true))
    term1 = np.zeros_like(y_true)
    mask = y_true > 0
    term1[mask] = y_true[mask] * np.log(y_true[mask])

    # Term 2: sum(y_true * log(y_pred))
    term2 = y_true * np.log(y_pred_clipped)

    # KL per sample
    kl_per_sample = np.sum(term1 - term2, axis=1)

    # Correlation Analysis
    analysis_df = val_df.copy()
    analysis_df["error_magnitude"] = kl_per_sample

    features_to_check = ["eeg_label_offset_seconds", "spectrogram_label_offset_seconds"]

    for feature in features_to_check:
        if feature in analysis_df.columns:
            corr = analysis_df[feature].corr(analysis_df["error_magnitude"])
            print(f"Correlation between {feature} and Error: {corr}")

    # 7. Submission Generation
    THRESHOLD = 0.6822116374969482

    if val_metric < THRESHOLD:
        logger.info(
            f"Validation metric {val_metric} < {THRESHOLD}. Generating submission..."
        )

        model.eval()
        test_preds_list = []

        # Inference Loop
        with torch.no_grad():
            for batch in test_loader:
                eeg = batch["eeg"].to(Config.DEVICE)
                spec = batch["spec"].to(Config.DEVICE)

                logits = model(eeg, spec)
                probs = torch.nn.functional.softmax(logits, dim=1)
                test_preds_list.append(probs.cpu().numpy())

        # Concatenate predictions
        test_preds_arr = np.concatenate(test_preds_list, axis=0)

        # Prepare Submission DataFrame
        test_df = test_loader.dataset.df
        submission_df = pd.DataFrame()
        submission_df["eeg_id"] = test_df["eeg_id"]

        # Assign probabilities to vote columns
        # Config.CLASS_NAMES order: seizure, lpd, gpd, lrda, grda, other
        submission_df["seizure_vote"] = test_preds_arr[:, 0]
        submission_df["lpd_vote"] = test_preds_arr[:, 1]
        submission_df["gpd_vote"] = test_preds_arr[:, 2]
        submission_df["lrda_vote"] = test_preds_arr[:, 3]
        submission_df["grda_vote"] = test_preds_arr[:, 4]
        submission_df["other_vote"] = test_preds_arr[:, 5]

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission successfully saved to {Config.SUBMISSION_PATH}")

    else:
        logger.info(
            f"Validation metric {val_metric} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
