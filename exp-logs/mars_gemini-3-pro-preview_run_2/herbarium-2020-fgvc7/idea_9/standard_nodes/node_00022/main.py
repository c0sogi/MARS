import os
import sys
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import f1_score

# Add current directory to path to ensure library imports work correctly
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_logger, load_checkpoint
from library.data import get_dataloaders, get_test_dataloader
from library.model import HerbariumNet
from library.train import run_training_pipeline

# Initialize logger
logger = get_logger("runfile")


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline Execution
    # We use a subset of data and fewer epochs to ensure completion within 2 hours
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 30000

    # Reduce epochs for both phases
    Config.PHASE1["epochs"] = 2
    Config.PHASE2["epochs"] = 2

    logger.info("Configuration overrides applied for fast baseline.")
    logger.info(f"Debug Mode: {Config.DEBUG}, Subset Size: {Config.DEBUG_SUBSET_SIZE}")
    logger.info(
        f"Phase 1 Epochs: {Config.PHASE1['epochs']}, Phase 2 Epochs: {Config.PHASE2['epochs']}"
    )

    # -------------------------------------------------------------------------
    # 2. Training Pipeline
    # -------------------------------------------------------------------------
    logger.info("Starting End-to-End Training Pipeline...")
    # This runs Phase 1 (224x224) and Phase 2 (300x300) training
    run_training_pipeline()

    # -------------------------------------------------------------------------
    # 3. Model Loading for Evaluation
    # -------------------------------------------------------------------------
    device = torch.device(Config.DEVICE)
    model = HerbariumNet(pretrained=False)  # Load architecture
    model.to(device)

    # Attempt to load the best model from Phase 2
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        logger.warning("Best model not found. Checking for Phase 2 checkpoint...")
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "checkpoint_phase2.pth")

    if os.path.exists(checkpoint_path):
        logger.info(f"Loading model weights from {checkpoint_path}")
        load_checkpoint(checkpoint_path, model, device=Config.DEVICE)
    else:
        logger.error("No valid checkpoint found. Evaluation may be incorrect.")

    model.eval()

    # -------------------------------------------------------------------------
    # 4. Validation & Metric Calculation
    # -------------------------------------------------------------------------
    logger.info("Performing Validation Assessment...")

    # Get validation dataloader (Phase 2 settings: 300x300)
    # Note: We must respect the DEBUG flag to match the subset used/expected
    _, val_loader = get_dataloaders(
        Config.PHASE2["img_size"], Config.PHASE2["batch_size"], debug=Config.DEBUG
    )

    all_preds = []
    all_labels = []
    all_probs = []  # Store probability of the true class for failure analysis

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass (inference mode)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            # Get the probability assigned to the true class
            # gather expects index to have same dims as src except at dim
            true_probs = probs.gather(1, labels.view(-1, 1)).squeeze()

            # Predictions
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(true_probs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    # Calculate Macro F1
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"Final Validation Metric: {macro_f1}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    logger.info("Performing Failure Analysis...")

    # Error Magnitude = 1.0 - Probability(True Class)
    # High error means the model assigned low probability to the correct class
    errors = 1.0 - all_probs

    # Load Validation Metadata to correlate with features
    val_df = pd.read_csv(Config.VAL_CSV)

    # If using DEBUG/Subset, we must subset the dataframe exactly as the dataloader did
    if Config.DEBUG:
        val_df = val_df.sample(
            n=min(len(val_df), Config.DEBUG_SUBSET_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # Verify alignment
    if len(val_df) != len(errors):
        logger.warning(
            f"Metadata length ({len(val_df)}) mismatch with predictions ({len(errors)}). Skipping detailed correlation."
        )
    else:
        val_df["error_magnitude"] = errors
        val_df["true_label"] = all_labels

        # Correlation 1: Error vs Region ID
        # Checking if specific regions are harder to classify
        if "region_id" in val_df.columns:
            corr_region, _ = pearsonr(val_df["region_id"], val_df["error_magnitude"])
            print(f"Correlation between Error and Region ID: {corr_region}")

        # Correlation 2: Error vs Class Frequency (Training Set)
        # Checking if rare classes have higher error rates
        train_df = pd.read_csv(Config.TRAIN_CSV)
        class_counts = train_df["category_id"].value_counts().to_dict()

        # Map training frequency to validation samples
        val_df["train_freq"] = val_df["true_label"].map(class_counts).fillna(0)

        corr_freq, _ = pearsonr(val_df["train_freq"], val_df["error_magnitude"])
        print(f"Correlation between Error and Class Frequency: {corr_freq}")

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    threshold = 0.43008749389564027

    if macro_f1 > threshold:
        logger.info(
            f"Validation metric ({macro_f1}) exceeds threshold ({threshold}). Generating submission..."
        )

        # Get Test DataLoader
        test_loader = get_test_dataloader(
            Config.PHASE2["img_size"], Config.PHASE2["batch_size"]
        )

        test_ids = []
        test_preds = []

        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(device)

                # Forward
                logits = model(images)
                preds = torch.argmax(logits, dim=1)

                test_ids.extend(ids.numpy())
                test_preds.extend(preds.cpu().numpy())

        # Create Submission DataFrame
        sub_df = pd.DataFrame({"Id": test_ids, "Predicted": test_preds})

        # Save
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission file saved to {Config.SUBMISSION_PATH}")

    else:
        logger.info(
            f"Validation metric ({macro_f1}) did not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
