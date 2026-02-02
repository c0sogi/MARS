import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# 1. Configuration Override for Fast Baseline
# We must set this before importing library.train to ensure it picks up the modified value
import library.config

library.config.EPOCHS = 5  # Reduce epochs for fast execution

# Now import the rest of the library components
from library.config import (
    DEVICE,
    IDEA_DIR,
    SUBMISSION_PATH,
    N_FOLDS,
    NUM_WORKERS,
    BATCH_SIZE,
    SEED,
)
from library.train import run_kfold_training
from library.dataset import BraTSDataset, get_transforms
from library.model import CASIVNet
from library.utils import seed_everything


def main():
    # Set seed for reproducibility
    seed_everything(SEED)

    print("=== Starting Fast Baseline Pipeline ===")

    # 2. Run Training
    # This trains the models using K-Fold CV and generates a submission on the test set
    print("Running K-Fold Training...")
    run_kfold_training()

    # 3. Hold-out Validation
    # We need to explicitly evaluate on the 'val' split which was not used in K-Fold training
    print("Running Hold-out Validation...")

    # Load Validation Dataset
    val_ds = BraTSDataset(split="val", transform=get_transforms("val"))
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Load Trained Models
    models = []
    for fold in range(N_FOLDS):
        checkpoint_path = os.path.join(IDEA_DIR, f"best_model_fold{fold}.pth")
        if os.path.exists(checkpoint_path):
            try:
                model = CASIVNet()
                model.to(DEVICE)
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                # Load state dict
                if "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                model.eval()
                models.append(model)
            except Exception as e:
                print(f"Error loading fold {fold}: {e}")

    if not models:
        print("No models loaded. Aborting validation.")
        return

    # Inference Loop
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)

            # Ensemble Prediction
            batch_preds = []
            for model in models:
                logits = model(images)
                probs = torch.sigmoid(logits)
                batch_preds.append(probs.cpu().numpy())

            # Average probabilities across folds
            avg_preds = np.mean(batch_preds, axis=0)

            all_preds.extend(avg_preds.flatten())
            all_targets.extend(labels.numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 4. Metric Calculation
    try:
        val_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        val_auc = 0.5

    print(f"Final Validation Metric: {val_auc}")

    # 5. Failure Analysis
    print("Performing Failure Analysis...")
    errors = np.abs(all_preds - all_targets)

    # Access metadata features from the dataset's dataframe
    # The dataframe contains stats like 'flair_count', 'flair_depth', etc.
    df_val = val_ds.df

    # Define features to analyze
    features_to_check = [
        "flair_count",
        "t1wce_count",
        "t2w_count",
        "flair_depth",
        "t1wce_depth",
        "t2w_depth",
        "flair_avg_size_bytes",
        "t1wce_avg_size_bytes",
        "t2w_avg_size_bytes",
    ]

    print("Correlation between Error Magnitude and Metadata Features:")
    for feature in features_to_check:
        if feature in df_val.columns:
            # Calculate Pearson correlation
            # Handle potential NaNs or constant values
            feat_values = df_val[feature].values
            if len(np.unique(feat_values)) > 1:
                corr = np.corrcoef(errors, feat_values)[0, 1]
                print(f"{feature}: {corr:.4f}")
            else:
                print(f"{feature}: N/A (Constant)")

    # 6. Conditional Submission
    # Threshold defined in task
    THRESHOLD = 0.6705454545454544

    if val_auc > THRESHOLD:
        print(
            f"Validation metric ({val_auc}) exceeds threshold ({THRESHOLD}). Keeping submission."
        )
        # Ensure submission exists (run_kfold_training should have created it)
        if not os.path.exists(SUBMISSION_PATH):
            print("Submission file missing. Regenerating...")
            from library.inference import predict_test_set

            predict_test_set()
    else:
        print(
            f"Validation metric ({val_auc}) does not exceed threshold ({THRESHOLD}). Discarding submission."
        )
        if os.path.exists(SUBMISSION_PATH):
            os.remove(SUBMISSION_PATH)
            print("Submission file removed.")


if __name__ == "__main__":
    main()
