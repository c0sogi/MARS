import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.dataset import RSNADataset
from library.model import FractureMILModel
from library.loss import WeightedLogLoss
from library.train import Trainer
from library.inference import generate_submission


def main():
    # --- 1. Setup & Configuration ---
    print("=== Starting Pipeline ===")
    Config.seed_everything(Config.SEED)

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # --- 2. Training ---
    print("\n[Step 1/4] Training Model...")
    # Initialize Trainer with default config
    # The dataset is small (~160 training samples), so 10 epochs (default) is fast.
    trainer = Trainer(config=Config)

    # Run training loop
    trainer.fit()

    # --- 3. Validation Assessment ---
    print("\n[Step 2/4] Validating Model...")

    device = torch.device(Config.DEVICE)

    # Initialize model architecture
    model = FractureMILModel(pretrained=False)
    model.to(device)

    # Load the best model weights saved by the trainer
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model weights from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model not found. Using current model state.")
        model = trainer.model

    model.eval()

    # Load Validation Data
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    val_dataset = RSNADataset(val_df, Config, split="val", load_cached_paths=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Containers for metric calculation
    all_preds = []
    all_targets = []
    study_uids = []

    # Inference Loop on Validation Set
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            all_preds.append(outputs)
            all_targets.append(labels)

            # Track UIDs for failure analysis mapping
            start_idx = i * Config.BATCH_SIZE
            end_idx = start_idx + images.size(0)
            batch_uids = val_df.iloc[start_idx:end_idx]["StudyInstanceUID"].tolist()
            study_uids.extend(batch_uids)

    # Concatenate results
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute Final Metric
    criterion = WeightedLogLoss()
    # Ensure criterion buffers are on the same device
    criterion = criterion.to(device)

    validation_loss = criterion(all_preds, all_targets)
    print(f"Final Validation Metric: {validation_loss.item()}")

    # --- 4. Failure Analysis ---
    print("\n[Step 3/4] Performing Failure Analysis...")

    # Calculate loss per study to correlate with features
    # We re-implement the loss logic element-wise without reduction
    # Weights: 1.0 for vertebrae (first 7), 7.0 for overall (last 1)
    class_weights = torch.tensor([1.0] * 7 + [7.0], device=device)

    # Clamp predictions for numerical stability
    epsilon = 1e-7
    preds_clamped = torch.clamp(all_preds, epsilon, 1.0 - epsilon)

    # Compute weighted BCE per element
    bce_loss = -(
        all_targets * torch.log(preds_clamped)
        + (1.0 - all_targets) * torch.log(1.0 - preds_clamped)
    )
    weighted_bce = bce_loss * class_weights

    # Average loss per study (row-wise mean across the 8 classes)
    study_losses = weighted_bce.mean(dim=1).cpu().numpy()

    # Create Analysis DataFrame
    analysis_df = pd.DataFrame({"StudyInstanceUID": study_uids, "loss": study_losses})

    # Merge with ground truth metadata to get features
    full_analysis = pd.merge(analysis_df, val_df, on="StudyInstanceUID")

    # Feature 1: Fracture Count (Sum of C1-C7 labels)
    fracture_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    full_analysis["fracture_count"] = full_analysis[fracture_cols].sum(axis=1)

    # Compute Correlations
    # Correlation between Error (Loss) and Complexity (Fracture Count)
    corr_count = full_analysis["loss"].corr(full_analysis["fracture_count"])
    # Correlation between Error (Loss) and Target Class (Patient Overall)
    corr_overall = full_analysis["loss"].corr(full_analysis["patient_overall"])

    print(f"Correlation (Loss vs Fracture Count): {corr_count:.4f}")
    print(f"Correlation (Loss vs Patient Overall): {corr_overall:.4f}")

    # --- 5. Submission ---
    print("\n[Step 4/4] Checking Submission Criteria...")

    # Threshold from previous best run
    BASELINE_METRIC = 0.8661673665046692

    if validation_loss.item() < BASELINE_METRIC:
        print(
            f"Validation metric ({validation_loss.item():.4f}) improved over baseline ({BASELINE_METRIC:.4f})."
        )
        print("Generating Submission...")
        # Generate predictions for the test set
        generate_submission(config=Config, load_cached_data=True)
    else:
        print(
            f"Validation metric ({validation_loss.item():.4f}) did not improve over baseline ({BASELINE_METRIC:.4f})."
        )
        print("Skipping submission generation.")

    print("=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
