import sys
import os
import torch
import numpy as np
import pandas as pd

# Ensure library modules are accessible
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, get_device
from library.train import run_training
from library.inference import predict_test_set
from library.model import CalibratedHierarchicalSeqModel
from library.loss import WeightedMultiLabelLoss
from library.data import get_dataloaders


def main():
    # 1. Initialization and Configuration
    seed_everything(Config.SEED)
    device = get_device()

    # Fast Baseline Configuration
    # Reducing batch size to 2 to prevent OOM with the dense sequence length (96)
    # Increasing accumulation steps to maintain effective batch size
    EPOCHS = 5
    BATCH_SIZE = 2
    ACCUMULATION_STEPS = 8

    # 2. Training
    run_training(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        accumulation_steps=ACCUMULATION_STEPS,
        debug=False,
    )

    # 3. Validation & Metric Calculation
    print("Running Validation...")

    # Load Validation Data
    _, val_loader = get_dataloaders(load_cached_data=True)

    # Load Best Model
    model = CalibratedHierarchicalSeqModel(pretrained=False)
    model = model.to(device)

    checkpoint_path = Config.BEST_MODEL_PATH
    if not os.path.exists(checkpoint_path):
        checkpoint_path = Config.LAST_MODEL_PATH

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("Warning: No checkpoint found. Using random weights for validation.")

    model.eval()

    # Setup Loss Calculation
    # We need to compute the metric exactly as specified: averaged across all rows.
    # The WeightedMultiLabelLoss class computes the weighted average per batch.
    # We will accumulate the sum of weighted losses and divide by total samples.

    weights_list = Config.CLASS_WEIGHTS
    weights = torch.tensor(weights_list, dtype=torch.float32).to(device)
    weights = weights / weights.sum()  # Normalize to sum to 1.0

    bce_loss_none = torch.nn.BCEWithLogitsLoss(reduction="none")

    total_weighted_loss = 0.0
    total_samples = 0

    # Lists for Failure Analysis
    study_losses = []

    with torch.no_grad():
        for images, targets in val_loader:
            batch_size_curr = images.size(0)
            images = images.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)

            logits = model(images)

            # Calculate weighted loss per study
            # 1. BCE per label: (B, 8)
            raw_loss = bce_loss_none(logits, targets)

            # 2. Weighted BCE per label: (B, 8)
            weighted_raw_loss = raw_loss * weights

            # 3. Sum over labels to get loss per study: (B,)
            # Since weights sum to 1, this is the weighted average loss for the study
            loss_per_study = weighted_raw_loss.sum(dim=1)

            # Accumulate
            total_weighted_loss += loss_per_study.sum().item()
            total_samples += batch_size_curr

            # Store for analysis
            study_losses.extend(loss_per_study.cpu().numpy())

    final_metric = total_weighted_loss / total_samples if total_samples > 0 else 0.0

    # Print Metric exactly as requested
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("Performing Failure Analysis...")
    val_df = val_loader.dataset.df.copy()

    # Align losses with dataframe
    # The dataloader is not shuffled (shuffle=False in get_dataloaders for val), so order is preserved.
    if len(val_df) == len(study_losses):
        val_df["loss"] = study_losses

        # Correlation: Error vs Patient Overall (Class)
        if val_df["patient_overall"].std() > 0:
            corr_overall = np.corrcoef(val_df["loss"], val_df["patient_overall"])[0, 1]
            print(f"Correlation (Error vs Patient_Overall): {corr_overall:.6f}")
        else:
            print(
                "Correlation (Error vs Patient_Overall): Undefined (Single class in val)"
            )

        # Correlation: Error vs Fracture Count (Severity)
        fracture_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
        val_df["fracture_count"] = val_df[fracture_cols].sum(axis=1)

        if val_df["fracture_count"].std() > 0:
            corr_count = np.corrcoef(val_df["loss"], val_df["fracture_count"])[0, 1]
            print(f"Correlation (Error vs Fracture_Count): {corr_count:.6f}")
        else:
            print("Correlation (Error vs Fracture_Count): Undefined")

        # Correlation: Error vs Slice Count
        # Extract slice count from dataset's path cache
        try:
            study_paths = val_loader.dataset.study_paths
            val_df["slice_count"] = val_df["StudyInstanceUID"].apply(
                lambda x: len(study_paths.get(x, []))
            )
            if val_df["slice_count"].std() > 0:
                corr_slice = np.corrcoef(val_df["loss"], val_df["slice_count"])[0, 1]
                print(f"Correlation (Error vs Slice_Count): {corr_slice:.6f}")
        except Exception:
            pass

    else:
        print("Skipping failure analysis due to size mismatch.")

    # 5. Submission
    THRESHOLD = 0.1241588886
    if final_metric < THRESHOLD:
        print("Metric check passed. Generating submission...")
        predict_test_set(load_cached_data=True, batch_size=BATCH_SIZE, debug=False)
    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
