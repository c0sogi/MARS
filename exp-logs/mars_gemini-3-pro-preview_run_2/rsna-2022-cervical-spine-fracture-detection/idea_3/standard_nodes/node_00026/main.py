import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import warnings

# Import provided library modules
from library.config import Config
from library.trainer import Trainer
from library.utils import seed_everything, load_checkpoint, get_device
from library.losses import WeightedMultiLabelLoss

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def count_slices(study_id, root_dir):
    """
    Counts the number of .dcm files in a study directory.
    Used for failure analysis feature extraction.
    """
    study_path = os.path.join(root_dir, study_id)
    if not os.path.exists(study_path):
        return 0
    try:
        # Fast count of files
        return len([name for name in os.listdir(study_path) if name.endswith(".dcm")])
    except OSError:
        return 0


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override epochs for a fast baseline execution
    Config.EPOCHS = 5

    # Setup directories
    Config.setup()

    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    device = get_device()
    print(f"Running on device: {device}")

    # =========================================================================
    # 2. Training
    # =========================================================================
    print("\n=== Starting Training ===")
    trainer = Trainer(load_cached_data=True)
    trainer.fit()

    # =========================================================================
    # 3. Validation Assessment
    # =========================================================================
    print("\n=== Performing Validation Assessment ===")

    # Load the best model saved during training
    # Note: Trainer.fit() saves the best model to Config.MODEL_CHECKPOINT_PATH
    if os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        load_checkpoint(
            trainer.model, filename=Config.MODEL_CHECKPOINT_PATH, device=device
        )
    else:
        print("Warning: No checkpoint found. Using current model weights.")

    trainer.model.eval()

    # Initialize loss function for metric calculation
    # Using the same configuration as training
    criterion = WeightedMultiLabelLoss(
        pos_weight_value=Config.POS_WEIGHT, class_weights=Config.CLASS_WEIGHTS
    ).to(device)

    val_loader = trainer.val_loader
    total_loss = 0.0
    total_samples = 0

    # Storage for failure analysis
    study_ids_list = []
    sample_losses_list = []
    slice_counts_list = []

    # Pos weight tensor for manual per-sample loss calculation
    pos_weight_tensor = torch.full((Config.NUM_CLASSES,), Config.POS_WEIGHT).to(device)

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            batch_study_ids = batch["study_id"]

            # Forward Pass
            logits = trainer.model(images)

            # 1. Calculate Aggregate Metric
            # criterion returns mean loss over the batch
            batch_loss = criterion(logits, targets)
            batch_size = images.size(0)
            total_loss += batch_loss.item() * batch_size
            total_samples += batch_size

            # 2. Calculate Per-Sample Loss for Failure Analysis
            # We compute BCE with reduction='none' to get loss per element
            raw_loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight_tensor, reduction="none"
            )
            # Average over classes (dim 1) to get scalar loss per study
            per_sample_loss = raw_loss.mean(dim=1).cpu().numpy()

            # Store data
            study_ids_list.extend(batch_study_ids)
            sample_losses_list.extend(per_sample_loss)

            # Extract metadata (Slice Count)
            for sid in batch_study_ids:
                # Validation images are located in the train_images directory
                cnt = count_slices(sid, Config.TRAIN_IMAGES_DIR)
                slice_counts_list.append(cnt)

    # Compute Final Metric
    final_metric = total_loss / total_samples if total_samples > 0 else 0.0
    print(f"Final Validation Metric: {final_metric}")

    # =========================================================================
    # 4. Failure Analysis
    # =========================================================================
    print("\n=== Performing Failure Analysis ===")

    if len(sample_losses_list) > 0:
        analysis_df = pd.DataFrame(
            {
                "StudyInstanceUID": study_ids_list,
                "Loss": sample_losses_list,
                "SliceCount": slice_counts_list,
            }
        )

        # Calculate Correlation
        if analysis_df["SliceCount"].std() > 0:
            correlation = analysis_df["Loss"].corr(analysis_df["SliceCount"])
            print(
                f"Correlation between Error Magnitude and Slice Count: {correlation:.4f}"
            )
        else:
            print(
                "Correlation between Error Magnitude and Slice Count: Undefined (Constant Feature)"
            )

        print("\nTop 5 Worst Predictions (Highest Loss):")
        print(analysis_df.sort_values("Loss", ascending=False).head(5))
    else:
        print("No validation samples found for analysis.")

    # =========================================================================
    # 5. Submission Generation
    # =========================================================================
    print("\n=== Submission Check ===")

    THRESHOLD = 0.38122559812935913

    if final_metric < THRESHOLD:
        print(
            f"Metric ({final_metric:.6f}) < Threshold ({THRESHOLD}). Generating submission..."
        )
        trainer.generate_submission()
    else:
        print(
            f"Metric ({final_metric:.6f}) >= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
