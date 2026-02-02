import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from scipy.stats import pearsonr

# Ensure local library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, weighted_log_loss
from library.dataset import CervicalSpineDataset
from library.model import CervicalFractureNet
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Create submission directory as per instructions
    os.makedirs("./submission", exist_ok=True)

    # Override Config for Fast Baseline if needed
    # The dataset is small (161 train), so 10 epochs is very fast (~10 mins)
    # We keep the defaults from Config.py but ensure debug is False
    Config.setup(debug=False)

    print(f"Running Fast Baseline with {Config.EPOCHS} epochs...")

    # Purge stale artifacts to prevent configuration ghosting (Cite debug_lesson_11)
    if os.path.exists(Config.MODEL_PATH):
        os.remove(Config.MODEL_PATH)

    # 2. Training
    trainer = Trainer()
    trainer.fit()

    # 3. Validation & Metric Calculation
    print("Loading best model for validation...")
    device = torch.device(Config.DEVICE)
    model = CervicalFractureNet()
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    val_dataset = CervicalSpineDataset(
        metadata_path=Config.VAL_METADATA, mode="val", load_cached_data=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    all_preds = []
    all_targets = []
    study_uids = val_dataset.df["StudyInstanceUID"].tolist()

    print("Performing validation inference...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            with autocast():
                logits = model(images)
                probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute Metric
    final_metric = weighted_log_loss(all_targets, all_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate loss per study
    # Weights: C1-C7 = 1.0, patient_overall = 7.0
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0])
    epsilon = 1e-15
    y_pred_clipped = np.clip(all_preds, epsilon, 1 - epsilon)

    # BCE per column: -(y log(p) + (1-y) log(1-p))
    bce_matrix = -(
        all_targets * np.log(y_pred_clipped)
        + (1 - all_targets) * np.log(1 - y_pred_clipped)
    )

    # Weighted average per row (study)
    study_losses = np.average(bce_matrix, axis=1, weights=weights)

    # Get slice counts for correlation analysis
    # Access the cached path map from the dataset
    slice_counts = []
    for uid in study_uids:
        files = val_dataset.path_map.get(uid, [])
        slice_counts.append(len(files))

    # Calculate correlation
    if len(study_losses) > 1:
        corr, _ = pearsonr(study_losses, slice_counts)
        print(f"Correlation between Error (Loss) and Slice Count: {corr:.4f}")
    else:
        print("Not enough samples for correlation analysis.")

    # 5. Submission Generation
    # Condition: Generate only if metric < 0.15364714496434773
    THRESHOLD = 0.15364714496434773

    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) < Threshold ({THRESHOLD}). Generating submission..."
        )

        test_dataset = CervicalSpineDataset(
            metadata_path=Config.TEST_METADATA, mode="test", load_cached_data=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )

        submission_rows = []
        target_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

        with torch.no_grad():
            for images, study_ids in test_loader:
                images = images.to(device)
                with autocast():
                    logits = model(images)
                    probs = torch.sigmoid(logits)

                probs_np = probs.cpu().numpy()

                for i, study_id in enumerate(study_ids):
                    study_probs = probs_np[i]
                    for col_idx, col_name in enumerate(target_cols):
                        row_id = f"{study_id}_{col_name}"
                        prob = study_probs[col_idx]
                        submission_rows.append({"row_id": row_id, "fractured": prob})

        submission_df = pd.DataFrame(submission_rows)
        save_path = "./submission/submission.csv"
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nMetric ({final_metric}) >= Threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
