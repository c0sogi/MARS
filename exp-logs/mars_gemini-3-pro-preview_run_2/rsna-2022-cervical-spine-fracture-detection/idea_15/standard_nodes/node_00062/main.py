import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.train import run_training
from library.model import CalibratedSequenceNetwork
from library.utils import load_checkpoint, seed_everything
from library.data import get_dataloaders, get_test_loader


def main():
    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)

    # 2. Train the Model (Fast Baseline)
    # The run_training function handles the training loop, saving the best model
    # to Config.WORKING_DIR/best_model.pth based on validation loss.
    print("Starting training...")
    run_training(debug=False)

    # 3. Validation & Metric Calculation
    print("Starting validation...")
    device = Config.DEVICE

    # Initialize model and load the best checkpoint
    model = CalibratedSequenceNetwork()
    model.to(device)

    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    load_checkpoint(model, checkpoint_path, device=device)
    model.eval()

    # Load validation data (using cached paths for speed)
    _, val_loader = get_dataloaders(load_cached_data=True, debug=False)

    all_probs = []
    all_targets = []

    # Inference loop
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    # Calculate Weighted Multi-Label Logarithmic Loss
    # Weights: [1, 1, 1, 1, 1, 1, 1, 7] corresponding to C1..C7, patient_overall
    weights = np.array(Config.LOSS_WEIGHTS)

    # Clip probabilities for numerical stability (avoid log(0))
    epsilon = 1e-15
    all_probs = np.clip(all_probs, epsilon, 1 - epsilon)

    # Compute weighted log loss element-wise
    # Formula: -w * [y * log(p) + (1-y) * log(1-p)]
    loss_matrix = -weights * (
        all_targets * np.log(all_probs) + (1 - all_targets) * np.log(1 - all_probs)
    )

    # The final metric is the average across all rows (samples * classes)
    final_metric = np.mean(loss_matrix)

    # Print the required metric
    print(f"Final Validation Metric: {final_metric:.10f}")

    # 4. Failure Analysis
    # Analyze correlation between error magnitude and input feature (slice count)

    # Calculate error magnitude per study (mean loss across the 8 targets)
    study_errors = np.mean(loss_matrix, axis=1)

    # Access metadata to retrieve slice counts
    val_dataset = val_loader.dataset
    val_meta = val_dataset.metadata_df.copy()

    # Ensure alignment (val_loader is not shuffled, so order matches metadata)
    if len(val_meta) == len(study_errors):
        val_meta["error"] = study_errors

        # Retrieve slice counts from the dataset's file map
        slice_counts = []
        for uid in val_meta["StudyInstanceUID"]:
            if uid in val_dataset.study_file_map:
                slice_counts.append(len(val_dataset.study_file_map[uid]))
            else:
                slice_counts.append(0)

        val_meta["slice_count"] = slice_counts

        # Calculate Pearson correlation
        if val_meta["slice_count"].std() > 0:
            corr = val_meta["error"].corr(val_meta["slice_count"])
            print(f"Correlation between error magnitude and slice count: {corr:.4f}")
        else:
            print(
                "Correlation between error magnitude and slice count: NaN (Constant slice count)"
            )
    else:
        print("Warning: Metadata length mismatch. Skipping failure analysis.")

    # 5. Submission Generation
    # Only generate submission if the model meets the performance threshold
    THRESHOLD = 0.1241588886

    if final_metric < THRESHOLD:
        print("Metric check passed. Generating submission...")

        test_loader = get_test_loader(load_cached_data=True, debug=False)
        submission_rows = []

        with torch.no_grad():
            for images, study_ids in test_loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()

                for i, study_id in enumerate(study_ids):
                    # Get probabilities for this study (shape: 8,)
                    p_study = probs[i]

                    # Create a row for each target column
                    for j, col in enumerate(Config.TARGET_COLUMNS):
                        row_id = f"{study_id}_{col}"
                        score = float(p_study[j])
                        submission_rows.append({"row_id": row_id, "fractured": score})

        # Save submission file
        submission_df = pd.DataFrame(submission_rows)

        sub_dir = "./submission"
        os.makedirs(sub_dir, exist_ok=True)
        sub_path = os.path.join(sub_dir, "submission.csv")

        submission_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"Metric {final_metric:.10f} is not lower than threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
