import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import provided library components
from library.config import Config
from library.train import run_training
from library.inference import predict_submission
from library.data import get_dataloader
from library.model import AsymmetricEfficientNet


def main():
    # Set seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    print("=== Starting Runfile Execution ===")

    # 1. Training Phase
    # We use the full dataset (debug=False) to maximize performance.
    # The dataset is small (~500 subjects), so 10 epochs will run quickly (minutes).
    print("\n--- Initiating Training ---")
    run_training(epochs=10, batch_size=Config.BATCH_SIZE, debug=False)

    # 2. Validation & Failure Analysis Phase
    print("\n--- Initiating Validation & Failure Analysis ---")

    # Load validation metadata
    if not os.path.exists(Config.VAL_METADATA):
        print("Error: Validation metadata not found.")
        return

    val_df = pd.read_csv(Config.VAL_METADATA)

    # Create validation loader (shuffle=False to align with dataframe for analysis)
    val_loader = get_dataloader(
        val_df, phase="val", batch_size=Config.BATCH_SIZE, shuffle=False
    )

    # Load the best model
    device = Config.DEVICE
    model = AsymmetricEfficientNet(num_classes=1)
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print("Error: Model file not found.")
        return

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Inference on Validation Set
    all_preds = []
    all_targets = []
    anchor_indices = []

    # Access dataset to retrieve cached anchors for analysis
    dataset = val_loader.dataset

    print("Running inference on validation set...")
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = images.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            targets_np = targets.numpy().flatten()

            all_preds.extend(probs)
            all_targets.extend(targets_np)

            # Extract metadata features for failure analysis
            # Since shuffle=False, we can map batch indices to dataframe rows
            start_idx = i * val_loader.batch_size
            current_batch_size = images.size(0)
            batch_indices = range(start_idx, start_idx + current_batch_size)

            for idx in batch_indices:
                row = val_df.iloc[idx]
                subject_id = row["BraTS21ID"]
                # Get the anchor index used for this subject
                anchor = dataset.roi_anchors.get(subject_id, 0)
                anchor_indices.append(anchor)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    anchor_indices = np.array(anchor_indices)

    # Calculate Final Metric
    # Handle edge case where validation set might have only one class (unlikely but possible in small debug runs)
    if len(np.unique(all_targets)) > 1:
        final_metric = roc_auc_score(all_targets, all_preds)
    else:
        final_metric = 0.5

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    errors = np.abs(all_preds - all_targets)

    # 1. Correlation with Target (Systematic bias towards 0 or 1?)
    if np.std(all_targets) > 0 and np.std(errors) > 0:
        corr_target = np.corrcoef(errors, all_targets)[0, 1]
        print(f"Correlation between Error and Target Label: {corr_target:.10f}")
    else:
        print(
            "Correlation between Error and Target Label: Undefined (Variance is zero)"
        )

    # 2. Correlation with Anchor Index (Does tumor location in Z-axis affect error?)
    if np.std(anchor_indices) > 0 and np.std(errors) > 0:
        corr_anchor = np.corrcoef(errors, anchor_indices)[0, 1]
        print(f"Correlation between Error and Anchor Index: {corr_anchor:.10f}")
    else:
        print(
            "Correlation between Error and Anchor Index: Undefined (Variance is zero)"
        )

    # 3. Submission Phase
    threshold = 0.6321818181818182
    print("\n--- Submission Check ---")
    print(f"Threshold: {threshold}")
    print(f"Achieved:  {final_metric}")

    if final_metric > threshold:
        print("Threshold met. Generating submission...")
        predict_submission(
            model_path=Config.MODEL_SAVE_PATH,
            output_path=Config.SUBMISSION_FILE,
            batch_size=Config.BATCH_SIZE,
            device=device,
        )
    else:
        print("Threshold not met. Submission skipped.")


if __name__ == "__main__":
    main()
