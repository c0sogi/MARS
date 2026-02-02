import os
import sys
import torch
import numpy as np
import pandas as pd
from library import config, dataset, model, utils, train, inference


def main():
    # --- 1. Setup ---
    # Set seed for reproducibility
    utils.set_seed(config.SEED)

    # Modify configuration for the task requirements
    # Limit training epochs for a fast baseline execution
    config.NUM_EPOCHS = 10

    # Set submission path as per Task instructions
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    config.SUBMISSION_PATH = os.path.join(submission_dir, "submission.csv")

    # --- 2. Training ---
    # Train the model using the provided library function
    # load_cached_data=True utilizes pre-processed .npy files if available to speed up loading
    train.run_training(num_epochs=config.NUM_EPOCHS, load_cached_data=True)

    # --- 3. Validation & Failure Analysis ---
    print("Starting validation analysis...")

    # Load the best model checkpoint
    net = model.HDNet().to(config.DEVICE)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    net.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    net.eval()

    # Get validation dataloader
    _, val_loader, _ = dataset.get_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        load_cached_data=True,
    )

    # Containers for analysis
    all_probs = []
    all_labels = []
    error_records = []

    # Access metadata directly to correlate errors with features
    # val_loader is not shuffled (shuffle=False in dataset.py), so index correspondence is maintained
    val_metadata = val_loader.dataset.metadata
    current_idx = 0

    with torch.no_grad():
        for volumes, labels in val_loader:
            volumes = volumes.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            # Inference
            outputs = net(volumes)
            probs = torch.sigmoid(outputs)

            # Store for global metric calculation
            # Move to CPU to save GPU memory
            probs_np = probs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            volumes_np = volumes.cpu().numpy()

            all_probs.append(probs_np)
            all_labels.append(labels_np)

            # --- Failure Analysis: Per-sample stats ---
            batch_size = volumes.size(0)

            # Calculate Mean Absolute Error (MAE) per patch
            # Shape: (B, 1, H, W) -> (B,)
            mae = np.mean(np.abs(probs_np - labels_np), axis=(1, 2, 3))

            # Calculate simple input features (Mean Intensity, Std Intensity)
            # Shape: (B, 65, H, W) -> (B,)
            mean_intensity = np.mean(volumes_np, axis=(1, 2, 3))
            std_intensity = np.std(volumes_np, axis=(1, 2, 3))

            # Retrieve spatial metadata
            batch_meta = val_metadata.iloc[current_idx : current_idx + batch_size]

            for i in range(batch_size):
                record = {
                    "error_magnitude": mae[i],
                    "mean_intensity": mean_intensity[i],
                    "std_intensity": std_intensity[i],
                    "x": batch_meta.iloc[i]["x"],
                    "y": batch_meta.iloc[i]["y"],
                }
                error_records.append(record)

            current_idx += batch_size

    # --- 4. Metric Calculation ---
    # Concatenate all batches
    y_true_val = np.concatenate(all_labels).flatten()
    y_probs_val = np.concatenate(all_probs).flatten()

    # Find the best F0.5 score by optimizing threshold
    # We implement the search here to capture the exact score value for printing
    thresholds = np.linspace(0.01, 0.99, 100)
    best_score = 0.0

    for thr in thresholds:
        y_pred_bin = (y_probs_val >= thr).astype(np.uint8)
        score = utils.calculate_fbeta(y_true_val, y_pred_bin, beta=0.5)
        if score > best_score:
            best_score = score

    # Print required metric
    print(f"Final Validation Metric: {best_score}")

    # --- 5. Failure Analysis Output ---
    if error_records:
        df_errors = pd.DataFrame(error_records)
        # Calculate correlation matrix
        correlations = df_errors.corr()["error_magnitude"].drop("error_magnitude")
        print("\nFailure Analysis - Correlation with Error Magnitude:")
        print(correlations)

    # --- 6. Conditional Submission ---
    TARGET_METRIC = 0.41758

    if best_score > TARGET_METRIC:
        print(f"\nMetric {best_score} > {TARGET_METRIC}. Generating submission...")
        # inference.generate_submission uses the config.SUBMISSION_PATH we set earlier
        inference.generate_submission(load_cached_data=True)
    else:
        print(f"\nMetric {best_score} <= {TARGET_METRIC}. Submission skipped.")


if __name__ == "__main__":
    main()
