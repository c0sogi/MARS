import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from library import config, utils, data, model as model_lib, train


def main():
    # 1. Setup
    utils.set_seed(config.SEED)
    device = utils.get_device()

    print("Starting 5-Fold Cross-Validation Pipeline...")

    # Storage for Out-Of-Fold (OOF) data for global metric calculation and failure analysis
    oof_preds = []
    oof_targets = []
    oof_errors = []
    oof_angles = []
    oof_img_means = []

    # 2. Training Loop
    for fold in range(config.NUM_FOLDS):
        # Train the fold
        # Limiting epochs to 40 for a fast baseline execution as requested.
        # Early stopping (patience=12) is active within run_fold.
        print(f"\n--- Processing Fold {fold} ---")
        _ = train.run_fold(fold_index=fold, epochs=40)

        # Reload the best model for this fold to generate OOF predictions
        model = model_lib.TSICNN().to(device)
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"model_fold_{fold}.pth")

        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint for fold {fold} not found at {checkpoint_path}")
            continue

        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()

        # Get validation loader for this fold
        _, val_loader = data.get_fold_loaders(fold, batch_size=config.BATCH_SIZE)

        fold_probs = []
        fold_labels = []
        fold_angles = []
        fold_means = []

        # Inference on validation set
        with torch.no_grad():
            for images, angles, labels in val_loader:
                images = images.to(device)
                angles_in = angles.to(device)

                # Forward pass
                logits = model(images, angles_in)
                probs = torch.sigmoid(logits).cpu().numpy()

                # Store results
                fold_probs.extend(probs)
                fold_labels.extend(labels.numpy())
                fold_angles.extend(angles.numpy())

                # Calculate mean intensity of HH band (channel 0) for failure analysis
                # images shape: (B, 3, H, W)
                imgs_np = images.cpu().numpy()
                # Mean over H and W for channel 0
                batch_means = np.mean(imgs_np[:, 0, :, :], axis=(1, 2))
                fold_means.extend(batch_means)

        # Calculate errors for this fold
        fold_probs = np.array(fold_probs)
        fold_labels = np.array(fold_labels)
        fold_errors = np.abs(fold_probs - fold_labels)

        # Append to global OOF lists
        oof_preds.extend(fold_probs)
        oof_targets.extend(fold_labels)
        oof_errors.extend(fold_errors)
        oof_angles.extend(fold_angles)
        oof_img_means.extend(fold_means)

    # 3. Validation & Failure Analysis
    print("\n--- Validation & Failure Analysis ---")

    # Calculate Final Metric (Log Loss)
    # Clip predictions slightly to avoid log(0) errors, though log_loss handles this usually
    final_metric = log_loss(oof_targets, oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    df_analysis = pd.DataFrame(
        {"error": oof_errors, "angle": oof_angles, "img_mean": oof_img_means}
    )

    # Calculate correlations
    corr_angle = df_analysis["error"].corr(df_analysis["angle"])
    corr_img = df_analysis["error"].corr(df_analysis["img_mean"])

    print(f"Correlation (Error vs Angle): {corr_angle}")
    print(f"Correlation (Error vs Image Intensity): {corr_img}")

    # 4. Submission Generation
    threshold = 0.17174082291273365
    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric}) is below threshold ({threshold}). Generating submission..."
        )
        generate_submission(device)
    else:
        print(
            f"\nMetric ({final_metric}) is NOT below threshold ({threshold}). Submission skipped."
        )


def generate_submission(device):
    """
    Generates submission file by averaging predictions from all 5 fold models.
    """
    # Load Test Data
    test_loader, test_ids = data.get_test_loader(batch_size=config.BATCH_SIZE)

    # Array to store sum of predictions
    test_probs_sum = None

    print("Running inference on test set...")

    for fold in range(config.NUM_FOLDS):
        # Load model
        model = model_lib.TSICNN().to(device)
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"model_fold_{fold}.pth")
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()

        fold_probs = []

        with torch.no_grad():
            for images, angles in test_loader:
                images = images.to(device)
                angles = angles.to(device)

                logits = model(images, angles)
                probs = torch.sigmoid(logits).cpu().numpy()
                fold_probs.extend(probs)

        fold_probs = np.array(fold_probs)

        if test_probs_sum is None:
            test_probs_sum = fold_probs
        else:
            test_probs_sum += fold_probs

    # Average predictions
    avg_probs = test_probs_sum / config.NUM_FOLDS

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": avg_probs})

    # Save
    sub_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    df_sub.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


if __name__ == "__main__":
    main()
