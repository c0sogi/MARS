import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

from library.utils import seed_everything, get_device
from library.dataset import get_dataloaders
from library.model import MicroResNet, train_model
from library.inference import predict_with_tta


def main():
    # 1. Setup Environment
    seed_everything(42)
    device = get_device()

    # 2. Data Loading
    # We use get_dataloaders to load the specific Train and Hold-out Validation sets
    # defined in the metadata directory.
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=32, num_workers=2, load_cached_data=True
    )

    # 3. Model Initialization
    model = MicroResNet()

    # Directory for temporary model weights
    working_dir = "./working/single_run"
    os.makedirs(working_dir, exist_ok=True)
    save_path = os.path.join(working_dir, "model_best.pth")

    # 4. Training
    # We train on the training set and validate on the hold-out validation set.
    # 35 epochs is chosen as a balance between speed and convergence for this dataset size.
    print("Starting training...")
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=35,
        lr=1e-3,
        patience=10,
        save_path=save_path,
    )

    # 5. Validation Assessment & Metric Calculation
    print("Evaluating on hold-out validation set...")
    model.eval()
    val_preds = []
    val_targets = []
    val_angles = []
    val_images_list = []

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            angles = batch["angle"].to(device)
            lbls = batch["label"].to(device)

            # Forward pass
            outputs = model(imgs, angles)

            # Store results
            val_preds.extend(outputs.cpu().numpy())
            val_targets.extend(lbls.cpu().numpy())
            val_angles.extend(angles.cpu().numpy())
            val_images_list.append(imgs.cpu().numpy())

    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    val_angles = np.array(val_angles)
    val_images = np.concatenate(val_images_list, axis=0)  # Shape: (N, 3, 75, 75)

    # Calculate Log Loss
    # Ensure strict 0-1 clipping isn't strictly necessary as sigmoid outputs (0,1),
    # but good practice for numerical stability in log_loss if values are exactly 0 or 1.
    final_metric = log_loss(val_targets, val_preds, labels=[0, 1])
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(val_targets - val_preds)

    # Extract features for correlation
    # Band 1 is channel 0, Band 2 is channel 1
    # We compute the mean intensity for each image
    feat_b1_mean = np.mean(val_images[:, 0, :, :], axis=(1, 2))
    feat_b2_mean = np.mean(val_images[:, 1, :, :], axis=(1, 2))

    # Compute correlations
    # Handle potential NaN in angles if imputation failed (though dataset.py handles it)
    # Just in case, we ensure no NaNs propagate to pearsonr
    valid_mask = ~np.isnan(val_angles)
    if np.sum(valid_mask) > 1:
        corr_angle = pearsonr(errors[valid_mask], val_angles[valid_mask])[0]
    else:
        corr_angle = 0.0

    corr_b1 = pearsonr(errors, feat_b1_mean)[0]
    corr_b2 = pearsonr(errors, feat_b2_mean)[0]

    print("Correlation between Error Magnitude and Features:")
    print(f"  Incidence Angle: {corr_angle:.4f}")
    print(f"  Band 1 Mean: {corr_b1:.4f}")
    print(f"  Band 2 Mean: {corr_b2:.4f}")

    # 7. Submission Generation
    threshold = 0.18145903282502943

    if final_metric < threshold:
        print(
            f"\nMetric {final_metric} meets threshold {threshold}. Generating submission..."
        )

        # Predict on Test Set using TTA
        test_preds = predict_with_tta(model, test_loader, device)

        # Retrieve IDs from the test dataset
        test_ids = test_loader.dataset.ids

        # Create Submission DataFrame
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": test_preds})

        df_sub.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

    else:
        print(
            f"\nMetric {final_metric} does not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
