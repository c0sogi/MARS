import os
import numpy as np
import torch
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

import library.config as config
import library.utils as utils
import library.model as model_lib
import library.data_loader as data_loader
import library.train_eval as train_eval


def main():
    # 1. Setup
    utils.set_seed(config.SEED)
    config.setup_directories()

    # Override config for fast execution as per requirements
    config.NUM_EPOCHS = 25
    config.PATIENCE = 5

    device = config.DEVICE

    # Containers for global validation results
    all_val_preds = []
    all_val_targets = []

    # Containers for failure analysis metadata
    meta_inc_angles = []
    meta_b1_mean = []
    meta_b2_mean = []

    # 2. Train & Validate Loop
    for fold_idx in range(config.NUM_FOLDS):
        # Train the fold (saves best checkpoint internally)
        train_eval.train_fold(fold_idx, load_cached_data=True)

        # Load best model for this fold to perform inference
        model = model_lib.MCICNN().to(device)
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"model_fold_{fold_idx}.pth")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        # Get validation data loader for this fold
        _, val_loader = data_loader.get_data_loaders(fold_idx, load_cached_data=True)

        # Inference on validation set
        with torch.no_grad():
            for images, angles, labels in val_loader:
                images = images.to(device)
                angles_gpu = angles.to(device)

                # Forward pass
                logits = model(images, angles_gpu)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

                # Store predictions and targets
                all_val_preds.extend(probs)
                all_val_targets.extend(labels.numpy())

                # Extract metadata for failure analysis
                # images shape: (B, 3, 75, 75). Channel 0 is Band 1, Channel 1 is Band 2.
                imgs_np = images.cpu().numpy()
                b1 = imgs_np[:, 0, :, :]
                b2 = imgs_np[:, 1, :, :]

                # Compute simple stats
                meta_b1_mean.extend(np.mean(b1, axis=(1, 2)))
                meta_b2_mean.extend(np.mean(b2, axis=(1, 2)))
                meta_inc_angles.extend(angles.numpy())

    # 3. Final Metric Calculation
    # Calculate Log Loss on the aggregated validation set
    final_metric = log_loss(all_val_targets, all_val_preds)
    print(f"Final Validation Metric: {final_metric:.15f}")

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")
    y_true = np.array(all_val_targets)
    y_pred = np.array(all_val_preds)
    errors = np.abs(y_true - y_pred)

    # Prepare metadata arrays
    angles_arr = np.array(meta_inc_angles)
    b1_mean_arr = np.array(meta_b1_mean)
    b2_mean_arr = np.array(meta_b2_mean)

    # Calculate correlations
    corr_angle, _ = pearsonr(errors, angles_arr)
    corr_b1, _ = pearsonr(errors, b1_mean_arr)
    corr_b2, _ = pearsonr(errors, b2_mean_arr)

    print(f"Correlation (Error vs Inc Angle): {corr_angle:.4f}")
    print(f"Correlation (Error vs Band 1 Mean): {corr_b1:.4f}")
    print(f"Correlation (Error vs Band 2 Mean): {corr_b2:.4f}")

    # 5. Submission Generation
    threshold = 0.17174082291273365
    if final_metric < threshold:
        train_eval.generate_submission(load_cached_data=True)
    else:
        print(f"Validation metric {final_metric} >= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
