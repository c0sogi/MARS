import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from library import config, utils, data, model, train


def main():
    # ==========================================
    # 1. SETUP & CONFIGURATION
    # ==========================================
    # Ensure reproducibility
    utils.seed_everything(config.SEED)

    # Override configuration for fast baseline execution
    # 30 epochs is sufficient for convergence on this small dataset
    # and ensures we stay well within the 17-minute time limit.
    config.NUM_EPOCHS = 30

    print(f"Starting execution with {config.NUM_EPOCHS} epochs per fold...")

    # ==========================================
    # 2. DATA LOADING
    # ==========================================
    # Load processed data (uses cache if available)
    data_dict, scaler = data.process_and_cache_data(load_cached_data=True)

    # ==========================================
    # 3. CROSS-VALIDATION TRAINING
    # ==========================================
    oof_preds = []
    oof_targets = []
    oof_angles = []

    # Store fold-wise validation indices to align with global arrays if needed,
    # but simple concatenation works because we just need global metric.

    for fold_idx in range(config.NUM_FOLDS):
        print(f"\n--- Processing Fold {fold_idx} ---")

        # Train the model for this fold
        # run_fold returns the state_dict of the best model
        best_weights = train.run_fold(fold_idx, data_dict, scaler)

        # Save weights for later test inference
        weight_path = os.path.join(config.WORKING_DIR, f"model_fold_{fold_idx}.pth")
        torch.save(best_weights, weight_path)

        # --- OOF Inference ---
        # Initialize a fresh model
        net = model.RDP_WBN().to(config.DEVICE)
        net.load_state_dict(best_weights)
        net.eval()

        # Get validation loader
        _, val_loader, _ = data.get_dataloaders(fold_idx, data_dict, scaler)

        fold_preds = []
        fold_targets = []
        fold_angles_batch = []

        with torch.no_grad():
            for images, angles, labels in val_loader:
                images = images.to(config.DEVICE)
                angles = angles.to(config.DEVICE)

                # Forward pass
                outputs = net(images, angles)

                fold_preds.append(outputs.cpu().numpy())
                fold_targets.append(labels.cpu().numpy())
                fold_angles_batch.append(angles.cpu().numpy())

        # Concatenate batch results
        fold_preds = np.concatenate(fold_preds)
        fold_targets = np.concatenate(fold_targets)
        fold_angles_batch = np.concatenate(fold_angles_batch)

        oof_preds.append(fold_preds)
        oof_targets.append(fold_targets)
        oof_angles.append(fold_angles_batch)

    # ==========================================
    # 4. VALIDATION ASSESSMENT
    # ==========================================
    # Flatten all OOF arrays
    all_preds = np.concatenate(oof_preds).flatten()
    all_targets = np.concatenate(oof_targets).flatten()
    all_angles = np.concatenate(oof_angles).flatten()

    # Calculate Log Loss
    # Scikit-learn log_loss handles probabilities and binary targets
    final_metric = log_loss(all_targets, all_preds, labels=[0, 1])

    print("\n" + "=" * 30)
    print(f"Final Validation Metric: {final_metric}")
    print("=" * 30)

    # ==========================================
    # 5. FAILURE ANALYSIS
    # ==========================================
    print("\n[Failure Analysis]")

    # Calculate absolute error
    errors = np.abs(all_targets - all_preds)

    # Create DataFrame for analysis
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": all_angles,
            "target": all_targets,
            "prediction": all_preds,
        }
    )

    # Calculate correlations with error
    correlations = analysis_df.corr()["error"].sort_values(ascending=False)
    print("Correlation between Error and Features:")
    print(correlations)

    # ==========================================
    # 6. SUBMISSION GENERATION
    # ==========================================
    THRESHOLD = 0.14772333549413377

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} meets threshold {THRESHOLD}. Generating submission..."
        )

        # Get Test Loader (Fold 0 loader is sufficient as test set is constant)
        _, _, test_loader = data.get_dataloaders(0, data_dict, scaler)

        num_test_samples = len(data_dict["ids_test"])
        test_preds_sum = np.zeros((num_test_samples, 1))

        # Ensemble Inference
        for fold_idx in range(config.NUM_FOLDS):
            print(f"Inference with model fold {fold_idx}...")

            # Load Model
            net = model.RDP_WBN().to(config.DEVICE)
            weight_path = os.path.join(config.WORKING_DIR, f"model_fold_{fold_idx}.pth")
            net.load_state_dict(torch.load(weight_path, map_location=config.DEVICE))
            net.eval()

            fold_test_preds = []

            with torch.no_grad():
                for images, angles in test_loader:
                    images = images.to(config.DEVICE)
                    angles = angles.to(config.DEVICE)

                    outputs = net(images, angles)
                    fold_test_preds.append(outputs.cpu().numpy())

            # Accumulate predictions
            test_preds_sum += np.concatenate(fold_test_preds)

        # Average predictions
        avg_preds = test_preds_sum / config.NUM_FOLDS

        # Create Submission DataFrame
        sub_df = pd.DataFrame(
            {"id": data_dict["ids_test"], "is_iceberg": avg_preds.flatten()}
        )

        # Save to file
        sub_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
