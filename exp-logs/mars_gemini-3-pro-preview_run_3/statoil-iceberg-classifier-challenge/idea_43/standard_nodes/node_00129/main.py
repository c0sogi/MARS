import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from library.utils import set_seed, get_device
from library.model import CAFPCNN
from library.data import get_loaders
from library.train import fit_fold


def main():
    # 1. Setup
    set_seed(42)
    device = get_device()

    # Hyperparameters & Paths
    N_FOLDS = 5
    EPOCHS = 50  # Sufficient for convergence on this small dataset
    BATCH_SIZE = 32
    PATIENCE = 12
    CHECKPOINT_DIR = "./working/idea_43/checkpoints/"
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # 2. Training Loop
    print("Starting 5-Fold Cross-Validation...")
    for fold in range(N_FOLDS):
        fit_fold(
            fold=fold,
            n_folds=N_FOLDS,
            epochs=EPOCHS,
            patience=PATIENCE,
            batch_size=BATCH_SIZE,
            save_dir=CHECKPOINT_DIR,
        )

    # 3. Validation & Failure Analysis
    print("\nStarting Validation and Failure Analysis...")
    oof_preds = []
    oof_targets = []
    oof_angles = []
    oof_b1_means = []
    oof_b2_means = []

    # Generate Out-Of-Fold Predictions
    for fold in range(N_FOLDS):
        # Load the best model for this fold
        model = CAFPCNN()
        model_path = os.path.join(CHECKPOINT_DIR, f"model_fold_{fold}.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Get validation loader for this fold
        _, val_loader, _ = get_loaders(
            fold=fold, n_folds=N_FOLDS, batch_size=BATCH_SIZE
        )

        with torch.no_grad():
            for images, angles, targets in val_loader:
                images = images.to(device)
                angles_gpu = angles.to(device)

                # Inference
                logits = model(images, angles_gpu)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

                # Store Predictions and Metadata
                oof_preds.extend(probs)
                oof_targets.extend(targets.numpy().flatten())
                oof_angles.extend(angles.numpy().flatten())

                # Calculate simple image stats for failure analysis
                # Image shape: (B, 3, 75, 75). Band 1 is idx 0, Band 2 is idx 1.
                imgs_np = images.cpu().numpy()
                b1_mean = np.mean(imgs_np[:, 0, :, :], axis=(1, 2))
                b2_mean = np.mean(imgs_np[:, 1, :, :], axis=(1, 2))
                oof_b1_means.extend(b1_mean)
                oof_b2_means.extend(b2_mean)

    # Calculate Final Metric
    y_true = np.array(oof_targets)
    y_pred = np.array(oof_preds)
    # Clip predictions to prevent log(0) errors
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
    final_metric = log_loss(y_true, y_pred_clipped)

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    errors = np.abs(y_true - y_pred)
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": oof_angles,
            "b1_mean": oof_b1_means,
            "b2_mean": oof_b2_means,
        }
    )

    print("\nFailure Analysis (Correlation with Error Magnitude):")
    correlations = analysis_df.corr()["error"].drop("error")
    print(correlations)

    # 4. Submission
    THRESHOLD = 0.1806015565870406

    if final_metric < THRESHOLD:
        print("\nMetric passed threshold. Generating submission...")

        # Get Test Loader (same for all folds)
        _, _, test_loader = get_loaders(fold=0, n_folds=N_FOLDS, batch_size=BATCH_SIZE)

        test_ids = []
        ensemble_preds = None

        # Ensemble Inference
        for fold in range(N_FOLDS):
            model = CAFPCNN()
            model_path = os.path.join(CHECKPOINT_DIR, f"model_fold_{fold}.pth")
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            fold_preds = []
            current_ids = []

            with torch.no_grad():
                for images, angles, ids in test_loader:
                    images = images.to(device)
                    angles_gpu = angles.to(device)

                    logits = model(images, angles_gpu)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()

                    fold_preds.extend(probs)
                    if fold == 0:
                        current_ids.extend(ids)

            fold_preds = np.array(fold_preds)

            if ensemble_preds is None:
                ensemble_preds = fold_preds
                test_ids = current_ids
            else:
                ensemble_preds += fold_preds

        # Average predictions
        avg_preds = ensemble_preds / N_FOLDS

        # Save Submission
        submission_df = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})

        sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
