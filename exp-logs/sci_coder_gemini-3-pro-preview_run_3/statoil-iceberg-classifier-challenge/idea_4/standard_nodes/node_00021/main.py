import os
import numpy as np
import torch
from sklearn.metrics import log_loss
from library.config import N_FOLDS, WORKING_DIR, DEVICE, SEED
from library.utils import set_seed
from library.model import IcebergSECNN
from library.train import run_fold
from library.data import get_dataloaders
from library.predict import generate_submission


def main():
    # Ensure reproducibility
    set_seed(SEED)

    print("Starting training pipeline...")
    # Train all folds
    for fold_idx in range(N_FOLDS):
        print(f"--- Training Fold {fold_idx} ---")
        run_fold(fold_idx)

    print("Training complete. Starting validation...")

    # Load validation data
    # get_dataloaders returns (train_loader, val_loader, test_loader)
    # We only need val_loader which corresponds to the metadata/val.csv split
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Load all trained models for ensembling
    models = []
    for fold_idx in range(N_FOLDS):
        model_path = os.path.join(WORKING_DIR, f"fold_{fold_idx}", "model_best.pth")
        model = IcebergSECNN().to(DEVICE)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            models.append(model)
        else:
            print(f"Warning: Model for fold {fold_idx} not found.")

    if not models:
        print("No models available for validation.")
        return

    # Validation Inference
    all_preds = []
    all_targets = []
    all_angles = []

    with torch.no_grad():
        for images, angles, labels in val_loader:
            images = images.to(DEVICE)
            angles_dev = angles.to(DEVICE)

            batch_preds = []
            for model in models:
                # TTA: Original
                out1 = model(images, angles_dev)
                # TTA: Horizontal Flip
                out2 = model(torch.flip(images, [3]), angles_dev)
                # TTA: Vertical Flip
                out3 = model(torch.flip(images, [2]), angles_dev)

                # Average TTA for this model
                avg_out = (out1 + out2 + out3) / 3.0
                batch_preds.append(avg_out.cpu().numpy())

            # Average across ensemble (Folds)
            ensemble_batch_pred = np.mean(batch_preds, axis=0)

            all_preds.append(ensemble_batch_pred)
            all_targets.append(labels.numpy())
            all_angles.append(angles.numpy())

    # Flatten arrays
    y_pred = np.concatenate(all_preds).flatten()
    y_true = np.concatenate(all_targets).flatten()
    angles_arr = np.concatenate(all_angles).flatten()

    # Calculate Metric
    final_metric = log_loss(y_true, y_pred, labels=[0, 1])
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    errors = np.abs(y_true - y_pred)
    # Calculate correlation (using numpy to avoid scipy dependency issues)
    # Pearson correlation coefficient
    if len(errors) > 1:
        corr_matrix = np.corrcoef(errors, angles_arr)
        correlation = corr_matrix[0, 1]
    else:
        correlation = 0.0

    print(f"Correlation between Error and Incidence Angle: {correlation}")

    # Submission Generation
    threshold = 0.18145903282502943
    if final_metric < threshold:
        print(
            f"Metric {final_metric} is below threshold {threshold}. Generating submission..."
        )
        generate_submission(load_cached_data=True)
    else:
        print(
            f"Metric {final_metric} is not below threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
