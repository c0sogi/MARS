import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from library.config import Config
from library.train import train_model
from library.inference import generate_submission, load_ensemble, predict_with_tta
from library.dataset import CactusDataset


def main():
    # -------------------------------------------------------------------------
    # 1. Training Phase
    # -------------------------------------------------------------------------
    print("========================================")
    print("           TRAINING PHASE               ")
    print("========================================")

    # Train 5 independent models using Homogeneous Seed Averaging.
    # The dataset is small (32x32 images), so we use the full training set
    # and the default epoch count (20) which fits comfortably within the time limit.
    for seed in Config.SEEDS:
        train_model(seed=seed, epochs=Config.EPOCHS, debug=False)

    # -------------------------------------------------------------------------
    # 2. Validation Phase
    # -------------------------------------------------------------------------
    print("\n========================================")
    print("          VALIDATION PHASE              ")
    print("========================================")

    device = Config.DEVICE

    # Load the ensemble of trained models
    # This automatically handles structural re-parameterization for inference
    models = load_ensemble(device)

    # Load Validation Dataset
    val_dataset = CactusDataset(mode="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    all_labels = []
    all_preds = []

    # Accumulators for failure analysis meta-features
    meta_stats = {
        "brightness": [],
        "contrast": [],
        "red_mean": [],
        "green_mean": [],
        "blue_mean": [],
    }

    print("Running inference on validation set with TTA...")

    # Inference loop (No Grad for efficiency)
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)

            # Get averaged predictions from ensemble + TTA
            # predict_with_tta returns a numpy array of probabilities
            batch_preds = predict_with_tta(models, images)

            all_preds.extend(batch_preds)
            all_labels.extend(labels.numpy())

            # --- Feature Extraction for Failure Analysis ---
            # Move images to CPU for stats calculation
            # images shape: (B, 3, 32, 32), range [0, 1]
            imgs_np = images.cpu().numpy()

            # Calculate mean intensity per channel per image
            # Axis 2, 3 are Height and Width
            batch_means = np.mean(imgs_np, axis=(2, 3))  # Shape: (B, 3)

            # Brightness: Mean of all channels
            batch_brightness = np.mean(batch_means, axis=1)  # Shape: (B,)

            # Contrast: Standard deviation of all pixels in the image
            batch_contrast = np.std(imgs_np, axis=(1, 2, 3))  # Shape: (B,)

            meta_stats["brightness"].extend(batch_brightness)
            meta_stats["contrast"].extend(batch_contrast)
            meta_stats["red_mean"].extend(batch_means[:, 0])
            meta_stats["green_mean"].extend(batch_means[:, 1])
            meta_stats["blue_mean"].extend(batch_means[:, 2])

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Calculate Final Metric (ROC AUC)
    val_auc = roc_auc_score(all_labels, all_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # -------------------------------------------------------------------------
    # 3. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n========================================")
    print("          FAILURE ANALYSIS              ")
    print("========================================")

    # Calculate error magnitude: |y_true - y_pred|
    errors = np.abs(all_labels - all_preds)

    print("Correlation between Error Magnitude and Input Features:")
    for feat_name, values in meta_stats.items():
        values = np.array(values)
        # Ensure sufficient variance to calculate correlation
        if len(values) > 1 and np.std(values) > 1e-9:
            corr, _ = pearsonr(errors, values)
            print(f"{feat_name}: {corr:.4f}")
        else:
            print(f"{feat_name}: Undefined (constant input)")

    # -------------------------------------------------------------------------
    # 4. Submission Phase
    # -------------------------------------------------------------------------
    print("\n========================================")
    print("         SUBMISSION PHASE               ")
    print("========================================")

    # Generate submission file for the test set
    # Note: The prompt requirement "metric > 1.0" is strictly impossible for AUC.
    # We proceed with submission to ensure the task is completed and scored.
    generate_submission(debug=False)


if __name__ == "__main__":
    main()
