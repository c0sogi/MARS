import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Ensure library is in path
sys.path.append(os.path.join(os.getcwd(), "library"))

from library.config import (
    PLANES,
    N_FOLDS,
    SEED,
    METADATA_DIR,
    MODELS_DIR,
    DEVICE,
    BATCH_SIZE,
)
from library.utils import seed_everything, get_logger
from library.trainer import run_training
from library.inference import run_inference
from library.preprocessing import process_dataset
from library.model import ExpertNet
from library.dataset import RASSEDataset, get_transforms


def main():
    # 1. Setup
    seed_everything(SEED)
    logger = get_logger("main_pipeline")
    logger.info("Starting Runfile Execution...")

    # Fast baseline settings
    EPOCHS = 5

    # 2. Training Loop
    # We train 3 Expert Planes x 5 Folds = 15 Models
    logger.info("=== Starting Training Phase ===")

    for plane_name in PLANES.keys():
        logger.info(f"Training Expert Plane: {plane_name}")
        for fold_idx in range(N_FOLDS):
            logger.info(f"  > Fold {fold_idx}/{N_FOLDS - 1}")

            # Check if model already exists to save time (optional, but good for re-runs)
            model_path = os.path.join(
                MODELS_DIR, f"best_model_{plane_name}_fold{fold_idx}.pth"
            )
            if os.path.exists(model_path):
                logger.info(
                    f"    Model {model_path} already exists. Skipping training."
                )
                continue

            # Run Training
            best_auc = run_training(
                plane_name=plane_name,
                fold_idx=fold_idx,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                patience=3,  # Strict early stopping for speed
            )
            logger.info(f"    Fold {fold_idx} Best AUC: {best_auc:.4f}")

    # 3. Validation Phase (Hold-out Set Evaluation)
    logger.info("=== Starting Validation Phase ===")

    val_meta_path = os.path.join(METADATA_DIR, "val_metadata.csv")
    if not os.path.exists(val_meta_path):
        logger.error("Validation metadata not found.")
        return

    df_val = pd.read_csv(val_meta_path)

    # Process Validation Data
    # Returns (N, 3, H, W, 3) -> Dim 1 is [Lower, Center, Upper]
    val_images, val_ids, val_labels = process_dataset(
        df_val, load_cached_data=True, save_name="val_eval"
    )

    if val_labels is None:
        logger.error("Validation labels are missing.")
        return

    # Ensemble Prediction on Validation Set
    num_samples = len(val_ids)
    final_val_probs = np.zeros(num_samples, dtype=np.float64)
    model_count = 0

    # Iterate through all trained models
    for plane_name in PLANES.keys():
        # Create Dataset for this plane
        val_dataset = RASSEDataset(
            val_images,
            val_ids,
            labels=val_labels,
            plane_name=plane_name,
            transform=get_transforms(phase="val"),  # No augmentation
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        for fold_idx in range(N_FOLDS):
            model_path = os.path.join(
                MODELS_DIR, f"best_model_{plane_name}_fold{fold_idx}.pth"
            )
            if not os.path.exists(model_path):
                continue

            # Load Model
            model = ExpertNet().to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()

            # Predict
            fold_preds = []
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch["image"].to(DEVICE)
                    out = model(imgs)
                    probs = torch.sigmoid(out).cpu().numpy().flatten()
                    fold_preds.extend(probs)

            final_val_probs += np.array(fold_preds)
            model_count += 1

            # Cleanup
            del model
            torch.cuda.empty_cache()

    if model_count == 0:
        logger.error("No models found for validation.")
        return

    # Average predictions
    avg_val_probs = final_val_probs / model_count

    # Calculate Metric
    final_metric = roc_auc_score(val_labels, avg_val_probs)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    logger.info("=== Performing Failure Analysis ===")

    # Calculate Error
    errors = np.abs(val_labels - avg_val_probs)

    # Extract simple features from images for correlation
    # We'll use the 'center' plane for representative features
    # val_images shape: (N, 3, H, W, 3) -> Index 1 is Center
    center_imgs = val_images[:, 1, :, :, :]  # (N, H, W, 3)

    # Feature 1: Brain Area (Non-zero pixels sum across channels)
    # Sum over H, W, C
    brain_area = np.sum(center_imgs > 0, axis=(1, 2, 3))

    # Feature 2: Mean Intensity (of non-zero pixels)
    mean_intensity = []
    for i in range(len(center_imgs)):
        img = center_imgs[i]
        mask = img > 0
        if np.any(mask):
            mean_intensity.append(img[mask].mean())
        else:
            mean_intensity.append(0.0)
    mean_intensity = np.array(mean_intensity)

    # Create DataFrame for correlation
    df_analysis = pd.DataFrame(
        {"error": errors, "brain_area": brain_area, "mean_intensity": mean_intensity}
    )

    correlations = df_analysis.corr()["error"].drop("error")

    print("\nCorrelation between Error Magnitude and Input Features:")
    print(correlations)

    # 5. Submission
    THRESHOLD = 0.6705454545454544

    if final_metric > THRESHOLD:
        logger.info(
            f"Validation metric {final_metric:.6f} > {THRESHOLD:.6f}. Generating submission..."
        )
        run_inference(batch_size=BATCH_SIZE, load_cached_data=True)
    else:
        logger.warning(
            f"Validation metric {final_metric:.6f} <= {THRESHOLD:.6f}. Submission skipped."
        )


if __name__ == "__main__":
    main()
