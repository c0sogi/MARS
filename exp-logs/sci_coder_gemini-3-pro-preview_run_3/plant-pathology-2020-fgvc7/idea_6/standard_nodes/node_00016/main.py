import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import prepare_folds, get_loaders
from library.training import run_fold
from library.inference import generate_submission


def main():
    # 1. Configuration & Setup
    seed_everything(Config.SEED)

    # Fast Baseline Settings
    # Increased epochs to allow convergence (Cite solution_lesson_node_00008)
    Config.EPOCHS = 20

    print("Starting orchestration...")

    # 2. Training & OOF Generation
    # Ensure folds are prepared and cached
    df_folds = prepare_folds(load_cached_data=False)

    # Storage for OOF predictions: image_id -> accumulated probability vector
    oof_preds_accum = {}

    # Initialize dictionary for all images in the cross-validation setup
    image_ids = df_folds["image_id"].values
    for img_id in image_ids:
        oof_preds_accum[img_id] = np.zeros(Config.NUM_CLASSES, dtype=np.float32)

    model_types = ["effnet", "swin"]

    # Iterate through folds
    for fold in range(Config.N_FOLDS):
        print(f"\n=== Processing Fold {fold}/{Config.N_FOLDS - 1} ===")

        for model_type in model_types:
            # Train the model
            # run_fold handles training, saving best model, and reloading it
            model = run_fold(fold, model_type)

            # Determine image size based on model type
            if model_type == "effnet":
                img_size = Config.IMG_SIZE_EFFNET
            else:
                img_size = Config.IMG_SIZE_SWIN

            # Get Validation Loader for inference
            # We use the same batch size as training
            _, val_loader = get_loaders(fold, img_size, Config.BATCH_SIZE)

            # Inference on Validation Fold
            model.eval()
            fold_preds = []

            # Retrieve image IDs for the current validation fold to map predictions
            # val_loader.dataset is an AppleDataset instance which has .image_ids
            dataset_ids = val_loader.dataset.image_ids

            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(Config.DEVICE)

                    # Forward pass
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)

                    fold_preds.append(probs.cpu().numpy())

            fold_preds = np.concatenate(fold_preds)

            # Accumulate predictions
            # We sum probabilities from both models; later we divide by len(model_types)
            for idx, img_id in enumerate(dataset_ids):
                oof_preds_accum[img_id] += fold_preds[idx]

            # Clean up to save memory
            del model
            torch.cuda.empty_cache()

    # 3. Process OOF Predictions
    # Average over the 2 models (Ensemble)
    for img_id in oof_preds_accum:
        oof_preds_accum[img_id] /= len(model_types)

    # 4. Filter for Hold-out Validation Set
    # We strictly use the metadata/val.csv as the hold-out set for the final metric
    val_meta_path = os.path.join(Config.METADATA_DIR, "val.csv")
    if not os.path.exists(val_meta_path):
        raise FileNotFoundError(f"Validation metadata not found at {val_meta_path}")

    val_meta_df = pd.read_csv(val_meta_path)

    y_true = []
    y_pred = []
    val_file_paths = []

    print("\nCalculating metrics on hold-out validation set...")

    for idx, row in val_meta_df.iterrows():
        img_id = row["image_id"]
        if img_id in oof_preds_accum:
            # Construct one-hot ground truth vector
            true_vec = row[Config.CLASSES].values.astype(np.float32)
            y_true.append(true_vec)

            # Retrieve prediction
            y_pred.append(oof_preds_accum[img_id])

            # Store path for failure analysis
            val_file_paths.append(row["file_path"])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 5. Calculate Final Metric
    # Metric: Mean column-wise ROC AUC
    try:
        final_metric = roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr")
    except Exception as e:
        print(f"Error calculating metric: {e}")
        final_metric = 0.0

    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")

    # Calculate error magnitude per sample
    # Error = 1.0 - Probability assigned to the correct class
    # Since y_true is one-hot, sum(y_true * y_pred) gives the prob of the correct class
    prob_correct = np.sum(y_true * y_pred, axis=1)
    errors = 1.0 - prob_correct

    # Extract image features (Brightness and Contrast)
    brightness = []
    contrast = []

    for rel_path in val_file_paths:
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        img = cv2.imread(full_path)
        if img is None:
            # Fallback if image read fails (should not happen)
            brightness.append(0)
            contrast.append(0)
            continue

        # Convert to grayscale for stats
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness.append(np.mean(gray))
        contrast.append(np.std(gray))

    brightness = np.array(brightness)
    contrast = np.array(contrast)

    # Calculate Correlations
    if len(errors) > 1:
        corr_bright, _ = pearsonr(errors, brightness)
        corr_contrast, _ = pearsonr(errors, contrast)

        print(f"Correlation (Error vs Brightness): {corr_bright:.4f}")
        print(f"Correlation (Error vs Contrast): {corr_contrast:.4f}")
    else:
        print("Not enough samples for correlation analysis.")

    # 7. Submission
    threshold = 0.9924834132836718

    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) exceeds threshold ({threshold}). Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
