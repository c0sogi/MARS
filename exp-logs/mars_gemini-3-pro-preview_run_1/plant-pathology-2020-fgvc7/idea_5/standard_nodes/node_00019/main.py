import os
import pandas as pd
import numpy as np
import torch
import cv2
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.train import run_training_fold
from library.inference import generate_submission, predict_with_tta
from library.models import get_model
from library.dataset import AppleLeafDataset, get_transforms


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    seed_everything(Config.SEED)

    # Modify Config for Optimal Convergence (Cite Lesson 00017)
    # Increasing epochs to ensure full convergence for the single model
    Config.EPOCHS = 15
    # Synchronize scheduler cycle with epochs (Cite Lesson 00015)
    Config.T_0 = Config.EPOCHS

    print(
        f"Configuration: Epochs={Config.EPOCHS}, Folds={Config.N_FOLDS}, Device={Config.DEVICE}"
    )

    # ==========================================
    # 2. Training Loop (Stratified K-Fold)
    # ==========================================
    # Load training metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            f"Train metadata not found at {Config.TRAIN_METADATA_PATH}"
        )

    full_train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Prepare Stratified K-Fold
    # We use 'stratify_label' which represents the dominant class
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Iterate over architectures and folds
    for arch in Config.MODEL_ARCHS:
        print(f"\n==== Training Architecture: {arch} ====")

        # The split returns indices
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(full_train_df, full_train_df["stratify_label"])
        ):
            print(f"--- Fold {fold} ---")

            fold_train_df = full_train_df.iloc[train_idx].reset_index(drop=True)
            fold_val_df = full_train_df.iloc[val_idx].reset_index(drop=True)

            # Run training for this specific fold
            # This function saves the model to disk automatically
            best_auc = run_training_fold(arch, fold_train_df, fold_val_df, fold)

            # Clean up memory
            del fold_train_df, fold_val_df
            import gc

            gc.collect()

    # ==========================================
    # 3. Validation on Hold-out Set
    # ==========================================
    print("\n==== Starting Evaluation on Hold-out Validation Set ====")

    if not os.path.exists(Config.VAL_METADATA_PATH):
        raise FileNotFoundError(
            f"Validation metadata not found at {Config.VAL_METADATA_PATH}"
        )

    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Prepare Validation DataLoader
    val_dataset = AppleLeafDataset(
        val_df,
        transforms=get_transforms("valid"),  # Use valid transforms (deterministic)
        mode="test",  # We want image and ID/dummy, but we have labels in DF for metric calc
    )

    # Note: We use mode="test" to get raw images easily for inference function compatibility,
    # but we need labels for metric calculation.
    # Let's actually use mode="test" for the dataset passed to predict_with_tta
    # and extract labels directly from dataframe for metric.

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Ensemble Inference
    models_dir = os.path.join(Config.WORKING_DIR, "models")
    num_classes = Config.NUM_CLASSES
    num_samples = len(val_df)

    aggregated_preds = np.zeros((num_samples, num_classes), dtype=np.float32)
    model_count = 0

    for arch in Config.MODEL_ARCHS:
        for fold in range(Config.N_FOLDS):
            model_path = os.path.join(models_dir, f"{arch}_fold_{fold}.pth")

            if not os.path.exists(model_path):
                print(f"Warning: Model {model_path} not found. Skipping.")
                continue

            # Load Model
            model = get_model(arch, num_classes, pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
            model.to(Config.DEVICE)

            # Predict
            preds, _ = predict_with_tta(model, val_loader, Config.DEVICE)

            aggregated_preds += preds
            model_count += 1

            del model
            torch.cuda.empty_cache()

    if model_count == 0:
        raise RuntimeError("No models available for validation.")

    final_preds = aggregated_preds / model_count

    # Get Ground Truth
    # The columns are defined in Config.CLASS_LABELS
    y_true = val_df[Config.CLASS_LABELS].values

    # Calculate Metric
    final_metric = calculate_metric(y_true, final_preds)

    # PRINT REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\n==== Failure Analysis ====")

    # Calculate error per sample
    # Error = 1.0 - probability assigned to the true class
    # We assume y_true is one-hot or soft labels. We take argmax for "True Class" concept
    true_class_indices = np.argmax(y_true, axis=1)

    # Extract predicted probability for the true class
    # final_preds is (N, 4), true_class_indices is (N,)
    probs_at_true_class = final_preds[np.arange(len(final_preds)), true_class_indices]
    errors = 1.0 - probs_at_true_class

    # Extract Meta-Features
    # We need to read images to get Width, Height, Intensity
    print("Extracting image meta-features for correlation analysis...")
    widths = []
    heights = []
    intensities = []

    for idx, row in val_df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        img = cv2.imread(full_path)
        if img is None:
            # Fallback
            widths.append(0)
            heights.append(0)
            intensities.append(0)
            continue

        h, w, c = img.shape
        # Calculate mean intensity (grayscale equivalent roughly)
        mean_intensity = img.mean() / 255.0

        widths.append(w)
        heights.append(h)
        intensities.append(mean_intensity)

    # Calculate Correlations
    if len(errors) == len(widths):
        corr_width, _ = pearsonr(errors, widths)
        corr_height, _ = pearsonr(errors, heights)
        corr_intensity, _ = pearsonr(errors, intensities)

        print(f"Correlation between Error and Image Width: {corr_width:.4f}")
        print(f"Correlation between Error and Image Height: {corr_height:.4f}")
        print(f"Correlation between Error and Mean Intensity: {corr_intensity:.4f}")

        # Simple interpretation
        if abs(corr_intensity) > 0.1:
            print("-> Model performance seems sensitive to image brightness.")
    else:
        print("Skipping correlation analysis due to data length mismatch.")

    # ==========================================
    # 5. Submission Generation
    # ==========================================
    THRESHOLD = 0.9871488489626378

    if final_metric > THRESHOLD:
        print(
            f"\nValidation Metric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"\nValidation Metric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
