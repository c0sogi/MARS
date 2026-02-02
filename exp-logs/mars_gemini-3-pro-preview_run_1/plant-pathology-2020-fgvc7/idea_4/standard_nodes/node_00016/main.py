import os
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

# Import provided library modules
from library import config, utils, data, modeling, training, inference


def main():
    # ==========================================
    # 1. Configuration Override
    # ==========================================
    # Limit epochs for a fast baseline execution while ensuring convergence
    config.CFG.epochs = 10
    config.CFG.debug = False  # Use full data to aim for the best score

    # Set seeds for reproducibility
    utils.seed_everything(config.CFG.seed)

    # ==========================================
    # 2. Training
    # ==========================================
    print("Starting Training of Heterogeneous Ensemble...")
    training.train_models()

    # ==========================================
    # 3. Validation & Metric Calculation
    # ==========================================
    print("Starting Validation and OOF Prediction...")

    # Load full training data
    df = data.load_full_train_data()

    # Ensure stratify label exists
    if "stratify_label" not in df.columns:
        df["stratify_label"] = df[config.CFG.target_cols].idxmax(axis=1)

    # Re-create the Stratified K-Fold split
    skf = StratifiedKFold(
        n_splits=config.CFG.n_folds, shuffle=True, random_state=config.CFG.seed
    )

    # Initialize arrays for OOF predictions
    # Shape: (N_samples, N_classes)
    oof_preds = np.zeros((len(df), config.CFG.num_classes), dtype=np.float32)
    # Track how many models predicted each sample to average later
    sample_counts = np.zeros((len(df),), dtype=np.float32)

    device = torch.device(config.CFG.device)

    # Iterate through all architectures and folds to gather OOF predictions
    for arch in config.CFG.model_architectures:
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(df, df["stratify_label"])
        ):
            model_path = os.path.join(config.CFG.models_dir, f"{arch}_fold_{fold}.pth")

            if not os.path.exists(model_path):
                print(f"Warning: Checkpoint {model_path} not found. Skipping.")
                continue

            # Prepare Validation Data for this fold
            val_df = df.iloc[val_idx].reset_index(drop=True)
            val_dataset = data.AppleDataset(
                val_df, transform=data.get_transforms("valid")
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.CFG.batch_size,
                shuffle=False,
                num_workers=config.CFG.num_workers,
                pin_memory=True,
            )

            # Load Model
            model = modeling.get_model(arch, config.CFG.num_classes, pretrained=False)
            try:
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
                continue

            model.to(device)
            model.eval()

            fold_preds = []

            # Inference with TTA (Original + Horizontal Flip)
            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(device)

                    # Original
                    out_orig = model(images)
                    prob_orig = torch.softmax(out_orig, dim=1)

                    # Flip
                    out_flip = model(torch.flip(images, dims=[3]))
                    prob_flip = torch.softmax(out_flip, dim=1)

                    # Average
                    batch_preds = (prob_orig + prob_flip) / 2.0
                    fold_preds.append(batch_preds.cpu().numpy())

            fold_preds = np.concatenate(fold_preds, axis=0)

            # Accumulate predictions
            # val_idx contains the original indices in the full dataframe
            oof_preds[val_idx] += fold_preds
            sample_counts[val_idx] += 1

            # Clean up
            del model
            torch.cuda.empty_cache()

    # Average the predictions
    # Avoid division by zero
    mask = sample_counts > 0
    oof_preds[mask] /= sample_counts[mask][:, None]

    # Calculate Final Metric
    targets = df[config.CFG.target_cols].values
    try:
        val_auc = roc_auc_score(targets, oof_preds, average="macro", multi_class="ovr")
    except Exception as e:
        print(f"Error calculating ROC AUC: {e}")
        val_auc = 0.0

    print(f"Final Validation Metric: {val_auc}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("Starting Failure Analysis...")

    # Calculate Error Magnitude: 1.0 - Probability assigned to the true class
    # We assume 'targets' is one-hot or soft-label. We take the class with max ground truth.
    true_class_indices = np.argmax(targets, axis=1)
    # Extract predicted probability for the true class
    pred_probs_true = oof_preds[np.arange(len(df)), true_class_indices]
    error_magnitude = 1.0 - pred_probs_true

    # Collect Meta-Features
    widths = []
    heights = []
    intensities = []

    for idx, row in df.iterrows():
        full_path = os.path.join(config.CFG.input_root, row["file_path"])
        img = cv2.imread(full_path)

        if img is not None:
            h, w, c = img.shape
            widths.append(w)
            heights.append(h)
            # Calculate mean intensity (normalized)
            # cv2 is BGR, but mean is same for intensity approx
            intensities.append(img.mean() / 255.0)
        else:
            widths.append(0)
            heights.append(0)
            intensities.append(0)

    widths = np.array(widths)
    heights = np.array(heights)
    intensities = np.array(intensities)

    # Avoid division by zero for aspect ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        aspect_ratios = np.true_divide(widths, heights)
        aspect_ratios[~np.isfinite(aspect_ratios)] = 0

    # Calculate Correlations
    def get_corr(feat_values, errors):
        if np.std(feat_values) < 1e-9:
            return 0.0
        return np.corrcoef(feat_values, errors)[0, 1]

    corr_width = get_corr(widths, error_magnitude)
    corr_height = get_corr(heights, error_magnitude)
    corr_ar = get_corr(aspect_ratios, error_magnitude)
    corr_intensity = get_corr(intensities, error_magnitude)

    print("Correlation between Error Magnitude and Input Features:")
    print(f"  Width: {corr_width:.4f}")
    print(f"  Height: {corr_height:.4f}")
    print(f"  Aspect Ratio: {corr_ar:.4f}")
    print(f"  Intensity: {corr_intensity:.4f}")

    # ==========================================
    # 5. Submission
    # ==========================================
    threshold = 0.9871488489626378

    if val_auc > threshold:
        print(
            f"Validation metric ({val_auc}) exceeds threshold ({threshold}). Generating submission..."
        )
        inference.predict_all_folds(debug=config.CFG.debug)
    else:
        print(
            f"Validation metric ({val_auc}) does not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
