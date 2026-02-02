import os
import sys
import torch
import pandas as pd
import numpy as np
import cv2
from scipy.stats import pointbiserialr

# Import library modules
from library.config import Config
import library.train as train
import library.inference as inference
import library.dataset as dataset
import library.model as model_lib
import library.utils as utils


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration Override
    # --------------------------------------------------------------------------
    # Override epochs to 1 for a fast baseline within the 2-hour limit.
    # The dataset is large (500k+), so 1 epoch is substantial.
    Config.EPOCHS = 1
    Config.DEBUG = False

    # Ensure reproducibility
    utils.seed_everything(Config.SEED)

    print(
        f"Configuration: EPOCHS={Config.EPOCHS}, DEBUG={Config.DEBUG}, DEVICE={Config.DEVICE}"
    )

    # --------------------------------------------------------------------------
    # 2. Training
    # --------------------------------------------------------------------------
    print("Starting training pipeline...")
    # fit returns the best validation score achieved during training
    _ = train.fit(epochs=Config.EPOCHS)

    # --------------------------------------------------------------------------
    # 3. Validation Inference & Metric Calculation
    # --------------------------------------------------------------------------
    print("Running full validation inference for metrics and failure analysis...")

    # Load the best model
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        print("Error: Best model checkpoint not found.")
        sys.exit(1)

    device = Config.DEVICE
    # Initialize model (load_cached_hierarchy=True to use the mapping created during training)
    model = model_lib.get_model(pretrained=False, load_cached_hierarchy=True)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # Get validation loader
    _, val_loader, _ = dataset.get_loaders(load_cached_data=True)

    all_preds = []
    all_targets = []

    # Inference loop
    with torch.no_grad():
        for images, species_ids, genus_ids, family_ids in val_loader:
            images = images.to(device)
            species_ids = species_ids.to(device)

            # Forward pass
            outputs = model(images)
            # We only care about the species head for the main metric
            logits = outputs["species"]
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(species_ids.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute Metric
    val_f1 = utils.get_score(all_targets, all_preds)
    print(f"Final Validation Metric: {val_f1}")

    # --------------------------------------------------------------------------
    # 4. Failure Analysis
    # --------------------------------------------------------------------------
    print("Performing failure analysis...")

    # Calculate error vector (1 = Incorrect, 0 = Correct)
    errors = (all_preds != all_targets).astype(int)

    # We need to correlate errors with image features (Size, Width, Height).
    # Since reading all 133k validation images is slow, we analyze a subset.
    # val_loader.dataset.df contains the file paths in order (shuffle=False for val).
    val_df = val_loader.dataset.df

    subset_size = 5000
    indices = np.arange(min(len(val_df), subset_size))

    file_sizes = []
    widths = []
    heights = []
    aspect_ratios = []
    subset_errors = errors[indices]

    print(f"Extracting features for {len(indices)} validation samples...")

    for idx in indices:
        rel_path = val_df.iloc[idx]["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        try:
            # File Size
            f_size = os.path.getsize(full_path)

            # Dimensions (Use OpenCV for speed, just reading header if possible would be better but cv2 is fast)
            # We read unchanged to avoid conversion overhead, though we still decode
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                h, w = img.shape[:2]
            else:
                h, w = 0, 0

            file_sizes.append(f_size)
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)

        except Exception as e:
            file_sizes.append(0)
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)

    # Calculate Correlations
    # We use Point-Biserial correlation since one variable is binary (error) and others are continuous

    def calc_corr(feature_name, feature_values, error_vec):
        if len(set(error_vec)) < 2:
            return 0.0  # Cannot compute correlation if all are correct or all incorrect

        # Handle potential NaNs or zeros if image read failed
        valid_mask = np.array(feature_values) > 0
        if valid_mask.sum() < 2:
            return 0.0

        f_clean = np.array(feature_values)[valid_mask]
        e_clean = error_vec[valid_mask]

        corr, _ = pointbiserialr(e_clean, f_clean)
        if np.isnan(corr):
            return 0.0
        return corr

    corr_size = calc_corr("File Size", file_sizes, subset_errors)
    corr_width = calc_corr("Width", widths, subset_errors)
    corr_height = calc_corr("Height", heights, subset_errors)
    corr_ar = calc_corr("Aspect Ratio", aspect_ratios, subset_errors)

    print("Correlation between Error Magnitude and Input Features:")
    print(f"  File Size: {corr_size:.4f}")
    print(f"  Width: {corr_width:.4f}")
    print(f"  Height: {corr_height:.4f}")
    print(f"  Aspect Ratio: {corr_ar:.4f}")

    # --------------------------------------------------------------------------
    # 5. Submission
    # --------------------------------------------------------------------------
    threshold = 0.5930838412243743

    if val_f1 > threshold:
        print(
            f"Validation metric ({val_f1}) > Threshold ({threshold}). Generating submission..."
        )
        inference.predict_all(load_cached_data=True)
    else:
        print(
            f"Validation metric ({val_f1}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
