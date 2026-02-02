import os
import cv2
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, compute_qwk
from library.engine import get_dataloaders, train, generate_submission
from library.model import ResNet18Regression

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_failure_analysis(model, val_loader, val_df, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlations between prediction error and image meta-features.
    """
    print("\nStarting Failure Analysis...")

    model.eval()
    all_preds = []
    all_labels = []

    # 1. Get Model Predictions (Continuous)
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.view(-1)

            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    # Calculate Error Magnitude (Absolute Error)
    # We use continuous predictions for more granular error analysis
    errors = np.abs(y_true - y_pred)

    # 2. Extract Image Meta-Features
    # We need to read the images again to get their original properties
    widths = []
    heights = []
    aspect_ratios = []
    mean_intensities = []

    # Ensure val_df is aligned with the loader (dataloaders are sequential for val)
    # The val_loader from get_dataloaders uses shuffle=False

    print("Extracting meta-features from validation images...")
    for idx, row in val_df.iterrows():
        # Construct path (metadata paths are relative to input root)
        # Note: val_df comes from metadata/val.csv which has 'file_path'
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            img = cv2.imread(full_path)
            if img is None:
                # Fallback for safety, though data should be clean
                widths.append(0)
                heights.append(0)
                aspect_ratios.append(0)
                mean_intensities.append(0)
                continue

            h, w, c = img.shape
            # Convert to RGB for intensity calculation to match training logic
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0)
            mean_intensities.append(img_rgb.mean() / 255.0)

        except Exception as e:
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)
            mean_intensities.append(0)

    # 3. Calculate Correlations
    meta_data = {
        "width": widths,
        "height": heights,
        "aspect_ratio": aspect_ratios,
        "mean_intensity": mean_intensities,
    }

    print("-" * 30)
    print("Correlation between Error Magnitude and Input Features (Spearman):")
    for feature_name, feature_values in meta_data.items():
        if len(feature_values) != len(errors):
            print(f"Warning: Length mismatch for {feature_name}. Skipping.")
            continue

        # Spearman correlation captures monotonic relationships (linear or non-linear)
        corr, p_val = spearmanr(errors, feature_values)
        print(f"{feature_name}: {corr:.4f} (p-value: {p_val:.4f})")
    print("-" * 30)

    return y_true, y_pred


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    # We use debug=False to train on the full dataset (it's small enough for a fast run)
    train_loader, val_loader, test_loader, test_df = get_dataloaders(debug=False)

    # Load validation dataframe separately for failure analysis file paths
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # 3. Training
    print("Starting training...")
    # The train function handles the loop, validation, and saving best model
    model = train(train_loader, val_loader, epochs=Config.NUM_EPOCHS, device=device)

    # 4. Final Validation Assessment
    print("Performing final validation assessment...")
    # We manually calculate QWK here to ensure we print it exactly as required
    # and to reuse the predictions for failure analysis if needed.
    # However, run_failure_analysis retrieves predictions too.

    # Get predictions and true labels
    y_true, y_pred_continuous = run_failure_analysis(model, val_loader, val_df, device)

    # Calculate QWK using the provided utility
    # compute_qwk handles clipping and rounding internally
    final_metric = compute_qwk(y_true, y_pred_continuous)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # 5. Submission
    if final_metric > 0.8997:
        print("Generating submission...")
        generate_submission(
            model,
            test_loader,
            test_df,
            device=device,
            output_path=Config.SUBMISSION_PATH,
        )
    else:
        print(
            f"Validation metric {final_metric} did not meet threshold 0.8997. Skipping submission."
        )

    print("Workflow completed successfully.")


if __name__ == "__main__":
    main()
