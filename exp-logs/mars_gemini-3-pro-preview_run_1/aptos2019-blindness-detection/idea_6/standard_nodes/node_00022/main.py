import os
import sys
import cv2
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, quadratic_weighted_kappa
from library.dataset import get_dataloaders
from library.model import run_training, generate_submission, EfficientNetV2Ordinal


def get_predictions(model, loader, device):
    """
    Runs inference on a loader and returns true labels and predicted scores.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            scores = probs.sum(dim=1)

            # Reconstruct integer class from ordinal vector for targets
            # targets shape: (batch, 4) -> sum gives class
            true_labels = targets.sum(dim=1)

            all_preds.append(scores.cpu().numpy())
            all_targets.append(true_labels.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def perform_failure_analysis(val_df, y_true, y_pred):
    """
    Analyzes the correlation between model error and image meta-features.
    """
    print("\n=== Failure Analysis ===")

    # Calculate errors (using rounded predictions for class distance)
    y_pred_rounded = np.rint(y_pred).astype(int)
    errors = np.abs(y_true - y_pred_rounded)

    # Collect meta-features
    widths = []
    heights = []
    aspect_ratios = []
    mean_intensities = []
    file_sizes = []

    input_dir = Config.INPUT_DIR

    # Iterate through validation samples to extract features
    # This might take a minute but is necessary for the analysis
    for idx, row in val_df.iterrows():
        file_path = os.path.join(input_dir, row["file_path"])

        try:
            # File size
            if os.path.exists(file_path):
                file_sizes.append(os.path.getsize(file_path))

                # Image stats
                img = cv2.imread(file_path)
                if img is not None:
                    h, w, c = img.shape
                    widths.append(w)
                    heights.append(h)
                    aspect_ratios.append(w / h if h > 0 else 0)
                    mean_intensities.append(img.mean())
                else:
                    # Fallback
                    widths.append(0)
                    heights.append(0)
                    aspect_ratios.append(0)
                    mean_intensities.append(0)
            else:
                file_sizes.append(0)
                widths.append(0)
                heights.append(0)
                aspect_ratios.append(0)
                mean_intensities.append(0)

        except Exception:
            file_sizes.append(0)
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)
            mean_intensities.append(0)

    # Create DataFrame for correlation
    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "width": widths,
            "height": heights,
            "aspect_ratio": aspect_ratios,
            "mean_intensity": mean_intensities,
            "file_size": file_sizes,
        }
    )

    # Compute correlations
    features = ["width", "height", "aspect_ratio", "mean_intensity", "file_size"]
    print("Correlation between Error Magnitude and Input Features:")
    for feat in features:
        if analysis_df[feat].std() > 0:
            corr, _ = pearsonr(analysis_df[feat], analysis_df["error"])
            print(f"{feat}: {corr:.4f}")
        else:
            print(f"{feat}: NaN (No variance)")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # Using default batch size and workers from Config
    train_loader, val_loader, test_loader = get_dataloaders(debug=False)

    # 3. Training
    # run_training handles the loop, validation, and saving best model
    print("Starting training...")
    best_val_qwk = run_training(
        train_loader,
        val_loader,
        epochs=Config.EPOCHS,
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # 4. Final Validation & Metric
    print("\nLoading best model for final validation...")
    model = EfficientNetV2Ordinal()
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Error: Best model checkpoint not found.")
        return

    model = model.to(device)

    # Get predictions on validation set
    y_pred_raw, y_true = get_predictions(model, val_loader, device)

    # Calculate QWK
    final_metric = quadratic_weighted_kappa(y_true, y_pred_raw)

    # Print required metric format
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    val_df = pd.read_csv(Config.VAL_META)
    perform_failure_analysis(val_df, y_true, y_pred_raw)

    # 6. Conditional Submission
    THRESHOLD = 0.922975135423079

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(test_loader, model_path=Config.BEST_MODEL_PATH)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
