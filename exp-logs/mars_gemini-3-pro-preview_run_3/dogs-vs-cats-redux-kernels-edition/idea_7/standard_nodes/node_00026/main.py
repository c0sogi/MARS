import os
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
import warnings

# Import from provided library
from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_dataloaders, load_metadata
from library.model_factory import create_model
from library.trainer import train_model
from library.inference import run_inference, predict_one_epoch

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def analyze_failures(y_true, y_pred, metadata_df):
    """
    Performs failure analysis by correlating error magnitude with image features.
    """
    print("\n--- Failure Analysis ---")

    # Calculate error magnitude
    errors = np.abs(y_true - y_pred)

    # Extract features from images
    widths = []
    heights = []
    aspect_ratios = []
    file_sizes = []

    print("Extracting image features for validation set...")

    # metadata_df['filepath'] is absolute path as per load_metadata implementation
    for filepath in metadata_df["filepath"]:
        try:
            # File size
            fsize = os.path.getsize(filepath)
            file_sizes.append(fsize)

            # Dimensions
            img = cv2.imread(filepath)
            if img is not None:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h if h > 0 else 0)
            else:
                widths.append(0)
                heights.append(0)
                aspect_ratios.append(0)
        except Exception:
            widths.append(0)
            heights.append(0)
            file_sizes.append(0)
            aspect_ratios.append(0)

    # Calculate correlations
    features = {
        "Width": widths,
        "Height": heights,
        "Aspect Ratio": aspect_ratios,
        "File Size": file_sizes,
    }

    print("Correlation between Error Magnitude and Input Features:")
    for name, values in features.items():
        if len(values) != len(errors):
            continue

        # Pearson correlation using numpy
        if np.std(values) == 0 or np.std(errors) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(values, errors)[0, 1]

        print(f"{name}: {corr:.10f}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()

    print(f"Starting pipeline on {device}")

    # 2. Data Loading
    # Load cached data if available
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    # 3. Training
    trained_models = []
    for model_name in Config.MODEL_BACKBONES:
        print(f"\n=== Training Model: {model_name} ===")

        # Check for existing checkpoint to enable resumability (Cite debug_lesson_3)
        checkpoint_path = os.path.join(Config.WORKING_DIR, f"{model_name}_best.pth")
        if os.path.exists(checkpoint_path):
            print(f"Checkpoint found at {checkpoint_path}. Skipping training.")
            trained_models.append(model_name)
            continue

        # train_model handles training, saving best checkpoint, and returns path
        train_model(model_name, train_loader, val_loader)
        trained_models.append(model_name)

    # 4. Validation & Ensemble Evaluation
    print("\n=== Validating Ensemble ===")

    # Load validation metadata for ground truth
    val_df = load_metadata("val", load_cached_data=True)
    y_true = val_df["label"].values

    # Accumulate predictions
    ensemble_probs = np.zeros(len(y_true))

    for model_name in trained_models:
        print(f"Generating validation predictions for {model_name}...")

        # Load model architecture
        model = create_model(model_name, pretrained=False, num_classes=1)
        model.to(device)

        # Load weights
        ckpt_path = os.path.join(Config.WORKING_DIR, f"{model_name}_best.pth")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)

        # Predict
        # predict_one_epoch returns (ids, probs). For val_loader, ids are labels.
        _, probs = predict_one_epoch(model, val_loader, device, use_tta=Config.USE_TTA)

        ensemble_probs += probs

    # Average predictions
    ensemble_probs /= len(trained_models)

    # Calculate Log Loss (Clip to avoid log(0))
    ensemble_probs_clipped = np.clip(ensemble_probs, 1e-15, 1 - 1e-15)
    metric = log_loss(y_true, ensemble_probs_clipped)

    # Print Metric
    print(f"Final Validation Metric: {metric}")

    # 5. Failure Analysis
    analyze_failures(y_true, ensemble_probs, val_df)

    # 6. Submission
    THRESHOLD = 0.009311713870561527

    if metric < THRESHOLD:
        print(
            f"\nMetric {metric} meets threshold {THRESHOLD}. Generating submission..."
        )
        run_inference(load_cached_data=True)
    else:
        print(
            f"\nMetric {metric} does NOT meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
