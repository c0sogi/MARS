import os
import sys
import pandas as pd
import numpy as np
import torch
import cv2
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add current directory to sys.path
sys.path.append(".")

# Import from provided library files
from library.config import Config
from library.train import run_training
from library.inference import run_inference
from library.utils import seed_everything, calculate_metric
from library.model import AppleEfficientNet
from library.data import get_loaders, load_dataset_df


def analyze_failures(val_df, val_preds, val_labels):
    """
    Performs failure analysis by correlating error magnitude with image meta-features.
    """
    print("\nPerforming Failure Analysis...")

    # Convert tensors to numpy
    if isinstance(val_preds, torch.Tensor):
        val_preds = val_preds.cpu().numpy()
    if isinstance(val_labels, torch.Tensor):
        val_labels = val_labels.cpu().numpy()

    # Calculate Error Magnitude
    # For each sample, error = 1.0 - probability assigned to the true class
    true_class_indices = np.argmax(val_labels, axis=1)
    prob_of_true_class = val_preds[np.arange(len(val_preds)), true_class_indices]
    error_magnitude = 1.0 - prob_of_true_class

    # Extract Meta Features for Correlation Analysis
    meta_stats = []

    for idx, row in val_df.iterrows():
        path = row["full_path"]

        # Default values
        w, h, intensity, f_size = 0, 0, 0, 0
        ar = 0

        if os.path.exists(path):
            f_size = os.path.getsize(path)
            img = cv2.imread(path)
            if img is not None:
                h, w, c = img.shape
                intensity = img.mean()
                ar = w / h if h > 0 else 0

        meta_stats.append(
            {
                "width": w,
                "height": h,
                "aspect_ratio": ar,
                "mean_intensity": intensity,
                "file_size": f_size,
                "error_magnitude": error_magnitude[idx],
            }
        )

    meta_df = pd.DataFrame(meta_stats)

    # Calculate and print correlations
    print("Correlation between Error Magnitude and Input Features:")
    features = ["width", "height", "aspect_ratio", "mean_intensity", "file_size"]

    for feat in features:
        if feat in meta_df.columns:
            # Check for constant columns to avoid NaN correlation
            if meta_df[feat].std() == 0:
                corr = 0.0
            else:
                corr = meta_df[feat].corr(meta_df["error_magnitude"])
            print(f"  {feat}: {corr:.4f}")


def main():
    # 1. Run Training
    # This uses the Config parameters (20 epochs, EfficientNet-B4)
    # It saves the best model to Config.BEST_MODEL_PATH
    run_training(load_cached_data=True)

    # 2. Validation Assessment
    print("\nStarting Validation Assessment...")
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Validation Data
    # We load the dataframe to get file paths for failure analysis
    val_df = load_dataset_df(
        Config.VAL_METADATA_PATH, "val_cache.parquet", load_cached_data=True
    )

    # Get DataLoader (ensuring shuffle=False to align with dataframe)
    # We discard the train_loader returned by get_loaders
    _, val_loader = get_loaders(load_cached_data=True)

    # Load Model
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Error: Model weights not found at {Config.BEST_MODEL_PATH}")
        return

    # Initialize model (pretrained=False to save time loading weights, we load our own)
    model = AppleEfficientNet(
        model_name=Config.MODEL_NAME,
        pretrained=False,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE,
    )

    state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Inference on Validation Set
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.softmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Calculate Metric
    metric = calculate_metric(all_labels, all_preds)

    # Print Metric (Full Precision)
    print(f"Final Validation Metric: {metric}")

    # 3. Failure Analysis
    analyze_failures(val_df, all_preds, all_labels)

    # 4. Submission Generation
    # Threshold defined in task
    THRESHOLD = 0.9827222308753782

    if metric > THRESHOLD:
        print(f"\nValidation metric ({metric}) exceeds threshold ({THRESHOLD}).")
        print("Generating submission...")
        run_inference(load_cached_data=True)
    else:
        print(
            f"\nValidation metric ({metric}) does not exceed threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
