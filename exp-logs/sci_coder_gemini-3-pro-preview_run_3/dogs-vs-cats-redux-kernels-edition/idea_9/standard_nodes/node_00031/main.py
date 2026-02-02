import os
import cv2
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import log_loss

from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.engine import fit_model, predict

# --- Configuration ---
# Triple Heterogeneous Ensemble with Decoupled Resolutions
MODELS_CONFIG = [
    {"name": "convnext_base", "res": 320},
    {"name": "resnet101", "res": 256},
    {"name": "swin_s", "res": 224},
]

# Training Hyperparameters
BATCH_SIZE = 24
EPOCHS = 5
LEARNING_RATE = 1e-4
THRESHOLD = 0.009241249605204765


def predict_val(model, loader, device):
    """
    Custom inference loop for validation data.
    Performs Test Time Augmentation (Horizontal Flip).
    Returns flat array of probabilities.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for images, _ in loader:  # Validation loader yields (image, label)
            images = images.to(device)

            # 1. Original Forward Pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            # 2. TTA: Horizontal Flip
            images_flipped = torch.flip(images, dims=[3])
            logits_flipped = model(images_flipped)
            probs_flipped = torch.sigmoid(logits_flipped)

            # Average
            avg_probs = (probs + probs_flipped) / 2.0
            preds.append(avg_probs.cpu().numpy())

    return np.concatenate(preds).flatten()


def analyze_failures(val_df, y_true, y_pred):
    """
    Performs failure analysis by correlating errors with image metadata.
    """
    print("\n--- Failure Analysis ---")

    # Calculate absolute error
    errors = np.abs(y_true - y_pred)

    # Collect metadata features
    file_sizes = []
    widths = []
    heights = []

    input_dir = "./input"

    # Iterate through validation set to get image stats
    # Using a loop is fast enough for 4500 images
    for _, row in val_df.iterrows():
        path = os.path.join(input_dir, row["filepath"])
        try:
            # File Size
            file_sizes.append(os.path.getsize(path))

            # Dimensions
            img = cv2.imread(path)
            if img is not None:
                h, w, _ = img.shape
                widths.append(w)
                heights.append(h)
            else:
                widths.append(0)
                heights.append(0)
        except Exception:
            file_sizes.append(0)
            widths.append(0)
            heights.append(0)

    # Convert to numpy for correlation
    file_sizes = np.array(file_sizes)
    widths = np.array(widths)
    heights = np.array(heights)

    # Filter out invalid reads (size 0 or width 0)
    valid_mask = (widths > 0) & (file_sizes > 0)

    if np.sum(valid_mask) > 10:  # Ensure enough samples
        corr_size, _ = pearsonr(errors[valid_mask], file_sizes[valid_mask])
        corr_w, _ = pearsonr(errors[valid_mask], widths[valid_mask])
        corr_h, _ = pearsonr(errors[valid_mask], heights[valid_mask])

        print(f"Error Correlation with File Size: {corr_size:.8f}")
        print(f"Error Correlation with Width: {corr_w:.8f}")
        print(f"Error Correlation with Height: {corr_h:.8f}")
    else:
        print("Insufficient valid image data for correlation analysis.")


def main():
    seed_everything(42)
    device = get_device()

    trained_models = []

    # -------------------------------------------------------------------------
    # 1. Training Phase
    # -------------------------------------------------------------------------
    print("Starting Ensemble Training...")

    for cfg in MODELS_CONFIG:
        print(f"\nTraining Model: {cfg['name']} @ {cfg['res']}x{cfg['res']}")

        # Train model using engine
        model = fit_model(
            model_name=cfg["name"],
            resolution=cfg["res"],
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            debug_subset=None,  # Use full dataset for best performance
        )
        trained_models.append((cfg, model))

    # -------------------------------------------------------------------------
    # 2. Validation Phase (Ensemble)
    # -------------------------------------------------------------------------
    print("\nStarting Ensemble Validation...")

    # Load ground truth
    val_df = pd.read_csv("./metadata/val.csv")
    y_true = val_df["label"].values

    # Accumulate predictions
    ensemble_preds = np.zeros(len(val_df))

    for cfg, model in trained_models:
        # Get validation loader for specific resolution
        _, val_loader, _ = get_dataloaders(
            batch_size=BATCH_SIZE, resolution=cfg["res"], load_cached_data=True
        )

        # Predict
        preds = predict_val(model, val_loader, device)
        ensemble_preds += preds

    # Average predictions
    ensemble_preds /= len(trained_models)

    # Compute Metric
    final_metric = log_loss(y_true, ensemble_preds)
    print(f"Final Validation Metric: {final_metric:.16f}")

    # -------------------------------------------------------------------------
    # 3. Failure Analysis
    # -------------------------------------------------------------------------
    analyze_failures(val_df, y_true, ensemble_preds)

    # -------------------------------------------------------------------------
    # 4. Submission Phase
    # -------------------------------------------------------------------------
    if final_metric < THRESHOLD:
        print("\nThreshold passed. Generating Test Submission...")

        test_ids = None
        ensemble_test_preds = None

        for i, (cfg, model) in enumerate(trained_models):
            # Get test loader for specific resolution
            _, _, test_loader = get_dataloaders(
                batch_size=BATCH_SIZE, resolution=cfg["res"], load_cached_data=True
            )

            # Predict using engine's predict (handles IDs and TTA)
            ids, probs = predict(model, test_loader, device, use_tta=True)

            if i == 0:
                ensemble_test_preds = probs
                test_ids = ids
            else:
                ensemble_test_preds += probs

        # Average predictions
        ensemble_test_preds /= len(trained_models)

        # Save Submission
        os.makedirs("./submission", exist_ok=True)
        sub_df = pd.DataFrame({"id": test_ids, "label": ensemble_test_preds})
        sub_path = "./submission/submission.csv"
        sub_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(
            f"\nMetric {final_metric:.16f} did not beat threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
