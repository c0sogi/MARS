import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import tqdm


# ==========================================
# 1. Suppress Progress Bars (TQDM)
# ==========================================
# We define a dummy class to replace tqdm, ensuring silent execution
# while maintaining compatibility with methods like set_postfix.
class DummyTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable if iterable is not None else []

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs):
        pass

    def set_description(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def close(self):
        pass


# Patch tqdm before importing library modules
tqdm.tqdm = DummyTqdm

# ==========================================
# 2. Imports from Library
# ==========================================
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import AppleDiseaseModel
from library.engine import train_model, predict_and_submit, reconstruct_probabilities


# ==========================================
# 3. Main Execution
# ==========================================
def main():
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    print("Starting training phase...")
    # Train the model (uses Config for hyperparameters)
    # This will save the best model to Config.BEST_MODEL_PATH
    train_model()

    print("Training complete. Starting validation and failure analysis...")

    # Load the best model for validation
    device = torch.device(Config.DEVICE)
    model = AppleDiseaseModel(pretrained=False)
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Get DataLoaders (using cached data if available)
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Containers for predictions and targets
    all_preds_2d = []
    all_targets_2d = []

    # Inference on Validation Set
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            all_preds_2d.append(probs.cpu().numpy())
            all_targets_2d.append(targets.numpy())

    # Concatenate results
    all_preds_2d = np.concatenate(all_preds_2d)
    all_targets_2d = np.concatenate(all_targets_2d)

    # Reconstruct 4-class probabilities (Healthy, Multiple, Rust, Scab)
    # The model outputs [Rust, Scab] probabilities.
    # We use the mathematical decomposition to reconstruct the full distribution.
    pred_4c = reconstruct_probabilities(all_preds_2d)
    target_4c = reconstruct_probabilities(all_targets_2d)

    # Calculate Mean Column-wise ROC AUC
    try:
        val_auc = roc_auc_score(target_4c, pred_4c, average="macro")
    except ValueError:
        val_auc = 0.5

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {val_auc:.16f}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("\nPerforming Failure Analysis...")

    # Calculate error magnitude per sample
    # We use Mean Absolute Error (MAE) across the 4 classes as the error metric
    errors = np.mean(np.abs(target_4c - pred_4c), axis=1)

    # Extract Meta-Features for Validation Set
    # We access the dataframe directly from the dataset
    val_df = val_loader.dataset.df

    meta_features = {"File Size": [], "Width": [], "Height": [], "Mean Intensity": []}

    # Iterate through validation images to extract features
    # Note: The DataLoader iterates sequentially, so order matches val_df if shuffle=False
    for _, row in val_df.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            # File Size
            f_size = os.path.getsize(full_path)

            # Image Stats (Dimensions, Intensity)
            img = cv2.imread(full_path)
            if img is not None:
                h, w, c = img.shape
                intensity = img.mean()
            else:
                h, w, intensity = 0, 0, 0

            meta_features["File Size"].append(f_size)
            meta_features["Width"].append(w)
            meta_features["Height"].append(h)
            meta_features["Mean Intensity"].append(intensity)

        except Exception:
            # Fallback for missing files
            meta_features["File Size"].append(0)
            meta_features["Width"].append(0)
            meta_features["Height"].append(0)
            meta_features["Mean Intensity"].append(0)

    # Calculate Correlations
    print("Correlation between Error Magnitude and Input Features:")
    for feature_name, values in meta_features.items():
        values = np.array(values)
        if np.std(values) > 0:
            corr, _ = pearsonr(errors, values)
            print(f"  {feature_name}: {corr:.4f}")
        else:
            print(f"  {feature_name}: NaN (No variance)")

    # ==========================================
    # 5. Submission
    # ==========================================
    THRESHOLD = 0.9902480620249655

    if val_auc > THRESHOLD:
        print(
            f"\nValidation metric ({val_auc:.6f}) exceeds threshold ({THRESHOLD:.6f})."
        )
        print("Generating submission...")
        predict_and_submit()
    else:
        print(
            f"\nValidation metric ({val_auc:.6f}) does not meet threshold ({THRESHOLD:.6f})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
