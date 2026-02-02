import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.utils import set_seed
from library.dataset import get_datasets
from library.model import ShallowCNN
from library.engine import train_model, validate, predict_and_submit


def perform_failure_analysis(model, val_loader, device):
    """
    Analyzes model errors on the validation set by correlating error magnitude
    with image meta-features (brightness, contrast, channel means).
    """
    print("\n--- Performing Failure Analysis ---")
    model.eval()

    errors = []
    meta_features = {
        "brightness": [],
        "contrast": [],
        "red_mean": [],
        "green_mean": [],
        "blue_mean": [],
    }

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Get predictions
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

            # Calculate absolute error per sample
            # labels shape: (B, 1), probs shape: (B, 1)
            batch_errors = torch.abs(probs - labels).cpu().numpy().flatten()
            errors.extend(batch_errors)

            # Extract image statistics from the tensor batch
            # inputs shape: (B, 3, 32, 32)
            # We compute stats per image in the batch

            # Move to CPU for numpy operations
            imgs_np = inputs.cpu().numpy()

            for img in imgs_np:
                # img shape: (3, 32, 32)
                # Global stats
                meta_features["brightness"].append(np.mean(img))
                meta_features["contrast"].append(np.std(img))

                # Channel stats
                meta_features["red_mean"].append(np.mean(img[0]))
                meta_features["green_mean"].append(np.mean(img[1]))
                meta_features["blue_mean"].append(np.mean(img[2]))

    errors = np.array(errors)

    print("Correlation between Error Magnitude and Image Features:")
    for feature_name, values in meta_features.items():
        values = np.array(values)
        # Calculate Pearson correlation
        corr, _ = pearsonr(errors, values)
        print(f"{feature_name}: {corr:.4f}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    # We use the full dataset (debug=False) because the dataset is small (32x32 images)
    # and training will be fast even with the full set.
    train_ds, val_ds, test_ds = get_datasets(debug=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    # 3. Model Initialization
    model = ShallowCNN()

    # 4. Training
    # train_model handles the loop, validation, and early stopping.
    # It returns the model with the best weights loaded.
    model = train_model(model, train_loader, val_loader, Config)

    # 5. Final Validation Metric
    # We run validation one last time to ensure we print the exact metric required.
    criterion = nn.BCEWithLogitsLoss()
    _, final_auc = validate(model, val_loader, criterion, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # 6. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 7. Submission
    predict_and_submit(model, test_loader, Config.SUBMISSION_PATH, device)


if __name__ == "__main__":
    main()
