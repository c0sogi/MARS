import sys
import os
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Add current directory to path
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.model import PlantClassifier
from library.engine import train_model, validate, predict
from library.loss import FocalLoss


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for fast baseline execution
    # 3 epochs is sufficient for a baseline on this large dataset with pretraining
    Config.NUM_EPOCHS = 3

    # Ensure reproducibility
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    print("Loading datasets...")
    train_loader, val_loader, test_loader, classes_list = get_dataloaders(
        load_cached_data=True
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("Initializing PlantClassifier (EfficientNet-B0)...")
    model = PlantClassifier(num_classes=Config.NUM_CLASSES).to(device)

    # ==========================================
    # 4. Training
    # ==========================================
    print("Starting training...")
    # train_model handles the loop, validation, and saving best model
    model = train_model(
        model, train_loader, val_loader, device, num_epochs=Config.NUM_EPOCHS
    )

    # ==========================================
    # 5. Final Validation & Metric
    # ==========================================
    print("Performing final validation...")
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA).to(device)

    # Calculate metric on the hold-out validation set
    val_loss, val_f1 = validate(model, val_loader, criterion, device)

    # REQUIRED: Print the final validation metric
    print(f"Final Validation Metric: {val_f1}")

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    print("Performing failure analysis...")

    # Extract predictions and targets from validation set
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate binary error (1 = Incorrect, 0 = Correct)
    errors = (all_preds != all_targets).astype(int)

    # Load validation metadata to retrieve features
    val_df = pd.read_csv(Config.VAL_CSV)

    # Ensure alignment (val_loader is sequential, so order matches val_df)
    if len(val_df) != len(errors):
        # Handle edge case where loader drops last batch or similar
        min_len = min(len(val_df), len(errors))
        val_df = val_df.iloc[:min_len]
        errors = errors[:min_len]
        all_targets = all_targets[:min_len]

    # Feature 1: Region ID
    regions = val_df["region_id"].values

    # Feature 2: Class Frequency (from training data)
    # Hypothesis: Model fails more on rare classes
    train_df = pd.read_csv(Config.TRAIN_CSV)
    class_counts = train_df["category_id"].value_counts().to_dict()

    # Map model targets back to original category_ids to look up frequency
    target_category_ids = [classes_list[t] for t in all_targets]
    frequencies = np.array(
        [class_counts.get(cat_id, 0) for cat_id in target_category_ids]
    )

    # Calculate Correlations
    if len(np.unique(regions)) > 1:
        corr_region, _ = pearsonr(errors, regions)
        print(f"Correlation between Error and Region ID: {corr_region:.4f}")
    else:
        print("Correlation between Error and Region ID: N/A (Constant Region)")

    if len(np.unique(frequencies)) > 1:
        corr_freq, _ = pearsonr(errors, frequencies)
        print(f"Correlation between Error and Class Frequency: {corr_freq:.4f}")
    else:
        print("Correlation between Error and Class Frequency: N/A (Constant Frequency)")

    # ==========================================
    # 7. Submission
    # ==========================================
    print("Generating submission...")
    predict(model, test_loader, classes_list, device)
    print("Runfile execution completed.")


if __name__ == "__main__":
    main()
