import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import BraTSDataset
from library.model import SiameseEfficientNet
from library.engine import run_training, predict_and_submit


def perform_failure_analysis(model, val_loader, val_dataset):
    """
    Analyzes model performance on the validation set.
    Computes correlations between error magnitude and input metadata features.
    """
    device = torch.device(Config.DEVICE)
    model.eval()

    all_preds = []
    all_targets = []

    # Inference on validation set
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate Error
    errors = np.abs(all_targets - all_preds)

    # Extract Metadata Features for Correlation Analysis
    # We will count files in the directories used by the validation set
    # to see if 'number of slices' correlates with error.
    meta_features = []

    # Access the dataframe used by the dataset
    df = val_dataset.df

    print("\nExtracting metadata for failure analysis...")

    for idx, row in df.iterrows():
        # Get file counts for the modalities
        feats = {"error": errors[idx], "target": all_targets[idx]}

        for mod in Config.SELECTED_MODALITIES:
            # Reconstruct directory path from the cached file path
            # The cache has specific slice paths, we want the parent folder count
            # Example cache entry: input/train/00000/FLAIR/Image-100.dcm
            # We want to count files in input/train/00000/FLAIR

            # We can look up the original metadata paths if needed, but
            # let's infer from the cached path or use the row if available.
            # The cached dataframe has columns like 'FLAIR_0.45'.

            # Pick one depth to get the folder
            sample_file = row.get(f"{mod}_{Config.SLICE_DEPTHS[0]}")

            count = 0
            if (
                sample_file
                and isinstance(sample_file, str)
                and os.path.exists(sample_file)
            ):
                folder = os.path.dirname(sample_file)
                # Fast count
                try:
                    count = len([f for f in os.listdir(folder) if f.endswith(".dcm")])
                except:
                    count = 0

            feats[f"{mod}_count"] = count

        meta_features.append(feats)

    df_analysis = pd.DataFrame(meta_features)

    # Calculate Correlation
    print("\nFailure Analysis: Correlation between Error and Features")
    correlations = df_analysis.corr()["error"].sort_values(ascending=False)
    print(correlations)

    return roc_auc_score(all_targets, all_preds)


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Adjust Config for Fast Baseline
    # We reduce epochs to ensure it fits within the "fast" requirement
    # while still allowing the Siamese network to learn.
    Config.EPOCHS = 15

    print(f"Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Model: {Config.BACKBONE}")

    # 2. Data Loading
    print("\nInitializing Datasets...")
    # Train Dataset
    train_dataset = BraTSDataset(split="train", load_cached_data=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # Validation Dataset
    val_dataset = BraTSDataset(split="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # Test Dataset (Prepared for later)
    test_dataset = BraTSDataset(split="test", load_cached_data=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    print(f"  Train size: {len(train_dataset)}")
    print(f"  Val size: {len(val_dataset)}")
    print(f"  Test size: {len(test_dataset)}")

    # 3. Training
    print("\nStarting Training Run...")
    # run_training returns the best AUC achieved during training loop validation
    best_val_auc_training = run_training(train_loader, val_loader)

    # 4. Validation Assessment & Failure Analysis
    print("\nPerforming Final Validation and Failure Analysis...")

    # Load the best model
    device = torch.device(Config.DEVICE)
    model = SiameseEfficientNet(pretrained=False)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        model.to(device)
    else:
        print("Error: Model checkpoint not found!")
        return

    # Run Analysis
    final_metric = perform_failure_analysis(model, val_loader, val_dataset)

    # Print Required Metric
    print(f"Final Validation Metric: {final_metric}")

    # 5. Submission
    THRESHOLD = 0.6705454545454544

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(test_loader, test_dataset.df)
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
