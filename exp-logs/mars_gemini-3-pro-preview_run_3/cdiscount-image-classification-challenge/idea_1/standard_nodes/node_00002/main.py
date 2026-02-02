import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import (
    TRAIN_BSON,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_BSON,
    TEST_META_PATH,
    WORKING_DIR,
    SUBMISSION_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE,
    NUM_CLASSES,
    SEED,
    seed_everything,
)
from library.utils import get_accuracy
from library.dataset import CdiscountDataset, get_transforms
from library.model import MobileNetV2Classifier
from library.engine import train_model, generate_predictions


def main():
    # 1. Setup
    seed_everything(SEED)
    print(f"Running on device: {DEVICE}")

    # 2. Data Preparation
    # To ensure the baseline runs quickly (within 2 hours), we train on a subset.
    # The bottleneck is likely the test inference on 700k+ items.
    TRAIN_SUBSET_SIZE = 50000

    print("Loading training metadata...")
    train_df = pd.read_csv(TRAIN_META_PATH)

    # Create a random subset for fast training
    if len(train_df) > TRAIN_SUBSET_SIZE:
        print(f"Subsampling training data to {TRAIN_SUBSET_SIZE} samples.")
        train_subset = train_df.sample(n=TRAIN_SUBSET_SIZE, random_state=SEED)
    else:
        train_subset = train_df

    subset_csv_path = os.path.join(WORKING_DIR, "train_subset.csv")
    train_subset.to_csv(subset_csv_path, index=False)

    # Datasets
    print("Initializing datasets...")
    train_dataset = CdiscountDataset(
        metadata_path=subset_csv_path,
        bson_path=TRAIN_BSON,
        mode="train",
        transform=get_transforms("train"),
    )

    val_dataset = CdiscountDataset(
        metadata_path=VAL_META_PATH,
        bson_path=TRAIN_BSON,
        mode="val",
        transform=get_transforms("val"),
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = MobileNetV2Classifier(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(DEVICE)

    # 4. Training
    # Train for 1 epoch on the subset to establish a baseline
    print("Starting training...")
    model = train_model(
        model,
        train_loader,
        val_loader,
        epochs=1,
        device=DEVICE,
        patience=1,
        load_cached_weights=True,
    )

    # 5. Validation & Failure Analysis
    print("Performing validation and failure analysis...")
    model.eval()

    all_preds = []
    all_labels = []

    # We need to collect predictions manually to align with metadata for analysis
    # Note: val_loader is not shuffled, so order matches val_dataset.metadata
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate Metric
    acc = get_accuracy(all_preds, all_labels)
    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {acc}")

    # Failure Analysis
    # Load validation metadata to get features
    val_df = pd.read_csv(VAL_META_PATH)

    # Ensure lengths match
    if len(val_df) != len(all_preds):
        print(
            f"Warning: Metadata length ({len(val_df)}) does not match predictions ({len(all_preds)}). Skipping detailed correlation."
        )
    else:
        # Calculate Error (1 for incorrect, 0 for correct)
        errors = (all_preds != all_labels).astype(int)

        # Feature: BSON Length (proxy for file size / amount of data)
        bson_lengths = val_df["bson_length"].values

        # Calculate Correlation
        if np.std(errors) == 0 or np.std(bson_lengths) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(errors, bson_lengths)[0, 1]

        print(f"Correlation between Error and BSON Length: {corr}")

    # 6. Submission
    print("Generating submission for test set...")
    test_dataset = CdiscountDataset(
        metadata_path=TEST_META_PATH,
        bson_path=TEST_BSON,
        mode="test",
        transform=get_transforms("test"),
    )

    # Batch size must be 1 for test mode as it returns variable sized image stacks
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    generate_predictions(model, test_loader, DEVICE, SUBMISSION_PATH)
    print("Process completed.")


if __name__ == "__main__":
    main()
