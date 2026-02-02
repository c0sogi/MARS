import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel

# Import provided library modules
from library.config import Config
from library import dataset, networks, engine


def main():
    # ==========================================
    # 1. Setup and Configuration
    # ==========================================
    print("--- Setting up Demo Environment ---")

    # Set random seed for reproducibility
    engine.set_seed(42)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Override Config parameters for a fast demo execution
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo

    # Ensure directories exist
    Config.setup()

    # ==========================================
    # 2. Data Loading and Processing
    # ==========================================
    print("\n--- Loading and Processing Data ---")

    # Load Training Data
    # dataset.load_data handles caching automatically
    train_imgs, train_angs, train_lbls, train_ids = dataset.load_data(
        Config.TRAIN_META_PATH, Config.TRAIN_JSON, "train"
    )

    # Load Validation Data
    val_imgs, val_angs, val_lbls, val_ids = dataset.load_data(
        Config.VAL_META_PATH, Config.TRAIN_JSON, "val"
    )

    # Subset data for speed (use only 50 samples)
    SUBSET_SIZE = 50
    print(f"Subsetting datasets to {SUBSET_SIZE} samples for speed...")

    train_imgs = train_imgs[:SUBSET_SIZE]
    train_angs = train_angs[:SUBSET_SIZE]
    train_lbls = train_lbls[:SUBSET_SIZE]

    val_imgs = val_imgs[:SUBSET_SIZE]
    val_angs = val_angs[:SUBSET_SIZE]
    val_lbls = val_lbls[:SUBSET_SIZE]

    # Create Datasets
    # Use get_transforms to retrieve augmentation pipelines
    train_dataset = dataset.IcebergDataset(
        train_imgs, train_angs, train_lbls, transform=dataset.get_transforms("train")
    )
    val_dataset = dataset.IcebergDataset(
        val_imgs, val_angs, val_lbls, transform=dataset.get_transforms("val")
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Validate Data Loading Logic
    print("Verifying DataLoader shapes...")
    sample_imgs, sample_angs, sample_lbls = next(iter(train_loader))

    # Assertions to ensure logic is correct
    assert sample_imgs.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), f"Unexpected image shape: {sample_imgs.shape}"
    assert sample_angs.shape == (
        Config.BATCH_SIZE,
    ), f"Unexpected angle shape: {sample_angs.shape}"
    assert sample_lbls.shape == (
        Config.BATCH_SIZE,
    ), f"Unexpected label shape: {sample_lbls.shape}"
    print("Data shapes verified.")

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\n--- Initializing Model ---")

    # Initialize the custom ResNet model
    # Using pretrained=False for speed and to avoid internet dependency in this demo
    model = networks.IcebergResNet(pretrained=False)
    model = model.to(device)

    # Verify Model Forward Pass
    print("Verifying model forward pass...")
    with torch.no_grad():
        # Move sample batch to device
        dummy_logits = model(sample_imgs.to(device), sample_angs.to(device))

    assert dummy_logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Unexpected output shape: {dummy_logits.shape}"
    print("Model forward pass verified.")

    # ==========================================
    # 4. Training Loop (Phase 1)
    # ==========================================
    print("\n--- Starting Training Demo (2 Epochs) ---")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, 3):
        # Train one epoch
        train_loss, train_acc = engine.train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Evaluate
        # Disable TTA for speed in this demo
        val_loss, val_acc = engine.evaluate(
            model, val_loader, criterion, device, tta=False
        )

        print(
            f"Epoch {epoch} Summary: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}"
        )

    # ==========================================
    # 5. SWA Demonstration (Phase 2)
    # ==========================================
    print("\n--- Demonstrating SWA (Stochastic Weight Averaging) ---")

    swa_model = AveragedModel(model)

    # Update SWA parameters (normally done over multiple epochs)
    engine.update_swa(swa_model, model)

    # Update Batch Normalization statistics
    print("Updating SWA Batch Normalization statistics...")
    engine.update_bn_custom(train_loader, swa_model, device)

    # Evaluate SWA model
    print("Evaluating SWA model...")
    swa_loss, swa_acc = engine.evaluate(
        swa_model, val_loader, criterion, device, tta=False
    )
    print(f"SWA Results: Loss={swa_loss:.4f}, Acc={swa_acc:.4f}")

    # ==========================================
    # 6. Inference and Submission
    # ==========================================
    print("\n--- Running Inference on Test Data ---")

    # Load Test Data
    test_imgs, test_angs, _, test_ids = dataset.load_data(
        Config.TEST_META_PATH, Config.TEST_JSON, "test"
    )

    # Subset Test Data
    test_imgs = test_imgs[:SUBSET_SIZE]
    test_angs = test_angs[:SUBSET_SIZE]
    test_ids = test_ids[:SUBSET_SIZE]

    # Create Test Dataset (Note: labels are None, ids are provided)
    test_dataset = dataset.IcebergDataset(
        test_imgs, test_angs, ids=test_ids, transform=dataset.get_transforms("test")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Run Inference
    model.eval()
    predictions = []
    ids_list = []

    print("Predicting...")
    with torch.no_grad():
        for batch_imgs, batch_angs, batch_ids in test_loader:
            batch_imgs = batch_imgs.to(device)
            batch_angs = batch_angs.to(device)

            # Forward pass
            logits = model(batch_imgs, batch_angs)
            probs = torch.sigmoid(logits)

            predictions.extend(probs.cpu().numpy().flatten())
            ids_list.extend(batch_ids)

    # Verify predictions
    assert len(predictions) == SUBSET_SIZE, "Prediction count mismatch"
    assert len(ids_list) == SUBSET_SIZE, "ID count mismatch"

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"id": ids_list, "is_iceberg": predictions})

    # Save Submission
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    # Verify File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    print("\n--- Demo Completed Successfully ---")
    print("Sample Submission Rows:")
    print(submission_df.head())


if __name__ == "__main__":
    main()
