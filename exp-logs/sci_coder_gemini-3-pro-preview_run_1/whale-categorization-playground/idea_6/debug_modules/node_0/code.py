import sys
import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.dataset import WhaleDataset, get_transforms, get_class_list
from library.model import WhaleDenseNet
from library.engine import fit, generate_submission


def run_demo():
    # ---------------------------------------------------------
    # 1. Setup Configuration for Speed & Demo
    # ---------------------------------------------------------
    print("Setting up configuration for demo...")
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 128  # Small subset for quick execution
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 2
    # Use a specific directory for demo outputs to keep things organized
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/submission"

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 2. Reproducibility
    # ---------------------------------------------------------
    seed_everything(Config.SEED)

    # ---------------------------------------------------------
    # 3. Data Preparation
    # ---------------------------------------------------------
    print("Preparing data...")

    # Generate/Load class list
    # This reads the full training metadata to establish the label encoder
    class_list = get_class_list(load_cached_data=False)

    # Verify class list matches Config
    # Config.NUM_CLASSES is hardcoded to 4029 based on the full dataset analysis
    assert (
        len(class_list) == Config.NUM_CLASSES
    ), f"Class list length {len(class_list)} does not match Config.NUM_CLASSES {Config.NUM_CLASSES}"

    # Initialize Datasets
    # Train Dataset
    train_dataset = WhaleDataset(
        csv_file=Config.TRAIN_CSV,
        mode="train",
        transform=get_transforms("train"),
        class_list=class_list,
    )

    # Validation Dataset
    val_dataset = WhaleDataset(
        csv_file=Config.VAL_CSV,
        mode="val",
        transform=get_transforms("val"),
        class_list=class_list,
    )

    # Test Dataset
    # We use Config.TEST_CSV because it contains the 'file_path' column required by WhaleDataset
    test_dataset = WhaleDataset(
        csv_file=Config.TEST_CSV, mode="test", transform=get_transforms("test")
    )

    # Verify Debug Subsetting
    # In DEBUG mode, the dataset should truncate itself to DEBUG_SUBSET_SIZE
    print(f"Train dataset size: {len(train_dataset)}")
    assert (
        len(train_dataset) == Config.DEBUG_SUBSET_SIZE
    ), "Debug subsetting failed for train dataset"

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,  # Drop last to ensure batch norm works reliably with small batches
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # ---------------------------------------------------------
    # 4. Model Initialization
    # ---------------------------------------------------------
    print("Initializing model...")
    device = Config.DEVICE
    model = WhaleDenseNet()
    model.to(device)

    # Verify Model Forward Pass
    # Create a dummy batch to verify shape and device placement
    dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(device)
    dummy_labels = torch.tensor([0, 1]).to(device)

    # Forward with labels (Training mode -> returns ArcFace logits)
    output_train = model(dummy_input, dummy_labels)
    assert output_train.shape == (
        2,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch (Train mode)"

    # Forward without labels (Inference mode -> returns Cosine similarity)
    output_infer = model(dummy_input)
    assert output_infer.shape == (
        2,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch (Inference mode)"
    print("Model verification passed.")

    # ---------------------------------------------------------
    # 5. Training Setup
    # ---------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # ---------------------------------------------------------
    # 6. Execution (Train & Eval)
    # ---------------------------------------------------------
    print("Starting training loop...")
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
        patience=1,
    )

    # ---------------------------------------------------------
    # 7. Submission Generation
    # ---------------------------------------------------------
    print("Generating submission...")
    generate_submission(test_loader, model, device)

    # Verify Submission File
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    print(f"Submission generated with {len(df_sub)} rows.")

    # Verify Columns
    assert (
        "Image" in df_sub.columns and "Id" in df_sub.columns
    ), "Submission columns missing."

    # Verify Content Format
    if len(df_sub) > 0:
        sample_pred = df_sub.iloc[0]["Id"]
        assert isinstance(sample_pred, str), "Prediction ID is not a string."
        assert len(sample_pred.split()) <= 5, "More than 5 predictions found."

    print("Demo execution completed successfully.")


if __name__ == "__main__":
    run_demo()
