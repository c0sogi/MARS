import os
import torch
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import (
    seed_everything,
    INPUT_ROOT,
    METADATA_DIR,
    WORKING_DIR,
    DEVICE,
    NUM_CLASSES,
    IMAGE_SIZE,
)
from library.dataset import CameraTrapDataset, get_transforms
from library.model import get_convnext_model
from library.utils import get_class_weights
from library.loss import ClassBalancedFocalLoss
from library.engine import train_model, predict_and_submit


def run_demo():
    # 1. Setup
    seed_everything(42)
    print("Starting Library Usage Demonstration...")

    # Define paths
    train_meta_path = os.path.join(METADATA_DIR, "train_meta.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val_meta.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test_meta.csv")

    # 2. Data Preparation (Using Subsets for Speed)
    print("\n[1/5] Loading Metadata Subsets...")
    if not os.path.exists(train_meta_path):
        raise FileNotFoundError(f"Metadata not found at {train_meta_path}")

    df_train_full = pd.read_csv(train_meta_path)
    df_val_full = pd.read_csv(val_meta_path)
    df_test_full = pd.read_csv(test_meta_path)

    # Create small subsets: 50 train, 20 val, 20 test
    df_train_demo = df_train_full.head(50).copy()
    df_val_demo = df_val_full.head(20).copy()
    df_test_demo = df_test_full.head(20).copy()

    print(f"    Train subset size: {len(df_train_demo)}")
    print(f"    Val subset size:   {len(df_val_demo)}")
    print(f"    Test subset size:  {len(df_test_demo)}")

    # 3. Dataset and DataLoader
    print("\n[2/5] Initializing Datasets and Loaders...")
    train_transform = get_transforms(phase="train")
    val_transform = get_transforms(phase="val")  # Same for test usually

    # Instantiate Datasets
    train_dataset = CameraTrapDataset(
        df_train_demo, INPUT_ROOT, transform=train_transform, is_test=False
    )
    val_dataset = CameraTrapDataset(
        df_val_demo, INPUT_ROOT, transform=val_transform, is_test=False
    )
    test_dataset = CameraTrapDataset(
        df_test_demo, INPUT_ROOT, transform=val_transform, is_test=True
    )

    # Verify __getitem__ logic
    sample_img, sample_label = train_dataset[0]
    assert sample_img.shape == (
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ), f"Image shape mismatch. Expected (3, {IMAGE_SIZE}, {IMAGE_SIZE}), got {sample_img.shape}"
    assert isinstance(sample_label, torch.Tensor), "Label should be a torch.Tensor"
    print("    Dataset integrity check passed.")

    # Create DataLoaders
    # Using a small batch size for the demo
    demo_batch_size = 8
    train_loader = DataLoader(
        train_dataset, batch_size=demo_batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=demo_batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=demo_batch_size, shuffle=False, num_workers=2
    )

    # 4. Model Initialization
    print("\n[3/5] Initializing Model...")
    # Using pretrained=False to avoid download overhead during this demo script
    model = get_convnext_model(
        model_name="convnext_tiny", num_classes=NUM_CLASSES, pretrained=False
    )
    model = model.to(DEVICE)

    # Verify Model Output Shape
    dummy_input = torch.randn(demo_batch_size, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    assert dummy_output.shape == (
        demo_batch_size,
        NUM_CLASSES,
    ), f"Model output shape mismatch. Expected ({demo_batch_size}, {NUM_CLASSES}), got {dummy_output.shape}"
    print("    Model forward pass check passed.")

    # 5. Loss Function and Training Loop
    print("\n[4/5] Running Training Loop (1 Epoch)...")

    # Calculate class weights from the training subset
    class_weights = get_class_weights(df_train_demo)

    # Initialize Loss
    criterion = ClassBalancedFocalLoss(alpha=class_weights, gamma=2.0)

    # Initialize Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Run Training
    # We override num_epochs to 1 for speed
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=1,
        patience=1,
        device=DEVICE,
    )

    # Verify model artifact creation
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "best_model.pth was not saved."
    print("    Training completed and model saved.")

    # 6. Prediction and Submission
    print("\n[5/5] Generating Predictions...")
    submission_file = os.path.join(WORKING_DIR, "demo_submission.csv")

    predict_and_submit(
        model=trained_model,
        test_loader=test_loader,
        device=DEVICE,
        submission_path=submission_file,
    )

    # Verify Submission
    assert os.path.exists(submission_file), "Submission file was not created."
    df_sub = pd.read_csv(submission_file)

    assert list(df_sub.columns) == [
        "Id",
        "Predicted",
    ], f"Submission columns mismatch. Expected ['Id', 'Predicted'], got {list(df_sub.columns)}"
    assert len(df_sub) == len(
        df_test_demo
    ), f"Submission length mismatch. Expected {len(df_test_demo)}, got {len(df_sub)}"

    print(f"    Submission verified. Rows: {len(df_sub)}")
    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    run_demo()
