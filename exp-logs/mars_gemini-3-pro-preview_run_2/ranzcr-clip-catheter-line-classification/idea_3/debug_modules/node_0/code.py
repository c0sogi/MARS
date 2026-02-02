import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Import from the provided library files
from library.config import Config
from library.dataset import CatheterDataset, get_transforms
from library.model import CatheterModel
from library.train import train_one_epoch, valid_one_epoch
from library.predict import predict
from library.utils import seed_everything, get_score


def run_demo():
    print("=== Starting Catheter Detection Pipeline Demo ===\n")

    # --- 1. Configuration & Setup ---
    # Override Config for a fast demonstration
    print("[1] Configuring environment for demo...")
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 2
    Config.PRETRAINED = False  # Disable download for speed/offline demo

    # Use a working directory for demo outputs
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    Config.OUTPUT_DIR = DEMO_DIR
    Config.BEST_MODEL_PATH = os.path.join(DEMO_DIR, "best_model_demo.pth")
    Config.SUBMISSION_FILE = os.path.join(DEMO_DIR, "submission_demo.csv")

    seed_everything(Config.SEED)
    print("    Configuration complete.")

    # --- 2. Dataset & Transforms Verification ---
    print("\n[2] Verifying Dataset and Transforms...")

    # Load training metadata
    df_train_full = pd.read_csv(Config.TRAIN_METADATA)
    # Take a tiny subset for the demo
    df_train_demo = df_train_full.head(16).copy()

    # Initialize Dataset
    train_dataset = CatheterDataset(
        df_train_demo, transforms=get_transforms(data="train"), mode="train"
    )

    print(f"    Dataset initialized with {len(train_dataset)} samples.")

    # Fetch one sample
    sample_img, sample_label = train_dataset[0]

    # Verify Image Shape: (3, 640, 640)
    assert sample_img.shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Incorrect image shape: {sample_img.shape}"

    # Verify Label Shape: (11,)
    assert sample_label.shape == (
        Config.NUM_CLASSES,
    ), f"Incorrect label shape: {sample_label.shape}"

    print("    Dataset shapes verified: Image (3, 640, 640), Label (11,).")

    # --- 3. Model Initialization & Forward Pass ---
    print("\n[3] Verifying Model Architecture...")

    device = Config.DEVICE
    model = CatheterModel(pretrained=Config.PRETRAINED)
    model.to(device)

    # Create a dummy batch
    dummy_batch = torch.stack(
        [train_dataset[i][0] for i in range(Config.BATCH_SIZE)]
    ).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(dummy_batch)

    # Verify Output Shape: (Batch_Size, Num_Classes)
    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Incorrect model output shape: {logits.shape}"

    print(f"    Forward pass successful. Output shape: {logits.shape}")

    # --- 4. Training Loop Demonstration ---
    print("\n[4] Demonstrating Training Step...")

    # Setup components
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )

    optimizer = AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = OneCycleLR(
        optimizer, max_lr=1e-3, epochs=1, steps_per_epoch=len(train_loader)
    )

    # Run one epoch of training
    print("    Running train_one_epoch...")
    avg_loss = train_one_epoch(
        model, train_loader, optimizer, scheduler, criterion, device
    )

    assert not np.isnan(avg_loss), "Training loss returned NaN."
    print(f"    Training complete. Average Loss: {avg_loss:.4f}")

    # --- 5. Validation Loop Demonstration ---
    print("\n[5] Demonstrating Validation Step...")

    # Use the same dataset as validation for demo purposes
    val_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    print("    Running valid_one_epoch...")
    val_loss, val_auc = valid_one_epoch(model, val_loader, criterion, device)

    assert not np.isnan(val_loss), "Validation loss returned NaN."
    assert 0.0 <= val_auc <= 1.0, f"Validation AUC out of range: {val_auc}"

    print(f"    Validation complete. Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # Save the model for the prediction step
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    print(f"    Model saved to {Config.BEST_MODEL_PATH}")

    # --- 6. Metric Utility Verification ---
    print("\n[6] Verifying Metric Calculation (get_score)...")

    # Synthetic Ground Truth: 10 samples, 11 classes
    y_true = np.zeros((10, 11))
    y_true[0:5, 0] = 1  # Class 0 has mix of 0 and 1
    # Class 1 remains all zeros to test edge case handling

    # Synthetic Predictions
    y_pred = np.random.rand(10, 11)

    score = get_score(y_true, y_pred)
    print(f"    Calculated AUC Score: {score:.4f}")
    assert isinstance(score, float), "Score is not a float."

    # --- 7. Inference & Submission ---
    print("\n[7] Demonstrating Inference (Predict)...")

    # Create a small test metadata file for the demo
    df_test_full = pd.read_csv(Config.TEST_METADATA)
    df_test_demo = df_test_full.head(10).copy()

    test_meta_path = os.path.join(DEMO_DIR, "test_demo.csv")
    df_test_demo.to_csv(test_meta_path, index=False)

    print(f"    Created temporary test metadata at {test_meta_path}")

    # Run prediction
    submission_df = predict(
        batch_size=Config.BATCH_SIZE,
        device=device,
        metadata_path=test_meta_path,
        model_path=Config.BEST_MODEL_PATH,
        output_path=Config.SUBMISSION_FILE,
    )

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created."
    assert (
        len(submission_df) == 10
    ), f"Expected 10 predictions, got {len(submission_df)}"
    assert (
        submission_df.shape[1] == 12
    ), f"Expected 12 columns (ID + 11 targets), got {submission_df.shape[1]}"
    assert (
        "StudyInstanceUID" in submission_df.columns
    ), "Missing ID column in submission."

    # Check probability range
    probs = submission_df.iloc[:, 1:].values
    assert (probs >= 0).all() and (
        probs <= 1
    ).all(), "Predictions contain values outside [0, 1]."

    print(f"    Submission generated successfully at {Config.SUBMISSION_FILE}")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
