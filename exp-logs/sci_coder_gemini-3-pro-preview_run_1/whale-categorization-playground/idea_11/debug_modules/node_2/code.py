import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_map5
from library.dataset import get_dataloaders
from library.model import WhaleDenseNet
from library.trainer import train_one_epoch, validate
from library.inference import predict_ensemble


def run_demo():
    print("=== Starting Whale Identification Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment...")
    seed_everything(42)

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10  # Use only 10 samples for demonstration
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in this script
    Config.MAX_EPOCHS = 1

    # Set temporary working directories
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Clean up previous runs if they exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Debug Mode: {Config.DEBUG}")

    # -------------------------------------------------------------------------
    # 2. Verify Metric Logic
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Metric (MAP@5)...")
    # Scenario:
    # Sample 1: Target is at rank 1 (index 0) -> AP = 1/1 = 1.0
    # Sample 2: Target is at rank 2 (index 1) -> AP = 1/2 = 0.5
    # Mean AP = (1.0 + 0.5) / 2 = 0.75
    preds = [[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]]
    targets = [10, 21]

    score = calculate_map5(preds, targets)
    expected_score = 0.75

    assert (
        abs(score - expected_score) < 1e-6
    ), f"MAP@5 mismatch: got {score}, expected {expected_score}"
    print("    MAP@5 calculation verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Data Loading
    # -------------------------------------------------------------------------
    print("\n[3] Initializing DataLoaders...")
    # get_dataloaders handles reading CSVs, creating Datasets, and DataLoaders
    # load_cached_data=False forces regeneration of the class mapping for this debug subset
    train_loader, val_loader, test_loader, class_to_idx, idx_to_class = get_dataloaders(
        load_cached_data=False, verbose=False
    )

    # Verify DataLoaders are not empty
    assert len(train_loader) > 0, "Train loader should not be empty."
    assert len(val_loader) > 0, "Val loader should not be empty."
    assert len(test_loader) > 0, "Test loader should not be empty."

    # Fetch a single batch to verify shapes
    images, labels = next(iter(train_loader))

    print(f"    Batch Image Shape: {images.shape}")
    print(f"    Batch Label Shape: {labels.shape}")

    assert images.shape == (Config.BATCH_SIZE, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    assert labels.shape == (Config.BATCH_SIZE,)

    # Verify labels are within valid range
    num_classes = len(class_to_idx)
    assert labels.max() < num_classes
    print(f"    Data loading verified. Num classes in subset: {num_classes}")

    # -------------------------------------------------------------------------
    # 4. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[4] Initializing Model...")
    device = torch.device(Config.DEVICE)

    # Instantiate model with the specific number of classes found in the debug subset
    model = WhaleDenseNet(num_classes=num_classes, pretrained=False)
    model.to(device)

    images = images.to(device)
    labels = labels.to(device)

    # Test Forward Pass (Training Mode - with Labels)
    # This invokes the ElasticArcFace margin logic
    logits_train = model(images, labels)
    assert logits_train.shape == (Config.BATCH_SIZE, num_classes)
    assert not torch.isnan(
        logits_train
    ).any(), "Model produced NaN logits in training mode."

    # Test Forward Pass (Inference Mode - no Labels)
    # This returns scaled cosine similarities
    logits_inf = model(images, labels=None)
    assert logits_inf.shape == (Config.BATCH_SIZE, num_classes)

    # Test Embedding Extraction
    embeddings = model.get_embedding(images)
    assert embeddings.shape == (Config.BATCH_SIZE, Config.EMBEDDING_DIM)

    print("    Model architecture and forward passes verified.")

    # -------------------------------------------------------------------------
    # 5. Training & Validation Loop
    # -------------------------------------------------------------------------
    print("\n[5] Running Training & Validation Steps...")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Run one epoch of training
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"    Train Loss: {train_loss:.4f}")
    assert train_loss > 0, "Training loss should be positive."

    # Run validation
    val_loss, val_map5 = validate(model, val_loader, criterion, device)
    print(f"    Val Loss: {val_loss:.4f}, Val MAP@5: {val_map5:.4f}")
    assert val_loss > 0, "Validation loss should be positive."
    assert 0.0 <= val_map5 <= 1.0, "MAP@5 should be between 0 and 1."

    # -------------------------------------------------------------------------
    # 6. Inference & Submission
    # -------------------------------------------------------------------------
    print("\n[6] Generating Submission...")

    # Save the current model state as a checkpoint
    checkpoint_dir = os.path.join(Config.WORKING_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "demo_checkpoint.pth.tar")

    checkpoint_data = {
        "epoch": 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_map5": val_map5,
        "class_to_idx": class_to_idx,  # Crucial for mapping predictions back to IDs
    }
    torch.save(checkpoint_data, checkpoint_path)
    print(f"    Checkpoint saved to {checkpoint_path}")

    # Run ensemble prediction (using our single checkpoint)
    # This function handles TTA, averaging, and CSV generation
    predict_ensemble([checkpoint_path], test_loader, device)

    # Verify Submission File
    submission_file = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_file), "Submission file was not created."

    df_sub = pd.read_csv(submission_file)
    print(f"    Submission file loaded. Rows: {len(df_sub)}")

    # Check dimensions (should match DEBUG_SAMPLE_SIZE)
    assert len(df_sub) == Config.DEBUG_SAMPLE_SIZE

    # Check columns
    assert "Image" in df_sub.columns
    assert "Id" in df_sub.columns

    # Check content format
    sample_id = df_sub.iloc[0]["Id"]
    assert isinstance(sample_id, str)
    # Should have 5 predictions separated by spaces
    assert len(sample_id.split()) == 5, f"Expected 5 predictions, got: {sample_id}"

    print("    Submission format verified.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
