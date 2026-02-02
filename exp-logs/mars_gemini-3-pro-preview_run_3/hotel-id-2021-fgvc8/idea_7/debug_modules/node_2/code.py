import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Import from the provided library
from library.config import Config, seed_everything
from library.dataset import get_dataloaders
from library.model import HotelIdModel
from library.engine import train_fn, validate_fn, generate_submission
from library.utils import save_checkpoint


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("Initializing Configuration...")

    # Override Config for speed and demonstration purposes
    Config.debug = True  # Use a small subset of data
    Config.epochs = 1  # Run only 1 epoch
    Config.batch_size = 32  # Smaller batch size for the demo subset
    Config.num_workers = 2  # Reduce workers for simple demo

    # Ensure reproducibility
    seed_everything(Config.seed)

    device = Config.device
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\nLoading Data (Debug Mode)...")
    # get_dataloaders handles metadata loading, label encoding, and dataset creation
    # It also updates Config.n_classes based on the unique IDs found in the subset
    train_loader, val_loader, test_loader, unique_ids = get_dataloaders(
        load_cached_data=False
    )

    # Verification: Check DataLoaders
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Number of classes in subset: {Config.n_classes}")

    assert len(train_loader) > 0, "Train loader is empty."
    assert len(val_loader) > 0, "Val loader is empty."
    assert Config.n_classes > 0, "No classes found."

    # Verification: Check a single batch structure
    imgs, labels = next(iter(train_loader))
    assert imgs.dim() == 4, f"Expected 4D image tensor, got {imgs.shape}"
    assert labels.dim() == 1, f"Expected 1D label tensor, got {labels.shape}"
    assert imgs.shape[1] == 3, "Expected 3 channels (RGB)"
    assert imgs.shape[2] == Config.image_size, "Image size mismatch"

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("\nInitializing Model...")
    model = HotelIdModel()
    model.to(device)

    # Verification: Check Model Output Shapes
    # 1. Forward pass with labels (Training mode -> ArcFace Logits)
    dummy_input = torch.randn(2, 3, Config.image_size, Config.image_size).to(device)
    dummy_labels = torch.tensor([0, 1]).to(device)

    # Ensure dummy labels are within valid range for the subset
    if Config.n_classes < 2:
        dummy_labels = torch.zeros(2, dtype=torch.long).to(device)

    logits = model(dummy_input, dummy_labels)
    assert logits.shape == (
        2,
        Config.n_classes,
    ), f"Expected logits shape (2, {Config.n_classes}), got {logits.shape}"

    # 2. Forward pass without labels (Inference mode -> Scaled Cosine Similarity)
    inference_out = model(dummy_input)
    assert inference_out.shape == (
        2,
        Config.n_classes,
    ), f"Expected inference output shape (2, {Config.n_classes}), got {inference_out.shape}"

    # 3. Feature extraction (for TTA/Retrieval)
    features = model.extract_features(dummy_input)
    assert features.shape == (
        2,
        Config.embedding_size,
    ), f"Expected feature shape (2, {Config.embedding_size}), got {features.shape}"

    # -------------------------------------------------------------------------
    # 4. Training Loop (Single Epoch)
    # -------------------------------------------------------------------------
    print("\nStarting Training Demo...")

    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    # Run one epoch
    train_loss, train_acc = train_fn(
        train_loader, model, criterion, optimizer, device, epoch=1
    )

    print(f"Training completed. Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

    # Verification: Loss should be a valid float
    assert not np.isnan(train_loss), "Training loss is NaN"
    assert train_loss > 0, "Training loss should be positive"

    # Save checkpoint (mock best)
    save_checkpoint(
        {
            "epoch": 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        is_best=True,
        output_dir=Config.output_dir,
    )
    assert os.path.exists(
        os.path.join(Config.output_dir, "best_model.pth")
    ), "Checkpoint not saved"

    # -------------------------------------------------------------------------
    # 5. Validation
    # -------------------------------------------------------------------------
    print("\nRunning Validation...")
    val_map5 = validate_fn(val_loader, model, device, unique_ids)

    # Verification: Score range
    assert 0.0 <= val_map5 <= 1.0, f"MAP@5 score {val_map5} out of range [0, 1]"

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    print("\nGenerating Submission...")
    generate_submission(test_loader, model, device, unique_ids)

    submission_path = "./submission/submission.csv"

    # Verification: Submission file
    assert os.path.exists(submission_path), "Submission file was not created"

    sub_df = pd.read_csv(submission_path)
    print(f"Submission shape: {sub_df.shape}")

    # Check columns
    assert (
        "image" in sub_df.columns and "hotel_id" in sub_df.columns
    ), "Submission missing required columns"

    # Check format of predictions (space delimited)
    sample_pred = sub_df.iloc[0]["hotel_id"]
    assert isinstance(sample_pred, str), "Prediction is not a string"
    assert (
        len(sample_pred.split(" ")) == 5
    ), f"Expected 5 predictions per image, got {len(sample_pred.split(' '))}"

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    main()
