import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import CFG
from library.utils import set_seed, get_score, get_logger
from library.dataset import BirdDataset
from library.models import BirdModel
from library.sam import SAM
from library.engine import train_fn, valid_fn


def main():
    print("Starting Library Usage Demonstration...")

    # 1. Setup and Configuration
    # Override CFG for speed and directory permissions
    CFG.seed = 42
    CFG.debug = True
    CFG.epochs = 1
    CFG.batch_size = 4
    CFG.output_dir = "./working/demo_execution"
    CFG.filtered_spectrogram_dir = "./input/supplemental_data/filtered_spectrograms"

    # Create working directory
    os.makedirs(CFG.output_dir, exist_ok=True)

    # Set seed
    set_seed(CFG.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Preparation
    print("\n--- Testing Data Loading ---")
    train_csv_path = "./metadata/train.csv"
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Metadata file not found: {train_csv_path}")

    df = pd.read_csv(train_csv_path)

    # Filter for existing files to avoid loading black images during demo check
    # The dataset class handles missing files, but for verification we want real data.
    valid_indices = []
    for idx, row in df.iterrows():
        fname = os.path.basename(row["file_path_spec"])
        full_path = os.path.join(CFG.filtered_spectrogram_dir, fname)
        if os.path.exists(full_path):
            valid_indices.append(idx)
        if len(valid_indices) >= 16:  # Just need a small batch
            break

    if not valid_indices:
        print(
            "Warning: No matching filtered spectrograms found. Using dummy data logic in Dataset."
        )
        # Fallback to first few rows; Dataset class will generate black images
        df_subset = df.head(16).copy()
    else:
        df_subset = df.loc[valid_indices].copy().reset_index(drop=True)
        print(f"Selected {len(df_subset)} valid samples for demonstration.")

    # 3. Dataset & DataLoader
    # Initialize Dataset
    train_dataset = BirdDataset(df_subset, mode="train", load_cached_data=False)
    val_dataset = BirdDataset(df_subset, mode="val", load_cached_data=False)

    # Verify Item Structure
    img, label = train_dataset[0]
    print(f"Image Shape: {img.shape}")
    print(f"Label Shape: {label.shape}")

    # Assertions
    assert img.shape == (
        3,
        CFG.img_height,
        CFG.img_width,
    ), f"Expected image shape (3, {CFG.img_height}, {CFG.img_width}), got {img.shape}"
    assert label.shape == (
        CFG.num_classes,
    ), f"Expected label shape ({CFG.num_classes},), got {label.shape}"
    assert isinstance(img, torch.Tensor), "Image should be a torch.Tensor"
    assert isinstance(label, torch.Tensor), "Label should be a torch.Tensor"

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=0,  # 0 for simple debugging
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=0
    )
    print("DataLoaders initialized successfully.")

    # 4. Model Initialization
    print("\n--- Testing Model ---")
    # Use resnet18 and pretrained=False for speed and to avoid download issues
    model = BirdModel(backbone_name="resnet18", pretrained=False)
    model.to(device)

    # Test Forward Pass with dummy input
    dummy_input = torch.randn(2, CFG.in_channels, CFG.img_height, CFG.img_width).to(
        device
    )
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (
        2,
        CFG.num_classes,
    ), f"Expected output shape (2, {CFG.num_classes}), got {output.shape}"

    # 5. Optimizer (SAM) & Loss
    print("\n--- Testing Optimizer (SAM) ---")
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, lr=1e-3, rho=0.05)
    criterion = nn.BCEWithLogitsLoss()

    # 6. Training Loop Demonstration
    print("\n--- Testing Training Loop (train_fn) ---")
    # Run for 1 epoch on the small subset
    avg_train_loss = train_fn(train_loader, model, criterion, optimizer, device)
    print(f"Average Train Loss: {avg_train_loss:.4f}")
    assert not np.isnan(avg_train_loss), "Training loss is NaN"

    # 7. Validation Loop Demonstration
    print("\n--- Testing Validation Loop (valid_fn) ---")
    avg_val_loss, val_score, preds = valid_fn(val_loader, model, criterion, device)
    print(f"Average Val Loss: {avg_val_loss:.4f}")
    print(f"Val Score (AUC): {val_score:.4f}")
    print(f"Predictions Shape: {preds.shape}")

    assert not np.isnan(avg_val_loss), "Validation loss is NaN"
    assert 0.0 <= val_score <= 1.0, "AUC score out of range [0, 1]"
    assert preds.shape == (len(df_subset), CFG.num_classes), "Prediction shape mismatch"

    # 8. Utility Verification
    print("\n--- Testing Utilities ---")
    # Test get_score with synthetic data
    # Case: Perfect prediction
    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    y_pred = np.array([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1], [0.8, 0.8, 0.2]])
    # Note: get_score handles cases where a class only has 1 label type by skipping it.
    # We ensure our synthetic data has variance in columns.
    score = get_score(y_true, y_pred)
    print(f"Utility get_score result: {score:.4f}")
    assert 0.0 <= score <= 1.0

    # Test Logger
    log_path = os.path.join(CFG.output_dir, "test.log")
    logger = get_logger(log_path)
    logger.info("Test log message.")
    assert os.path.exists(log_path), "Log file was not created."

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
