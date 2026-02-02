import os
import sys
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import CassavaDataset, get_transforms
from library.model import CassavaSwinModel
from library.engine import train_one_epoch, validate, inference_fn


def main():
    print("=== Starting Cassava Disease Classification Demo ===")

    # 1. Configuration & Setup
    # Override Config settings for a fast demonstration run
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4  # Small batch size for demo
    Config.DEBUG = True
    Config.WORKING_DIR = "./working/demo_run"
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    Config.PRETRAINED = (
        False  # Disable downloading weights for speed/offline capability
    )
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo script

    # Ensure working directories exist
    Config.setup_directories()

    # Set random seeds
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Preparation (Subsetting)
    print("\n--- Preparing Data Subsets ---")
    # Load metadata
    try:
        df_train_full = pd.read_csv(Config.TRAIN_METADATA_PATH)
        df_val_full = pd.read_csv(Config.VAL_METADATA_PATH)
        df_test_full = pd.read_csv(Config.TEST_METADATA_PATH)
    except FileNotFoundError as e:
        print(f"Error loading metadata: {e}")
        return

    # Create tiny subsets (enough for a few batches)
    # Ensure divisible by batch size for cleanliness in demo
    subset_size = 16
    df_train = df_train_full.head(subset_size).reset_index(drop=True)
    df_val = df_val_full.head(subset_size).reset_index(drop=True)
    df_test = df_test_full.head(subset_size).reset_index(drop=True)

    print(f"Train subset size: {len(df_train)}")
    print(f"Val subset size:   {len(df_val)}")
    print(f"Test subset size:  {len(df_test)}")

    # 3. Dataset & DataLoader Verification
    print("\n--- Verifying Dataset & DataLoader ---")
    train_dataset = CassavaDataset(
        df_train, transforms=get_transforms("train"), output_label=True
    )
    val_dataset = CassavaDataset(
        df_val, transforms=get_transforms("valid"), output_label=True
    )
    test_dataset = CassavaDataset(
        df_test, transforms=get_transforms("test"), output_label=False
    )

    # Verify single item retrieval
    img, label = train_dataset[0]
    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image shape mismatch: {img.shape}"
    assert isinstance(label, torch.Tensor), "Label is not a tensor"
    print("Dataset item shape verification passed.")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Verify batch retrieval
    batch_imgs, batch_labels = next(iter(train_loader))
    assert batch_imgs.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Batch image shape mismatch"
    assert batch_labels.shape == (Config.BATCH_SIZE,), "Batch label shape mismatch"
    print("DataLoader batch shape verification passed.")

    # 4. Model Initialization & Verification
    print("\n--- Initializing Model ---")
    model = CassavaSwinModel(pretrained=Config.PRETRAINED)
    model.to(device)

    # Verify forward pass
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
        output = model(dummy_input)
        assert output.shape == (
            2,
            Config.NUM_CLASSES,
        ), f"Model output shape mismatch: {output.shape}"
    print("Model forward pass verification passed.")

    # 5. Training Loop Demonstration
    print("\n--- Running Training Loop (1 Epoch) ---")
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # Run training for one epoch
    train_loss = train_one_epoch(
        epoch=0,
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scaler=scaler,
    )

    # Check if loss is valid
    assert not np.isnan(train_loss), "Training loss resulted in NaN"
    print(f"Epoch 0 Train Loss: {train_loss:.4f}")

    # Run validation
    val_loss, val_acc = validate(
        epoch=0, model=model, val_loader=val_loader, criterion=criterion, device=device
    )

    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert 0.0 <= val_acc <= 1.0, "Validation accuracy is out of bounds"
    print(f"Epoch 0 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Simulate model saving
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved"
    print("Model checkpoint saved successfully.")

    # 6. Inference & Submission
    print("\n--- Running Inference & Generating Submission ---")

    # Reload model to verify state dict loading
    inference_model = CassavaSwinModel(pretrained=False)
    inference_model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    )
    inference_model.to(device)

    # Run inference
    predictions = inference_fn(inference_model, test_loader, device)

    assert len(predictions) == len(
        df_test
    ), "Number of predictions does not match test set size"

    # Create submission file
    df_test["label"] = predictions
    submission = df_test[["image_id", "label"]]
    submission.to_csv(Config.SUBMISSION_PATH, index=False)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
