import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import from the provided library
from library.utils import set_seed, get_logger
from library.data import process_dataset, SFWIVDataset, get_transforms
from library.model import SFWIVModel
from library.train import train_one_epoch, validate
import library.config as config

# Define paths for demo files
WORKING_DIR = config.WORKING_DIR
DEMO_TRAIN_META = os.path.join(WORKING_DIR, "demo_train_metadata.csv")
DEMO_VAL_META = os.path.join(WORKING_DIR, "demo_val_metadata.csv")
DEMO_TEST_META = os.path.join(WORKING_DIR, "demo_test_metadata.csv")
DEMO_SUBMISSION = os.path.join(WORKING_DIR, "demo_submission.csv")


def create_mini_metadata():
    """
    Creates small metadata files in the working directory by sampling
    the original metadata. This ensures the demo runs quickly.
    """
    print("Creating mini-datasets for demonstration...")

    # Load original metadata
    df_train_full = pd.read_csv(config.TRAIN_METADATA)
    df_val_full = pd.read_csv(config.VAL_METADATA)
    df_test_full = pd.read_csv(config.TEST_METADATA)

    # Sample a few rows (e.g., 4 for train, 2 for val, 2 for test)
    # Using head() ensures deterministic selection without relying on random state here
    df_train_mini = df_train_full.head(4)
    df_val_mini = df_val_full.head(2)
    df_test_mini = df_test_full.head(2)

    # Save to working directory
    df_train_mini.to_csv(DEMO_TRAIN_META, index=False)
    df_val_mini.to_csv(DEMO_VAL_META, index=False)
    df_test_mini.to_csv(DEMO_TEST_META, index=False)

    print(f"Mini-train size: {len(df_train_mini)}")
    print(f"Mini-val size: {len(df_val_mini)}")
    print(f"Mini-test size: {len(df_test_mini)}")


def run_demo():
    # 1. Setup
    set_seed(42)
    device = config.DEVICE
    print(f"Running on device: {device}")

    # 2. Prepare Data
    create_mini_metadata()

    # 3. Process Datasets
    # We set load_cached_data=False to force the data processing logic to run on our new mini files
    # We use unique dataset names ('demo_train', etc.) to avoid conflicts with existing caches
    print("\n--- Processing Training Data ---")
    train_imgs, train_lbls, train_ids = process_dataset(
        DEMO_TRAIN_META, "demo_train", load_cached_data=False
    )

    print("\n--- Processing Validation Data ---")
    val_imgs, val_lbls, val_ids = process_dataset(
        DEMO_VAL_META, "demo_val", load_cached_data=False
    )

    # Verify Data Shapes
    # Expected: (N, 224, 224, 9)
    assert train_imgs.ndim == 4
    assert train_imgs.shape[1:] == (224, 224, 9)
    assert train_lbls.shape[0] == train_imgs.shape[0]
    print(f"Train images shape verified: {train_imgs.shape}")

    # 4. Create DataLoaders
    # We use the transforms provided by the library
    train_dataset = SFWIVDataset(
        train_imgs, train_lbls, transform=get_transforms("train")
    )
    val_dataset = SFWIVDataset(val_imgs, val_lbls, transform=get_transforms("val"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,  # Small batch size for demo
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead for tiny data
    )

    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)

    # 5. Initialize Model
    print("\n--- Initializing Model ---")
    model = SFWIVModel(pretrained=True)
    model = model.to(device)

    # Verify Model Architecture (Input Channels)
    # The first layer (conv_stem) should have 9 input channels
    first_layer = model.backbone.conv_stem
    assert (
        first_layer.in_channels == 9
    ), f"Expected 9 input channels, got {first_layer.in_channels}"
    print("Model architecture verified: Input layer accepts 9 channels.")

    # 6. Training Loop Demonstration
    print("\n--- Starting Training Loop (1 Epoch) ---")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Run 1 epoch
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}")

    # Run validation
    val_loss, val_auc = validate(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")

    # 7. Inference on Test Set
    print("\n--- Running Inference on Test Set ---")
    test_imgs, _, test_ids = process_dataset(
        DEMO_TEST_META, "demo_test", load_cached_data=False
    )

    test_dataset = SFWIVDataset(
        test_imgs, labels=None, transform=get_transforms("test")
    )
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

    model.eval()
    predictions = []

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            predictions.extend(probs.cpu().numpy().flatten())

    # 8. Create Submission File
    submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": predictions})

    submission_df.to_csv(DEMO_SUBMISSION, index=False)
    print(f"\nSubmission file generated at: {DEMO_SUBMISSION}")
    print(submission_df)

    # Final assertion to ensure file exists
    assert os.path.exists(DEMO_SUBMISSION)
    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    run_demo()
