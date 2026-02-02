import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Import components from the provided library files
from library.config import Config
from library.utils import set_seed
from library.dataset import CactusDataset, get_transforms, mixup_data
from library.model import ShallowConvNeXt
from library.engine import train_one_epoch, validate
from library.inference import run_inference


def main():
    print("=== Starting Cactus Identification Library Demo ===\n")

    # 1. Setup and Configuration Override
    # We override Config settings to ensure the script runs quickly for demonstration purposes.
    print("Step 1: Configuring environment...")
    set_seed(42)

    Config.EPOCHS = 1
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 50  # Use only 50 images for training/val demo
    Config.BATCH_SIZE = 10  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo

    # Ensure working directory exists (handled by Config, but good to double check)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    device = Config.DEVICE
    print(f"Device: {device}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Batch Size: {Config.BATCH_SIZE}")

    # 2. Dataset and DataLoader Verification
    print("\nStep 2: verifying Dataset and DataLoader...")

    # Instantiate Training Dataset in Debug mode
    train_dataset = CactusDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        phase="train",
        transform=get_transforms("train"),
        debug=True,
    )

    # Instantiate Validation Dataset in Debug mode
    val_dataset = CactusDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        phase="val",
        transform=get_transforms("val"),
        debug=True,
    )

    print(f"Train Dataset Length (Debug): {len(train_dataset)}")
    print(f"Val Dataset Length (Debug): {len(val_dataset)}")

    # Check DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True
    )

    # Fetch one batch
    images, labels = next(iter(train_loader))

    # Verify shapes
    # Images: (Batch, Channels, Height, Width) -> (10, 3, 32, 32)
    # Labels: (Batch,) -> (10,)
    print(f"Batch Images Shape: {images.shape}")
    print(f"Batch Labels Shape: {labels.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect image batch shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label batch shape"

    # 3. Mixup Augmentation Verification
    print("\nStep 3: Verifying Mixup Augmentation...")
    mixed_imgs, y_a, y_b, lam = mixup_data(
        images, labels, alpha=Config.MIXUP_ALPHA, device="cpu"
    )

    print(f"Mixed Images Shape: {mixed_imgs.shape}")
    print(f"Lambda: {lam}")

    assert mixed_imgs.shape == images.shape, "Mixup altered image dimensions"
    assert y_a.shape == labels.shape, "Mixup altered label dimensions"
    assert 0 <= lam <= 1, "Lambda is out of range [0, 1]"

    # 4. Model Verification
    print("\nStep 4: Verifying Model Architecture...")
    model = ShallowConvNeXt(
        in_chans=3,
        num_classes=1,
        depths=Config.MODEL_DEPTHS,
        dims=Config.MODEL_DIMS,
        drop_path_rate=Config.DROP_PATH_RATE,
    )
    model.to(device)

    # Perform forward pass with the batch fetched earlier
    inputs = images.to(device)
    outputs = model(inputs)

    print(f"Model Output Shape: {outputs.shape}")
    # Expected output: (Batch, 1) because num_classes=1
    assert outputs.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"

    # 5. Training Engine Verification
    print("\nStep 5: Verifying Training and Validation Loop...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Run one epoch of training
    print("Running train_one_epoch...")
    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        epoch_idx=0,
        mixup_alpha=Config.MIXUP_ALPHA,
    )
    print(f"Training Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # Run validation
    print("Running validate...")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    val_loss, val_auc = validate(model, val_loader, criterion, device)

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert 0 <= val_auc <= 1, "AUC score out of range"

    # 6. Inference Pipeline Verification
    print("\nStep 6: Verifying Inference Pipeline...")

    # Save the model checkpoint (required for run_inference)
    print(f"Saving model checkpoint to {Config.MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved"

    # Run full inference
    # Note: run_inference uses the default CactusDataset params, so it will load the full test set.
    # The test set is small (~3300 images), so this will be fast even without debug mode.
    print("Executing run_inference...")
    run_inference(
        model_path=Config.MODEL_SAVE_PATH,
        metadata_path=Config.TEST_METADATA_PATH,
        submission_path=Config.SUBMISSION_PATH,
        batch_size=Config.BATCH_SIZE,  # Use small batch size defined above
        device=device,
    )

    # Check submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission File Loaded. Shape: {submission_df.shape}")
    print(submission_df.head())

    # Validate submission content
    # We expect 3325 rows based on sample_submission.csv info in the prompt
    expected_rows = 3325
    assert (
        len(submission_df) == expected_rows
    ), f"Expected {expected_rows} rows, found {len(submission_df)}"
    assert (
        "id" in submission_df.columns and "has_cactus" in submission_df.columns
    ), "Missing required columns"
    assert (
        submission_df["has_cactus"].min() >= 0
        and submission_df["has_cactus"].max() <= 1
    ), "Probabilities out of range"

    print("\n=== All Demonstrations and Verifications Passed Successfully ===")


if __name__ == "__main__":
    main()
