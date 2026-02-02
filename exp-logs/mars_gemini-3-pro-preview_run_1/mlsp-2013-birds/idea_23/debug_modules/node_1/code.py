import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim.swa_utils import AveragedModel

# Import from the provided library
from library.config import Config
from library.utils import set_seed, get_logger, calculate_roc_auc
from library.data import (
    get_train_dataloader,
    get_val_dataloader,
    get_test_dataloader,
    mixup_data,
)
from library.model import get_model
from library.sam import SAM
from library.trainer import train_one_epoch, validate, update_swa_model, finalize_swa


def main():
    # 1. Setup and Configuration Override for Demo Speed
    print("--- Setting up Demo Configuration ---")
    # Override Config parameters to ensure the script runs quickly (within minutes)
    Config.EPOCHS_TEACHER = 2
    Config.SWA_START_EPOCH_TEACHER = 1
    Config.EPOCHS_STUDENT = 1
    Config.BATCH_SIZE = 8  # Small batch size for demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead/issues in demo script
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.PSEUDO_LABEL_PATH = os.path.join(
        Config.WORKING_DIR, "demo_pseudo_labels.parquet"
    )

    # Ensure working directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Loading Verification
    print("\n--- Verifying Data Loading ---")
    train_loader = get_train_dataloader(use_pseudo_labels=False)
    val_loader = get_val_dataloader()

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Fetch a batch to verify shapes and content
    images, targets = next(iter(train_loader))

    # Verify Image Shape: (Batch, Channels, Height, Width)
    # Config: 3 channels (stacked grayscale), 256x640
    expected_shape = (Config.BATCH_SIZE, 3, Config.IMG_HEIGHT, Config.IMG_WIDTH)
    assert (
        images.shape == expected_shape
    ), f"Image shape mismatch. Expected {expected_shape}, got {images.shape}"

    # Verify Target Shape: (Batch, Num_Classes)
    expected_target_shape = (Config.BATCH_SIZE, Config.NUM_CLASSES)
    assert (
        targets.shape == expected_target_shape
    ), f"Target shape mismatch. Expected {expected_target_shape}, got {targets.shape}"

    print("Data shapes verified successfully.")

    # 3. Mixup Verification
    print("\n--- Verifying Mixup Augmentation ---")
    mixed_x, y_a, y_b, lam = mixup_data(
        images, targets, alpha=Config.MIXUP_ALPHA, device="cpu"
    )
    assert mixed_x.shape == images.shape, "Mixed images shape mismatch"
    assert y_a.shape == targets.shape, "Target A shape mismatch"
    assert y_b.shape == targets.shape, "Target B shape mismatch"
    assert 0 <= lam <= 1, "Lambda mixup coefficient out of range"
    print("Mixup function verified.")

    # 4. Model Instantiation
    print("\n--- Verifying Model Architecture ---")
    model = get_model(
        pretrained=False
    )  # False for speed, avoiding download if not cached
    model = model.to(device)

    # Check output layer dimension
    assert (
        model.fc.out_features == Config.NUM_CLASSES
    ), f"Model output features mismatch. Expected {Config.NUM_CLASSES}"
    print("ResNet34 model instantiated and moved to device.")

    # 5. Optimizer (SAM) Setup
    print("\n--- Verifying SAM Optimizer ---")
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=0.01, momentum=0.9, rho=0.05)

    # Verify parameter groups
    assert len(optimizer.param_groups) > 0, "Optimizer has no param groups"
    print("SAM Optimizer initialized.")

    # 6. Training Loop Simulation (Teacher Stage)
    print("\n--- Simulating Training Loop (Teacher Stage) ---")
    criterion = nn.BCEWithLogitsLoss()

    # Initialize SWA Model
    swa_model = AveragedModel(model)

    for epoch in range(Config.EPOCHS_TEACHER):
        print(f"Epoch {epoch+1}/{Config.EPOCHS_TEACHER}")

        # Train Step
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        assert not np.isnan(train_loss), "Training loss is NaN"
        print(f"  Train Loss: {train_loss:.4f}")

        # Validation Step
        val_loss, val_auc = validate(model, val_loader, criterion, device)
        assert not np.isnan(val_loss), "Validation loss is NaN"
        assert 0.0 <= val_auc <= 1.0, f"Validation AUC out of range: {val_auc}"
        print(f"  Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

        # SWA Update Logic
        if epoch >= Config.SWA_START_EPOCH_TEACHER:
            print("  Updating SWA model...")
            update_swa_model(swa_model, model)

    # Finalize SWA
    print("Finalizing SWA statistics (running forward passes on train data)...")
    finalize_swa(swa_model, train_loader, device)
    print("SWA Finalized.")

    # 7. Pseudo-Labeling Verification
    print("\n--- Verifying Pseudo-Label Integration ---")
    # Create dummy pseudo-labels for the test set
    df_test = pd.read_csv(Config.TEST_CSV)
    print(f"Generating dummy pseudo-labels for {len(df_test)} test samples...")

    # Generate random probabilities
    dummy_probs = np.random.rand(len(df_test), Config.NUM_CLASSES).astype(np.float32)
    # Create DataFrame
    # Columns must match what get_train_dataloader expects (0..18 as strings or ints)
    cols = [str(i) for i in range(Config.NUM_CLASSES)]
    df_pseudo = pd.DataFrame(dummy_probs, columns=cols)
    df_pseudo["rec_id"] = df_test["rec_id"].values

    # Save to parquet
    df_pseudo.to_parquet(Config.PSEUDO_LABEL_PATH)
    assert os.path.exists(Config.PSEUDO_LABEL_PATH), "Pseudo-label file creation failed"

    # Reload train loader with pseudo-labels
    # The loader should now contain Train (Fold 0) + Test (Fold 1) samples
    train_loader_pl = get_train_dataloader(use_pseudo_labels=True)

    # Calculate expected size
    df_train_orig = pd.read_csv(Config.TRAIN_CSV)
    expected_len = len(df_train_orig) + len(df_test)
    assert (
        len(train_loader_pl.dataset) == expected_len
    ), f"Dataset size mismatch with pseudo-labels. Expected {expected_len}, got {len(train_loader_pl.dataset)}"

    print(
        f"Pseudo-label loading verified. New dataset size: {len(train_loader_pl.dataset)}"
    )

    # 8. Inference and Submission Format Check
    print("\n--- Verifying Inference and Submission Format ---")
    test_loader = get_test_dataloader()
    model.eval()

    submission_rows = []

    with torch.no_grad():
        for images, rec_ids in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            rec_ids = rec_ids.numpy()

            for i in range(len(rec_ids)):
                rid = rec_ids[i]
                row_probs = probs[i]

                # Format: Id is rec_id * 100 + species_id
                for species_id, p in enumerate(row_probs):
                    submission_id = int(rid * 100 + species_id)
                    submission_rows.append({"Id": submission_id, "Probability": p})

    df_sub = pd.DataFrame(submission_rows)
    print(f"Generated {len(df_sub)} submission rows.")

    # Check first few rows
    print(df_sub.head())

    # Verify constraints
    assert (
        df_sub["Probability"].min() >= 0 and df_sub["Probability"].max() <= 1
    ), "Probabilities out of bounds"
    assert not df_sub.isnull().values.any(), "Submission contains NaNs"

    # Save dummy submission
    sub_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    df_sub.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")

    print("\n=== All Verification Steps Passed Successfully ===")


if __name__ == "__main__":
    main()
