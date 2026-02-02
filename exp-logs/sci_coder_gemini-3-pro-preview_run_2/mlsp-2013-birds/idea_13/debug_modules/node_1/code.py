import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from provided library files
from library.config import Config
from library.utils import (
    seed_everything,
    get_pos_weights,
    mixup_data,
    calculate_roc_auc,
)
from library.data import (
    BirdDataset,
    get_transforms,
    get_fold_dataloaders,
    get_test_dataloader,
)
from library.models import BirdClassifier
from library.engine import train_one_epoch, validate


def run_demo():
    print("=== Starting Library Usage Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    print("[1] Configuring environment for fast demonstration...")

    # Override Config for speed
    Config.DEBUG = True  # Use a small subset of data (40 samples)
    Config.NUM_EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for demo

    # Ensure working directory exists
    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set device
    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Utils Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    seed_everything(Config.SEED)

    # Test get_pos_weights
    # Create dummy labels: 4 samples, 3 classes
    dummy_labels = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 0, 0]])
    weights = get_pos_weights(dummy_labels, device="cpu")

    # Expected weights: (Total - Pos) / (Pos + eps)
    # Class 0: 2 pos, 2 neg -> ~1.0
    # Class 1: 1 pos, 3 neg -> ~3.0
    # Class 2: 1 pos, 3 neg -> ~3.0
    assert weights.shape == (3,), "Weights shape mismatch"
    assert torch.is_tensor(weights), "Weights should be a tensor"
    print("    get_pos_weights(): Verified.")

    # Test mixup_data
    dummy_input = torch.randn(4, 3, 224, 224)
    dummy_target = torch.randn(4, 3)
    mixed_x, y_a, y_b, lam = mixup_data(
        dummy_input, dummy_target, alpha=0.4, device="cpu"
    )

    assert mixed_x.shape == dummy_input.shape, "Mixup output shape mismatch"
    assert y_a.shape == dummy_target.shape, "Mixup target shape mismatch"
    print("    mixup_data(): Verified.")

    # -------------------------------------------------------------------------
    # 3. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Data Loading...")

    # We will use 'resnet18' which expects (224, 448) resolution defined in Config
    model_name = "resnet18"
    fold_idx = 0

    # Get DataLoaders
    # Note: DEBUG=True inside get_fold_dataloaders will sample 40 rows from metadata
    train_loader, val_loader = get_fold_dataloaders(fold_idx, model_name)

    print(f"    Train Loader Length: {len(train_loader)}")
    print(f"    Val Loader Length: {len(val_loader)}")

    # Fetch one batch to verify shapes
    images, labels = next(iter(train_loader))

    expected_h, expected_w = Config.MODEL_SPECS[model_name]["resolution"]

    # Check Image Shape: (Batch, 3, H, W)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        expected_h,
        expected_w,
    ), f"Image shape mismatch. Got {images.shape}, expected {(Config.BATCH_SIZE, 3, expected_h, expected_w)}"

    # Check Label Shape: (Batch, Num_Species)
    assert labels.shape == (
        Config.BATCH_SIZE,
        Config.NUM_SPECIES,
    ), f"Label shape mismatch. Got {labels.shape}, expected {(Config.BATCH_SIZE, Config.NUM_SPECIES)}"

    print(f"    Batch Shapes Verified: Images {images.shape}, Labels {labels.shape}")

    # -------------------------------------------------------------------------
    # 4. Model Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    # Instantiate Model (pretrained=False for speed in demo)
    model = BirdClassifier(model_name, num_classes=Config.NUM_SPECIES, pretrained=False)
    model.to(device)

    # Run a forward pass with the batch fetched earlier
    images = images.to(device)
    logits = model(images)

    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_SPECIES,
    ), f"Model output shape mismatch. Got {logits.shape}"

    print("    BirdClassifier (ResNet18): Forward pass successful.")

    # -------------------------------------------------------------------------
    # 5. Engine Verification (Training & Validation Loop)
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Training and Validation Loops...")

    # Setup simple optimizer and loss
    # Using dataset labels to calculate weights for the specific batch/subset
    # In a real run, this uses the full training set labels
    subset_labels = train_loader.dataset.labels
    pos_weights = get_pos_weights(subset_labels, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Run Train Step
    print("    Running training epoch...")
    train_loss = train_one_epoch(model, optimizer, train_loader, criterion, device)

    assert not np.isnan(train_loss), "Training loss returned NaN"
    print(f"    Train Loss: {train_loss:.4f}")

    # Run Validation Step
    print("    Running validation...")
    val_loss, val_auc = validate(model, val_loader, criterion, device)

    assert not np.isnan(val_loss), "Validation loss returned NaN"
    # AUC might be 0.5 if model is random/untrained, which is fine for demo
    print(f"    Val Loss: {val_loss:.4f}, Val ROC AUC: {val_auc:.4f}")

    # -------------------------------------------------------------------------
    # 6. Inference Verification
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Test Inference...")

    test_loader = get_test_dataloader(model_name, batch_size=Config.BATCH_SIZE)

    # Check one batch
    test_images, test_placeholders, test_rec_ids = next(iter(test_loader))

    assert test_images.shape == (
        Config.BATCH_SIZE,
        3,
        expected_h,
        expected_w,
    ), "Test image shape mismatch"
    assert len(test_rec_ids) == Config.BATCH_SIZE, "Test rec_ids length mismatch"

    # Simulate prediction
    model.eval()
    with torch.no_grad():
        test_logits = model(test_images.to(device))
        test_probs = torch.sigmoid(test_logits)

    assert test_probs.shape == (
        Config.BATCH_SIZE,
        Config.NUM_SPECIES,
    ), "Prediction shape mismatch"
    print("    Test Inference: Successful.")

    # -------------------------------------------------------------------------
    # 7. Mock Submission Generation
    # -------------------------------------------------------------------------
    print("\n[7] Mock Submission Generation...")

    # Create a small submission dataframe from the batch predictions
    # Format: Id,Probability
    # Id = rec_id * 100 + species_id

    submission_rows = []
    probs_np = test_probs.cpu().numpy()
    rec_ids_np = test_rec_ids.numpy()

    for i in range(len(rec_ids_np)):
        r_id = rec_ids_np[i]
        p_vec = probs_np[i]
        for species_idx, prob in enumerate(p_vec):
            row_id = r_id * 100 + species_idx
            submission_rows.append({"Id": row_id, "Probability": prob})

    df_sub = pd.DataFrame(submission_rows)
    print(f"    Generated {len(df_sub)} submission rows from one batch.")
    print(f"    Sample:\n{df_sub.head(3)}")

    # Save to working dir
    sub_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    df_sub.to_csv(sub_path, index=False)
    print(f"    Saved mock submission to {sub_path}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
