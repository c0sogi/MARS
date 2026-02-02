import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import load_dataset_metadata, get_transforms, PathologyDataset
from library.models import get_model
from library.engine import train_one_epoch, valid_one_epoch, tta_inference_fn
from library.stacking import (
    train_meta_learner,
    predict_with_meta_learner,
    create_submission,
)


def run_demo():
    print("--- Starting Library Usage Demonstration ---")

    # 1. Setup and Configuration Override
    # We modify the Config class attributes directly to isolate this demo run
    # and ensure it executes quickly.
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_FILE = os.path.join(Config.WORKING_DIR, "submission.csv")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure demo directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    print(
        f"Configuration set. Device: {Config.DEVICE}, Working Dir: {Config.WORKING_DIR}"
    )

    # 2. Data Loading and Dataset Creation
    print("\n[Step 1] Loading Data and Creating Dataset...")

    # Load metadata (using 'train' for demonstration purposes)
    # We force load_cached_data=False to demonstrate raw loading logic,
    # though in practice caching is preferred.
    full_df = load_dataset_metadata(mode="train", load_cached_data=False)

    # Create a tiny subset for speed (32 samples)
    subset_df = full_df.head(32).copy()
    print(
        f"Created subset of {len(subset_df)} samples from {len(full_df)} total records."
    )

    # Initialize Dataset with training transforms
    train_dataset = PathologyDataset(
        df=subset_df, transforms=get_transforms(data="train")
    )

    # Initialize DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    # Verification: Check batch shapes
    images, labels = next(iter(train_loader))
    print(f"Batch shapes - Images: {images.shape}, Labels: {labels.shape}")

    # Assertions for data integrity
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.CROP_SIZE,
        Config.CROP_SIZE,
    ), f"Image batch shape mismatch. Expected {(Config.BATCH_SIZE, 3, Config.CROP_SIZE, Config.CROP_SIZE)}, got {images.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Label batch shape mismatch. Expected {(Config.BATCH_SIZE,)}, got {labels.shape}"

    # 3. Model Initialization
    print("\n[Step 2] Initializing Model...")
    # Using pretrained=False to avoid downloading weights during this short demo
    model_name = "convnext_tiny"
    model = get_model(model_name, pretrained=False)
    model.to(Config.DEVICE)

    # Verification: Forward pass
    dummy_input = torch.randn(2, 3, Config.CROP_SIZE, Config.CROP_SIZE).to(
        Config.DEVICE
    )
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model output shape: {output.shape}")
    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"

    # 4. Training and Validation Loop
    print("\n[Step 3] Running Training and Validation Loop...")
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Train for one epoch
    train_loss = train_one_epoch(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=Config.DEVICE,
        epoch=1,
    )
    print(f"Training finished. Avg Loss: {train_loss:.4f}")
    assert isinstance(train_loss, float), "Train loss should be a float."

    # Validate for one epoch (using the same loader for demo simplicity)
    val_auc, val_loss = valid_one_epoch(
        model=model, val_loader=train_loader, device=Config.DEVICE, epoch=1
    )
    print(f"Validation finished. AUC: {val_auc:.4f}, Loss: {val_loss:.4f}")
    assert 0 <= val_auc <= 1, "AUC must be between 0 and 1."

    # 5. Inference with TTA
    print("\n[Step 4] Running TTA Inference...")
    # Run inference on the subset
    preds = tta_inference_fn(model, train_loader, Config.DEVICE)

    print(f"Inference predictions shape: {preds.shape}")
    assert len(preds) == len(
        subset_df
    ), f"Prediction count mismatch. Expected {len(subset_df)}, got {len(preds)}"
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions must be probabilities [0, 1]."

    # 6. Stacking / Meta-Learner
    print("\n[Step 5] Demonstrating Stacking (Meta-Learner)...")

    # Simulate OOF predictions for 2 models
    n_samples = 100
    fake_targets = np.random.randint(0, 2, n_samples)

    # Create fake probabilities roughly correlated with targets for a realistic AUC check
    # Add some noise to targets to create probs
    noise1 = np.random.uniform(-0.2, 0.2, n_samples)
    noise2 = np.random.uniform(-0.2, 0.2, n_samples)

    model_a_preds = np.clip(fake_targets * 0.8 + 0.1 + noise1, 0, 1)
    model_b_preds = np.clip(fake_targets * 0.7 + 0.15 + noise2, 0, 1)

    oof_preds = {"model_a": model_a_preds, "model_b": model_b_preds}

    # Train meta-learner
    # We disable cache loading to force training a new meta-learner
    meta_auc = train_meta_learner(
        oof_preds_dict=oof_preds, targets=fake_targets, load_cached_data=False
    )
    print(f"Meta-learner trained. OOF AUC: {meta_auc:.4f}")

    # Predict with meta-learner
    test_preds_dict = {"model_a": np.random.rand(10), "model_b": np.random.rand(10)}
    ensemble_preds = predict_with_meta_learner(test_preds_dict)

    print(f"Ensemble predictions shape: {ensemble_preds.shape}")
    assert len(ensemble_preds) == 10, "Ensemble prediction count mismatch."

    # 7. Submission Creation
    print("\n[Step 6] Creating Submission File...")
    test_ids = [f"id_{i}" for i in range(10)]
    create_submission(test_ids, ensemble_preds)

    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created."

    # Verify file content
    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission file loaded. Shape: {sub_df.shape}")
    assert list(sub_df.columns) == ["id", "label"], "Submission columns mismatch."
    assert len(sub_df) == 10, "Submission row count mismatch."

    print("\n--- Demonstration Complete Successfully ---")


if __name__ == "__main__":
    run_demo()
