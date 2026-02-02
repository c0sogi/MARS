import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

# Import from provided library
from library.config import Config
from library.utils import seed_everything, probabilistic_f1
from library.data import get_dataloaders
from library.model import SHR_MTN
from library.train import train_one_epoch, validate_one_epoch, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Demonstration of Breast Cancer Detection Pipeline ===")

    # 1. Setup & Configuration Override
    # We override Config settings to ensure the demo runs quickly (Debug Mode)
    print("\n[Step 1] Configuring environment...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Small subset for speed
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_EPOCHS = 1  # Single epoch
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")
    print(f"Debug Mode: {Config.DEBUG}")

    # 2. Verify Metric Logic
    print("\n[Step 2] Verifying Probabilistic F1 Score...")
    # Test Case: Perfect prediction
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1.0, 0.0, 1.0])
    score = probabilistic_f1(y_true, y_pred)
    assert np.isclose(
        score, 1.0
    ), f"Expected pF1=1.0 for perfect predictions, got {score}"

    # Test Case: Partial confidence
    # y_true=[1], y_pred=[0.8]
    # pTP = 0.8, pFP = 0.0, TotalPos = 1
    # pPrec = 0.8/0.8 = 1.0, pRec = 0.8/1.0 = 0.8
    # pF1 = 2*(1*0.8)/(1.8) = 1.6/1.8 ≈ 0.8889
    score_partial = probabilistic_f1([1], [0.8])
    assert 0.88 < score_partial < 0.89, f"pF1 calculation mismatch: {score_partial}"
    print("Metric verification passed.")

    # 3. Data Loading
    print("\n[Step 3] Initializing DataLoaders...")
    # This triggers metadata processing and dataset creation
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=Config.DEBUG,
        load_cached_data=False,  # Force reprocessing for demo purposes
    )

    # Verify batch structure
    sample_batch = next(iter(train_loader))
    assert "image" in sample_batch, "Batch missing 'image' key"
    assert "aux_features" in sample_batch, "Batch missing 'aux_features' key"
    assert "targets" in sample_batch, "Batch missing 'targets' key"

    # Verify shapes
    # Image: (B, 3, 1024, 1024) -> EfficientNet expects 3 channels
    expected_img_shape = (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE[0],
        Config.IMAGE_SIZE[1],
    )
    assert (
        sample_batch["image"].shape == expected_img_shape
    ), f"Image shape mismatch. Expected {expected_img_shape}, got {sample_batch['image'].shape}"

    print(f"Batch loaded successfully. Image shape: {sample_batch['image'].shape}")

    # 4. Model Instantiation
    print("\n[Step 4] Instantiating SHR_MTN Model...")
    # Determine number of aux features from the dataset
    num_aux_features = len(train_loader.dataset.feature_cols)
    print(f"Number of auxiliary features: {num_aux_features}")

    model = SHR_MTN(num_aux_features=num_aux_features)
    model.to(device)

    # Verify Forward Pass
    images = sample_batch["image"].to(device)
    aux = sample_batch["aux_features"].to(device)

    with torch.no_grad():
        outputs = model(images, aux)

    assert "cancer" in outputs
    assert "birads" in outputs
    assert "density" in outputs
    assert outputs["cancer"].shape == (
        Config.BATCH_SIZE,
        1,
    ), "Cancer output shape mismatch"
    print("Model forward pass successful.")

    # 5. Training Loop Simulation
    print("\n[Step 5] Simulating Training Step...")

    # Setup Optimizer and Loss
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criteria = {
        "cancer": nn.BCEWithLogitsLoss(pos_weight=pos_weight),
        "birads": nn.MSELoss(),
        "density": nn.CrossEntropyLoss(),
    }

    # Run one training epoch (on subset)
    train_loss, train_cancer_loss, train_pf1 = train_one_epoch(
        model, train_loader, optimizer, criteria, scaler, device
    )
    print(f"Train Epoch Result -> Loss: {train_loss:.4f}, pF1: {train_pf1:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # Run one validation epoch
    val_loss, val_cancer_loss, val_pf1 = validate_one_epoch(
        model, val_loader, criteria, device
    )
    print(f"Val Epoch Result   -> Loss: {val_loss:.4f}, pF1: {val_pf1:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"

    # 6. Submission Generation
    print("\n[Step 6] Generating Submission...")
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    print(f"Submission generated at {Config.SUBMISSION_PATH}")
    print(f"Rows: {len(sub_df)}")
    print("Head:")
    print(sub_df.head())

    # Check columns
    assert "prediction_id" in sub_df.columns
    assert "cancer" in sub_df.columns

    # Check values
    assert (
        sub_df["cancer"].min() >= 0.0 and sub_df["cancer"].max() <= 1.0
    ), "Probabilities out of range [0, 1]"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
