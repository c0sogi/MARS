import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_logger, compute_qwk
from library.data import load_data_from_metadata, get_dataloaders, preprocess_and_cache
from library.models import RegressionModel, OrdinalModel
from library.engine import train_one_epoch, validate_one_epoch, AWP
from library.stacking import LGBMStacker, FeatureEngineer

# Ensure logs are visible
logging.basicConfig(level=logging.INFO)
logger = get_logger("demo")


def run_demo():
    print("=== Starting Essay Scoring Pipeline Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Fast Demonstration
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for speed...")
    # Use a tiny model to ensure the demo finishes in seconds
    Config.MODEL_PATH = "prajjwal1/bert-tiny"
    Config.HIDDEN_SIZE = 128  # Matches bert-tiny
    Config.MAX_LENGTH = 64  # Short sequence length for demo
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.GRADIENT_ACCUMULATION_STEPS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead
    Config.EXP_NAME = "demo_execution"

    # Re-setup environment directories based on new EXP_NAME
    Config.setup_environment()
    seed_everything(Config.SEED)

    print(f"Model: {Config.MODEL_PATH}")
    print(f"Working Dir: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Pipeline Demonstration
    # -------------------------------------------------------------------------
    print("\n[2] Testing Data Pipeline...")

    # Load a small subset of training data for the neural network demo
    full_train_df = load_data_from_metadata("train")
    subset_df = full_train_df.head(20).copy()
    print(f"Loaded subset of {len(subset_df)} samples.")

    # Generate Dataloader
    # This triggers tokenization and caching
    dataloader = get_dataloaders(
        subset_df,
        split_name="train_subset_demo",
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        load_cached_data=False,  # Force processing
    )

    # Verify Batch Structure
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    print(f"Batch keys: {list(batch.keys())}")
    print(f"Input shape: {input_ids.shape}")
    print(f"Labels shape: {labels.shape}")

    assert input_ids.shape == (Config.TRAIN_BATCH_SIZE, Config.MAX_LENGTH)
    assert labels.shape == (Config.TRAIN_BATCH_SIZE,)
    print("Data Pipeline Verification: PASSED")

    # -------------------------------------------------------------------------
    # 3. Model & Engine Demonstration (Regression)
    # -------------------------------------------------------------------------
    print("\n[3] Testing Neural Network Training Loop (Regression)...")

    device = Config.DEVICE
    print(f"Device: {device}")

    # Instantiate Model
    model = RegressionModel(model_path=Config.MODEL_PATH, pretrained=True).to(device)

    # Setup Training Components
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # Test AWP (Adversarial Weight Perturbation)
    awp = AWP(model, optimizer, adv_lr=1e-4, adv_eps=1e-2)

    # Run One Training Epoch
    print("Running training step...")
    train_loss = train_one_epoch(
        model,
        dataloader,
        optimizer,
        scheduler=None,
        device=device,
        epoch=1,
        criterion=criterion,
        scaler=scaler,
        awp=awp,
    )
    print(f"Train Loss: {train_loss:.4f}")

    # Run Validation
    print("Running validation step...")
    val_loss, val_qwk, preds, targets = validate_one_epoch(
        model, dataloader, device, criterion
    )
    print(f"Val Loss: {val_loss:.4f}, Val QWK: {val_qwk:.4f}")

    assert not np.isnan(train_loss)
    assert len(preds) == len(subset_df)
    print("NN Training Loop Verification: PASSED")

    # -------------------------------------------------------------------------
    # 4. Stacking / Meta-Model Demonstration
    # -------------------------------------------------------------------------
    print("\n[4] Testing LightGBM Stacker...")

    stacker = LGBMStacker()

    # Feature Engineering Check
    # We use the full train df here because the stacker usually runs on full OOFs
    # Feature extraction on 12k rows is fast (~seconds)
    print("Extracting meta-features...")
    feat_df = stacker.feature_engineer.get_features("train", load_cached_data=False)
    print(f"Meta-features shape: {feat_df.shape}")
    assert "char_count" in feat_df.columns

    # Simulate OOF Predictions for the Stacker
    # Must match the length of the 'train' metadata
    n_samples = len(full_train_df)
    dummy_oof_preds = np.random.uniform(1.0, 6.0, size=n_samples)

    train_oof_dict = {"deberta_v3": dummy_oof_preds}

    # Target values from metadata
    train_targets = full_train_df["score"].values

    # Train Stacker
    print("Training Stacker...")
    # Reduce n_folds and estimators for speed
    stacker.params["n_estimators"] = 10
    stacker.params["verbose"] = -1

    stacker_score = stacker.train(train_oof_dict, train_targets, n_folds=2)
    print(f"Stacker CV Score: {stacker_score:.4f}")

    assert len(stacker.models) == 2
    print("Stacker Training Verification: PASSED")

    # -------------------------------------------------------------------------
    # 5. Inference & Submission Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Testing Inference Pipeline...")

    # Load test metadata to get correct size
    df_test = load_data_from_metadata("test")
    n_test = len(df_test)

    # Simulate Test Predictions from the base model
    dummy_test_preds = np.random.uniform(1.0, 6.0, size=n_test)
    test_pred_dict = {"deberta_v3": dummy_test_preds}

    # Run Stacker Inference
    print("Generating submission...")
    stacker.run_inference_and_submit(test_pred_dict)

    # Verify Submission File
    submission_path = Config.SUBMISSION_FILE
    assert os.path.exists(submission_path)

    sub_df = pd.read_csv(submission_path)
    print(f"Submission shape: {sub_df.shape}")
    print(sub_df.head(2))

    assert sub_df.shape == (n_test, 2)
    assert list(sub_df.columns) == ["essay_id", "score"]
    assert sub_df["score"].between(1, 6).all()

    print("Inference Pipeline Verification: PASSED")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
