import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging
import shutil
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, setup_logger, calculate_auc
from library.dataset import load_metadata, PathologyDataset, get_transforms
from library.models import get_model
from library.engine import train_one_epoch, evaluate, predict_tta
from library.meta_learner import train_xgboost_meta_learner, inference_meta_learner


def run_demo():
    # --- 1. Setup and Configuration Overrides ---
    print("--- Step 1: Initialization & Configuration ---")

    # Override Config for the demo to ensure speed and isolation
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8  # Small batch size for demo
    Config.DEBUG = True

    # Re-run setup to create new directories
    Config.setup()

    # Setup Logger
    logger = setup_logger()
    logger.info("Starting Demo Execution")

    # Set Seeds
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")

    # --- 2. Data Loading & Verification ---
    print("\n--- Step 2: Data Loading & Verification ---")

    # Load metadata
    # We force load_cached_data=False to demonstrate loading from source CSVs provided in ./metadata
    df_train_full = load_metadata("train", load_cached_data=False)
    df_val_full = load_metadata("val", load_cached_data=False)
    df_test_full = load_metadata("test", load_cached_data=False)

    # Subsample for speed (Demo Mode)
    # Using 50 samples for train, 20 for val, 20 for test
    df_train = df_train_full.iloc[:50].reset_index(drop=True)
    df_val = df_val_full.iloc[:20].reset_index(drop=True)
    df_test = df_test_full.iloc[:20].reset_index(drop=True)

    logger.info(
        f"Subset sizes - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}"
    )

    # Initialize Datasets
    train_dataset = PathologyDataset(df_train, transforms=get_transforms("train"))
    val_dataset = PathologyDataset(df_val, transforms=get_transforms("val"))
    test_dataset = PathologyDataset(df_test, transforms=get_transforms("test"))

    # Verification: Check Dataset Output
    sample_img, sample_label = train_dataset[0]
    assert sample_img.shape == (
        3,
        Config.CROP_SIZE,
        Config.CROP_SIZE,
    ), f"Incorrect image shape: {sample_img.shape}"
    assert isinstance(sample_label, torch.Tensor), "Label should be a tensor"
    logger.info("Dataset verification passed.")

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # --- 3. Base Model Training Loop ---
    print("\n--- Step 3: Base Model Training & Inference ---")

    # Storage for OOF (Out-Of-Fold) predictions and Test predictions
    # Structure: {'model_name': [preds...]}
    oof_preds_dict = {}
    test_preds_dict = {}

    criterion = nn.BCEWithLogitsLoss()

    for model_name in Config.MODEL_ARCHS:
        logger.info(f"Processing Model Architecture: {model_name}")

        # Instantiate Model
        model = get_model(model_name, pretrained=True)
        model.to(device)

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Train (1 Epoch)
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        logger.info(f"[{model_name}] Train Loss: {avg_loss:.4f}")

        # Evaluate (Validation)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)
        logger.info(f"[{model_name}] Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

        # Generate OOF Preds (using evaluate logic but storing raw probabilities)
        # We re-run predict loop manually or use evaluate's side effect if modified.
        # Here we use predict_tta with steps=1 for validation to get simple probabilities
        val_probs = predict_tta(model, val_loader, device, tta_steps=1)
        oof_preds_dict[model_name] = val_probs

        # Generate Test Preds (TTA)
        # Using TTA_STEPS=2 for speed in demo
        test_probs = predict_tta(model, test_loader, device, tta_steps=2)
        test_preds_dict[model_name] = test_probs

        # Assertions
        assert len(val_probs) == len(
            df_val
        ), f"OOF preds length mismatch for {model_name}"
        assert len(test_probs) == len(
            df_test
        ), f"Test preds length mismatch for {model_name}"

        # Clean up to save memory
        del model, optimizer
        torch.cuda.empty_cache()

    # --- 4. Prepare Stacking Data ---
    print("\n--- Step 4: Prepare Meta-Learner Data ---")

    # Construct OOF DataFrame
    oof_df = df_val.copy()
    for model_name, preds in oof_preds_dict.items():
        oof_df[model_name] = preds

    # Construct Test Prediction DataFrame
    test_pred_df = df_test.copy()
    for model_name, preds in test_preds_dict.items():
        test_pred_df[model_name] = preds

    logger.info(f"OOF DataFrame shape: {oof_df.shape}")
    logger.info(f"Test Pred DataFrame shape: {test_pred_df.shape}")

    # --- 5. Meta-Learner Training ---
    print("\n--- Step 5: Meta-Learner Training ---")

    # Train Meta-Learner
    # We set load_cached_model=False to ensure we train a fresh one for the demo
    meta_model = train_xgboost_meta_learner(oof_df, load_cached_model=False)

    assert meta_model is not None, "Meta-learner training failed (returned None)"

    # --- 6. Final Inference & Submission ---
    print("\n--- Step 6: Final Inference ---")

    submission_df = inference_meta_learner(meta_model, test_pred_df)

    # --- 7. Final Verification ---
    print("\n--- Step 7: Final Verification ---")

    # Check submission file exists
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Check submission content
    assert len(submission_df) == len(
        df_test
    ), f"Submission length {len(submission_df)} does not match test set {len(df_test)}"

    assert (
        "id" in submission_df.columns and "label" in submission_df.columns
    ), "Submission missing required columns"

    # Check probability range
    assert (
        submission_df["label"].min() >= 0.0 and submission_df["label"].max() <= 1.0
    ), "Predictions out of probability range [0, 1]"

    logger.info("Demo Execution Completed Successfully.")
    print(f"Submission saved to: {Config.SUBMISSION_PATH}")
    print("Head of submission:")
    print(submission_df.head())


if __name__ == "__main__":
    run_demo()
