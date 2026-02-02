import sys
import os
import torch
import numpy as np
import pandas as pd
import shutil
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import WhaleClassifier
from library.engine import train_one_epoch, validate
from library.stacking import (
    StackingMetaLearner,
    generate_submission,
    prepare_meta_features,
    load_ground_truth,
)


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Override
    # -------------------------------------------------------------------------
    print("\n[1] Setting up Configuration for Demo...")

    # Override Config to run in a fast, debug mode
    Config.DEBUG = True
    Config.DEBUG_SIZE = 50  # Use only 50 samples for speed
    Config.EPOCHS = 1  # Single epoch
    Config.BATCH_SIZE = 8
    Config.NUM_FOLDS = 2  # Minimal folds for logic check

    # Redirect working directories to a demo folder
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Ensure clean working directory for demo
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Processing
    # -------------------------------------------------------------------------
    print("\n[2] Testing Data Loading & Processing...")

    # Get DataLoaders for Fold 0
    # This triggers cache generation in Config.WORKING_DIR
    # We use the standard hop length defined in Config
    train_loader, val_loader, test_loader = get_dataloaders(
        fold=0, hop_length=Config.HOP_LENGTH_STANDARD, load_cached_data=True
    )

    # Verify Train Loader Batch
    try:
        batch_data, batch_labels = next(iter(train_loader))
        print(f"Train Batch Data Shape: {batch_data.shape}")  # Expected: (B, 1, F, T)
        print(f"Train Batch Labels Shape: {batch_labels.shape}")  # Expected: (B,)

        # Assertions to verify logic
        assert (
            batch_data.dim() == 4
        ), "Data should be 4D tensor (Batch, Channel, Freq, Time)"
        assert batch_data.size(1) == 1, "Should have 1 channel (Spectrogram)"
        assert batch_labels.dim() == 1, "Labels should be 1D tensor"
        print("Data Loading Verification Passed.")
    except StopIteration:
        raise RuntimeError("Error: Train loader is empty!")

    # -------------------------------------------------------------------------
    # 3. Model Instantiation
    # -------------------------------------------------------------------------
    print("\n[3] Testing Model Instantiation...")

    # Use ResNet34 config (Model B)
    model_config = Config.MODEL_B
    print(f"Initializing {model_config['arch']}...")

    # Instantiate model
    # We set pretrained=False for the demo to avoid downloading weights
    model = WhaleClassifier(
        model_name=model_config["arch"],
        pretrained=False,
        in_chans=model_config["in_channels"],
    )
    model.to(Config.DEVICE)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_input = batch_data.to(Config.DEVICE)
        output = model(dummy_input)
        print(f"Model Output Shape: {output.shape}")

        # Output should be (Batch_Size, 1) for binary classification logits
        assert output.shape == (batch_data.size(0), 1), "Output shape mismatch"
        print("Model Forward Pass Verification Passed.")

    # -------------------------------------------------------------------------
    # 4. Training & Validation Loop (Engine)
    # -------------------------------------------------------------------------
    print("\n[4] Testing Training & Validation Engine...")

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Train one epoch
    print("Training for 1 epoch...")
    train_loss = train_one_epoch(model, train_loader, optimizer, Config.DEVICE)
    print(f"Epoch Train Loss: {train_loss:.4f}")

    # Validate
    print("Validating...")
    val_loss, val_auc = validate(model, val_loader, Config.DEVICE)
    print(f"Validation Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # Check metrics
    assert isinstance(train_loss, float)
    assert isinstance(val_auc, float)
    print("Engine Verification Passed.")

    # -------------------------------------------------------------------------
    # 5. Stacking Meta-Learner
    # -------------------------------------------------------------------------
    print("\n[5] Testing Stacking Meta-Learner...")

    # Load Ground Truth
    # In DEBUG mode, this loads the first Config.DEBUG_SIZE labels
    y_true = load_ground_truth()
    print(f"Loaded {len(y_true)} ground truth labels.")

    # Simulate OOF predictions for 2 models (Model A and Model B)
    # In a real scenario, these are collected from Cross-Validation
    np.random.seed(Config.SEED)
    oof_preds_a = np.random.uniform(0, 1, size=len(y_true))
    oof_preds_b = np.random.uniform(0, 1, size=len(y_true))

    predictions_dict = {"model_a": oof_preds_a, "model_b": oof_preds_b}

    # Prepare feature matrix for meta-learner
    X_oof = prepare_meta_features(predictions_dict)
    print(f"Meta-Feature Matrix Shape: {X_oof.shape}")

    # Initialize and Fit Meta-Learner
    meta_learner = StackingMetaLearner()
    auc_score = meta_learner.fit(X_oof, y_true)
    print(f"Meta-Learner Fit AUC: {auc_score:.4f}")

    # Simulate Test Predictions
    # We need to match the size of the test set in debug mode
    test_df = pd.read_csv(Config.TEST_CSV)
    if Config.DEBUG:
        test_df = test_df.iloc[: Config.DEBUG_SIZE]
    n_test = len(test_df)

    test_preds_a = np.random.uniform(0, 1, size=n_test)
    test_preds_b = np.random.uniform(0, 1, size=n_test)

    test_dict = {"model_a": test_preds_a, "model_b": test_preds_b}

    X_test = prepare_meta_features(test_dict)

    # Predict Final Probabilities
    final_test_probs = meta_learner.predict(X_test)
    print(f"Final Test Probabilities Shape: {final_test_probs.shape}")

    assert len(final_test_probs) == n_test
    print("Stacking Verification Passed.")

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    print("\n[6] Testing Submission Generation...")

    generate_submission(final_test_probs)

    assert os.path.exists(Config.SUBMISSION_PATH)
    print("Submission Generation Verification Passed.")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
