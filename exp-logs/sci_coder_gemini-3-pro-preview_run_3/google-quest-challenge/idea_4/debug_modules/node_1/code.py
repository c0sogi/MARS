import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, compute_metric
from library.dataset import load_and_preprocess_data, get_dataloaders
from library.features import MetaFeatureEngineer
from library.modeling import (
    SegmentAwareCrossEncoder,
    train_backbone,
    extract_features,
    train_ridge_ensemble,
    predict_and_submit,
)


def run_demo():
    print("Starting Library Usage Demonstration...")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # We override Config attributes to create a fast "demo" mode.
    # Using a tiny model ensures the script runs in seconds/minutes instead of hours.
    print("\n[1] Configuring Demo Environment...")

    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths to use the demo directory
    Config.WORKING_DIR = DEMO_DIR
    Config.M1_TRAIN_FEATS_PATH = os.path.join(DEMO_DIR, "m1_train_features.npy")
    Config.M1_VAL_FEATS_PATH = os.path.join(DEMO_DIR, "m1_val_features.npy")
    Config.M1_TEST_FEATS_PATH = os.path.join(DEMO_DIR, "m1_test_features.npy")
    Config.M1_MODEL_PATH = os.path.join(DEMO_DIR, "m1_finetuned.pth")
    Config.META_TRAIN_FEATS_PATH = os.path.join(DEMO_DIR, "meta_train_features.npy")
    Config.META_VAL_FEATS_PATH = os.path.join(DEMO_DIR, "meta_val_features.npy")
    Config.META_TEST_FEATS_PATH = os.path.join(DEMO_DIR, "meta_test_features.npy")
    Config.RIDGE_MODEL_PATH = os.path.join(DEMO_DIR, "ridge_model.joblib")
    Config.SUBMISSION_FILE_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Override Model Hyperparameters for Speed
    Config.MODEL_1_NAME = (
        "prajjwal1/bert-tiny"  # Tiny model for functional verification
    )
    Config.NUM_EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 16
    Config.EVAL_BATCH_SIZE = 32
    Config.MAX_LENGTH = 128  # Shorter sequence length for speed
    Config.RIDGE_ALPHAS = (1.0,)  # Single alpha to speed up RidgeCV

    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")
    print(f"    Model: {Config.MODEL_1_NAME}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # --------------------------------------------------------------------------
    # 2. Data Loading & Preprocessing
    # --------------------------------------------------------------------------
    print("\n[2] Testing Data Loading & Preprocessing...")

    # Force re-processing to verify logic
    train_df, val_df, test_df = load_and_preprocess_data(load_cached_data=False)

    # Verification
    assert os.path.exists(
        os.path.join(DEMO_DIR, "train_processed.parquet")
    ), "Train parquet not created"
    assert "text_q" in train_df.columns, "text_q column missing"
    assert "text_a" in train_df.columns, "text_a column missing"
    # Metadata generated in previous steps had ~4394 train rows
    print(f"    Train shape: {train_df.shape}")
    print(f"    Val shape: {val_df.shape}")
    print(f"    Test shape: {test_df.shape}")

    # Get DataLoaders
    print("    Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        Config.MODEL_1_NAME, load_cached_data=True
    )

    # Inspect one batch
    batch = next(iter(train_loader))
    assert "input_ids" in batch
    assert "targets" in batch
    assert batch["input_ids"].shape[0] == Config.TRAIN_BATCH_SIZE
    assert batch["targets"].shape[1] == 30
    print("    DataLoader batch verification passed.")

    # --------------------------------------------------------------------------
    # 3. Meta-Feature Engineering
    # --------------------------------------------------------------------------
    print("\n[3] Testing Meta-Feature Engineering...")

    mfe = MetaFeatureEngineer()
    # Process splits (calculates features, fits scaler on train, transforms others)
    X_meta_train, X_meta_val, X_meta_test = mfe.process_splits(load_cached_data=False)

    # Verification
    assert X_meta_train.shape[0] == len(train_df)
    assert X_meta_val.shape[0] == len(val_df)
    assert X_meta_test.shape[0] == len(test_df)
    assert not np.isnan(X_meta_train).any(), "NaNs found in meta features"

    n_meta_feats = X_meta_train.shape[1]
    print(f"    Generated {n_meta_feats} meta-features.")

    # --------------------------------------------------------------------------
    # 4. Model Initialization & Training (Backbone)
    # --------------------------------------------------------------------------
    print("\n[4] Testing Backbone Model Training...")

    model = SegmentAwareCrossEncoder(Config.MODEL_1_NAME, num_labels=30)
    model.to(device)

    # Verify Forward Pass logic manually
    print("    Verifying forward pass...")
    dummy_input = batch["input_ids"].to(device)
    dummy_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(dummy_input, dummy_mask)
        feats = model.get_segment_features(dummy_input, dummy_mask)

    assert logits.shape == (
        Config.TRAIN_BATCH_SIZE,
        30,
    ), f"Logits shape mismatch: {logits.shape}"
    # bert-tiny hidden size is 128. get_segment_features returns 4 * hidden (CLS, Q, A, Diff)
    expected_feat_dim = 128 * 4
    assert feats.shape == (
        Config.TRAIN_BATCH_SIZE,
        expected_feat_dim,
    ), f"Features shape mismatch: {feats.shape}"
    print("    Forward pass & Feature extraction shapes correct.")

    # Train for 1 epoch
    print("    Running training loop (1 epoch)...")
    trained_model = train_backbone(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.NUM_EPOCHS,
        lr=Config.LEARNING_RATE,
        device=device,
        save_path=Config.M1_MODEL_PATH,
    )

    assert os.path.exists(Config.M1_MODEL_PATH), "Model checkpoint not saved."

    # --------------------------------------------------------------------------
    # 5. Feature Extraction (Backbone)
    # --------------------------------------------------------------------------
    print("\n[5] Testing Feature Extraction...")

    # Extract features using the trained model
    # We use the library function which handles caching

    # Create a dedicated inference loader for the training set (no shuffle, no drop_last)
    # Cite debug_lesson_3: Centralize Data Access to Ensure Consistency
    train_loader_inference = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=Config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    X_bb_train = extract_features(
        trained_model,
        train_loader_inference,
        device,
        Config.M1_TRAIN_FEATS_PATH,
        load_cached_data=False,
    )
    X_bb_val = extract_features(
        trained_model,
        val_loader,
        device,
        Config.M1_VAL_FEATS_PATH,
        load_cached_data=False,
    )
    X_bb_test = extract_features(
        trained_model,
        test_loader,
        device,
        Config.M1_TEST_FEATS_PATH,
        load_cached_data=False,
    )

    assert X_bb_train.shape[1] == expected_feat_dim
    print(f"    Extracted backbone features with dimension {X_bb_train.shape[1]}.")

    # --------------------------------------------------------------------------
    # 6. Ensemble Training (Ridge Regression)
    # --------------------------------------------------------------------------
    print("\n[6] Testing Ridge Ensemble...")

    # Concatenate Meta features and Backbone features
    X_train_full = np.hstack([X_meta_train, X_bb_train])
    X_val_full = np.hstack([X_meta_val, X_bb_val])
    X_test_full = np.hstack([X_meta_test, X_bb_test])

    # Get targets
    y_train = train_df[Config.TARGET_COLS].values
    y_val = val_df[Config.TARGET_COLS].values

    # Train Ridge
    ridge_model = train_ridge_ensemble(
        X_train_full, y_train, X_val_full, y_val, Config.RIDGE_MODEL_PATH
    )

    assert os.path.exists(Config.RIDGE_MODEL_PATH), "Ridge model not saved."

    # --------------------------------------------------------------------------
    # 7. Prediction & Submission
    # --------------------------------------------------------------------------
    print("\n[7] Testing Prediction & Submission Generation...")

    predict_and_submit(ridge_model, X_test_full, test_df, Config.SUBMISSION_FILE_PATH)

    assert os.path.exists(Config.SUBMISSION_FILE_PATH), "Submission file not found."

    # Verify submission format
    sub_df = pd.read_csv(Config.SUBMISSION_FILE_PATH)
    assert sub_df.shape == (
        len(test_df),
        31,
    ), f"Submission shape mismatch: {sub_df.shape}"
    assert sub_df.columns[0] == "qa_id"
    assert list(sub_df.columns[1:]) == Config.TARGET_COLS

    # Verify values are in [0, 1]
    numeric_vals = sub_df.iloc[:, 1:].values
    assert (numeric_vals >= 0).all() and (
        numeric_vals <= 1
    ).all(), "Predictions out of range [0, 1]"

    print("\n[SUCCESS] All demonstration steps completed successfully.")
    print(f"Submission generated at: {Config.SUBMISSION_FILE_PATH}")


if __name__ == "__main__":
    run_demo()
