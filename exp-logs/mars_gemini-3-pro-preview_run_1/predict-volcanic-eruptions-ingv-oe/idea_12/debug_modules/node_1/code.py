import os
import sys
import numpy as np
import pandas as pd
import torch
import glob
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, load_npy, load_pickle
from library.feature_engineering import TabularFeatureEngineer
from library.spectrogram_ops import DualResSpectrogramGenerator
from library.model_tabular import run_lgbm_cv
from library.model_vision import train_cnn_fold, inference_cnn
from library.meta_learner import run_stacking


def main():
    # ==========================================
    # 1. SETUP & CONFIGURATION OVERRIDE
    # ==========================================
    print(">>> [Setup] Configuring environment for fast demonstration...")

    # Override Config for speed and demo purposes
    Config.CACHE_DIR = "./working/demo_cache"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce compute intensity
    Config.CNN_EPOCHS = 1  # Train for only 1 epoch
    Config.CNN_PATIENCE = 1  # Minimal patience
    Config.NUM_FOLDS = 2  # Use 2 folds instead of 5
    Config.BATCH_SIZE = 4  # Small batch size for small dataset
    Config.LGBM_PARAMS["n_estimators"] = 10  # Few trees for LightGBM
    Config.DEBUG = True

    # Create directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    seed_everything(Config.SEED)

    # ==========================================
    # 2. DATA LOADING & SUBSETTING
    # ==========================================
    print(">>> [Data] Loading metadata and creating mini-subsets...")

    # Load full metadata
    train_full = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_full = pd.read_csv(Config.VAL_METADATA_PATH)
    test_full = pd.read_csv(Config.TEST_METADATA_PATH)

    # Create mini-subsets (e.g., 20 samples for train, 10 for val/test)
    # This ensures the pipeline runs quickly while testing all logic
    subset_size_train = 20
    subset_size_val = 10
    subset_size_test = 10

    mini_train = train_full.head(subset_size_train).copy()
    mini_val = val_full.head(subset_size_val).copy()
    mini_test = test_full.head(subset_size_test).copy()

    print(
        f"    Mini Train: {len(mini_train)}, Mini Val: {len(mini_val)}, Mini Test: {len(mini_test)}"
    )

    # ==========================================
    # 3. TABULAR BRANCH DEMONSTRATION
    # ==========================================
    print("\n>>> [Branch A] Starting Tabular Feature Engineering & LightGBM...")

    # 3.1 Feature Engineering
    engineer = TabularFeatureEngineer()

    # Verify feature extraction on a single file first
    sample_file = os.path.join(Config.INPUT_DIR, mini_train.iloc[0]["file_path"])
    sample_feats = engineer._process_segment(
        sample_file, mini_train.iloc[0]["segment_id"]
    )

    # Assertions for Feature Engineering
    assert isinstance(
        sample_feats, dict
    ), "Feature extraction should return a dictionary"
    assert "segment_id" in sample_feats, "Features must contain segment_id"
    # Check for some expected keys (e.g., sensor_1_mean)
    assert (
        "sensor_1_mean" in sample_feats
    ), "Expected basic statistical features missing"

    # 3.2 Run LightGBM Pipeline
    # This function handles caching, CV training, and inference
    lgbm_oof, lgbm_test_preds = run_lgbm_cv(
        mini_train,
        mini_val,
        mini_test,
        load_cached_data=False,  # Force generation for demo
    )

    # Assertions for LightGBM Output
    assert len(lgbm_oof) == (
        len(mini_train) + len(mini_val)
    ), f"OOF size mismatch. Expected {len(mini_train) + len(mini_val)}, got {len(lgbm_oof)}"
    assert len(lgbm_test_preds) == len(
        mini_test
    ), f"Test preds size mismatch. Expected {len(mini_test)}, got {len(lgbm_test_preds)}"
    assert (
        not lgbm_oof["time_to_eruption"].isnull().any()
    ), "OOF predictions contain NaNs"

    print("    [Branch A] Tabular pipeline completed successfully.")

    # ==========================================
    # 4. VISION BRANCH DEMONSTRATION
    # ==========================================
    print("\n>>> [Branch B] Starting Vision Spectrogram Generation & CNN...")

    # 4.1 Spectrogram Generation Logic Check
    # We calculate the global max on the mini-train set first to ensure consistency
    # (In the real pipeline, this is done automatically, but we check the component here)
    spec_gen = DualResSpectrogramGenerator(global_max=Config.GLOBAL_MAX_READING)

    # Process one file
    raw_waveform = spec_gen._load_sensor_data(sample_file)
    spectrogram = spec_gen.transform(raw_waveform)

    # Assertions for Spectrograms
    assert isinstance(
        spectrogram, torch.Tensor
    ), "Spectrogram output should be a Tensor"
    assert spectrogram.shape == (
        20,
        256,
        256,
    ), f"Spectrogram shape mismatch. Expected (20, 256, 256), got {spectrogram.shape}"
    assert not torch.isnan(spectrogram).any(), "Spectrogram contains NaNs"

    # 4.2 Train CNN (Fold 0)
    # This will trigger dataset generation for the mini subsets
    print("    Training CNN Fold 0 (1 Epoch)...")
    best_mae, cnn_oof_fold = train_cnn_fold(
        mini_train,
        mini_val,
        fold_idx=0,
        load_cached_data=False,  # Force generation
        epochs=Config.CNN_EPOCHS,
        patience=Config.CNN_PATIENCE,
    )

    # Assertions for CNN Training
    model_path = os.path.join(Config.CACHE_DIR, "cnn_fold_0.pth")
    assert os.path.exists(model_path), "CNN model file was not saved"
    assert isinstance(best_mae, float), "Best MAE should be a float"
    assert len(cnn_oof_fold) == len(mini_val), "CNN OOF (Validation) size mismatch"

    # 4.3 CNN Inference
    print("    Running CNN Inference...")
    cnn_test_preds = inference_cnn(
        mini_test, model_paths=[model_path], load_cached_data=False
    )

    # Assertions for CNN Inference
    assert len(cnn_test_preds) == len(mini_test), "CNN Test predictions size mismatch"
    assert "time_to_eruption" in cnn_test_preds.columns, "Missing prediction column"

    print("    [Branch B] Vision pipeline completed successfully.")

    # ==========================================
    # 5. META-LEARNER STACKING DEMONSTRATION
    # ==========================================
    print("\n>>> [Ensemble] Starting Meta-Learner Stacking...")

    # For the stacking demo, we need OOF predictions that cover the ground truth we possess.
    # In the full pipeline, we run 5 folds to get OOF for everyone.
    # Here, we only ran Fold 0 for CNN (covering mini_val) and full CV for LGBM.
    # To demonstrate stacking, we will subset the LGBM OOF to match the CNN OOF (mini_val only).

    # Filter LGBM OOF to only include segment_ids present in mini_val
    val_ids = mini_val["segment_id"].values
    lgbm_oof_val = lgbm_oof[lgbm_oof["segment_id"].isin(val_ids)].copy()

    # CNN OOF already corresponds to mini_val (since we trained on mini_train and eval on mini_val)
    # Ensure column naming alignment (train_cnn_fold returns 'time_to_eruption_pred')

    # Run Stacking
    # Note: We pass mini_val as the ground_truth_df
    submission_df = run_stacking(
        tabular_oof=lgbm_oof_val,
        tabular_test=lgbm_test_preds,
        vision_oof=cnn_oof_fold,
        vision_test=cnn_test_preds,
        ground_truth_df=mini_val,
    )

    # Assertions for Stacking
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"
    assert len(submission_df) == len(mini_test), "Submission DataFrame size mismatch"
    assert submission_df.columns.tolist() == [
        "segment_id",
        "time_to_eruption",
    ], "Submission columns are incorrect"
    assert not submission_df.isnull().any().any(), "Submission contains NaNs"

    # Check values are non-negative
    assert (
        submission_df["time_to_eruption"] >= 0
    ).all(), "Predictions contain negative time values"

    print("    [Ensemble] Stacking completed successfully.")

    print("\n>>> All demonstrations passed successfully!")
    print(f">>> Final Submission generated at: {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
