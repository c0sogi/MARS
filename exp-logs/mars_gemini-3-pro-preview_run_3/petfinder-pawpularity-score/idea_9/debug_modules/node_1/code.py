import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.data_handling import PetDataset, get_pet_dataloader
from library.feature_engine import extract_and_cache_features
from library.data_processor import process_and_cache_data
from library.ensemble_model import StackingEnsemble

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def run_demo():
    print("=== Starting Library Usage Demonstration ===")

    # ==========================================
    # 1. Configuration Overrides for Speed
    # ==========================================
    print("\n[1] Configuring environment for rapid demonstration...")

    # Enable DEBUG mode to limit feature extraction to just a few batches (defined in feature_engine.py)
    Config.DEBUG = True

    # Reduce ensemble complexity for speed
    Config.N_FOLDS = 2  # Minimum folds for CV
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.ET_PARAMS["n_estimators"] = 10
    Config.SVR_PARAMS["max_iter"] = (
        100  # Limit SVR iterations if applicable (mostly for speed)
    )

    # Ensure working directory is clean for this run to prove file generation
    if os.path.exists(Config.WORKING_DIR):
        # We don't delete the whole dir to avoid permission issues, but we can clean specific files if needed.
        # For this demo, we rely on the overwrite logic in the libraries.
        pass

    seed_everything(Config.SEED)
    print("Configuration updated: DEBUG=True, N_FOLDS=2, Reduced Estimators.")

    # ==========================================
    # 2. Data Handling Demonstration
    # ==========================================
    print("\n[2] Demonstrating Data Handling (PetDataset & DataLoader)...")

    # Instantiate Dataset
    train_dataset = PetDataset(Config.TRAIN_META_PATH, mode="train")
    print(f"Train Dataset Length: {len(train_dataset)}")

    # Validate a single sample
    sample = train_dataset[0]
    required_keys = {"global_view", "zoomed_view", "metadata", "target", "id"}
    assert required_keys.issubset(
        sample.keys()
    ), f"Missing keys in dataset sample. Found: {sample.keys()}"

    # Validate Shapes
    # Image: (3, 224, 224)
    assert sample["global_view"].shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect global view shape"
    assert sample["zoomed_view"].shape == (
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect zoomed view shape"
    # Metadata: 12 features
    assert sample["metadata"].shape[0] == 12, "Incorrect metadata feature count"

    print("Dataset sample verification passed.")

    # Validate DataLoader
    train_loader = get_pet_dataloader(
        Config.TRAIN_META_PATH, mode="train", batch_size=4
    )
    batch = next(iter(train_loader))
    assert batch["global_view"].shape[0] == 4, "Batch size mismatch"
    print("DataLoader verification passed.")

    # ==========================================
    # 3. Feature Extraction Demonstration
    # ==========================================
    print("\n[3] Demonstrating Feature Extraction (BackboneExtractor)...")
    print("Note: This step loads heavy models (Swin, EffNet, DINO, CLIP). Please wait.")

    # We manually trigger extraction for 'train' to verify the dictionary structure.
    # Because DEBUG=True, this will only process a few batches.
    # We set load_cached_data=False to force execution of the extraction logic.
    raw_train_features = extract_and_cache_features("train", load_cached_data=False)

    # Verify Raw Output
    expected_streams = [
        "swin_global",
        "swin_zoomed",
        "effnet_global",
        "effnet_zoomed",
        "dino_global",
        "dino_zoomed",
        "clip_global",
        "clip_zoomed",
    ]
    for stream in expected_streams:
        assert stream in raw_train_features, f"Missing stream: {stream}"
        assert len(raw_train_features[stream]) > 0, f"Stream {stream} is empty"

    print(
        f"Feature extraction successful. Extracted {len(raw_train_features['ids'])} samples (DEBUG mode)."
    )

    # ==========================================
    # 4. Data Processing (PCA) Demonstration
    # ==========================================
    print("\n[4] Demonstrating Data Processing (MultiStreamPCA & Metadata Merging)...")

    # This function orchestrates extraction (if not cached) and PCA fitting/transforming.
    # It will reuse the 'train' features we just computed if we allow loading from cache,
    # but will compute 'val' and 'test' from scratch (limited by DEBUG mode).
    X_train, y_train, X_val, y_val, X_test, test_ids = process_and_cache_data(
        load_cached_data=True
    )

    # Verify Output Shapes
    print(f"Processed Train Shape: {X_train.shape}")
    print(f"Processed Val Shape:   {X_val.shape}")
    print(f"Processed Test Shape:  {X_test.shape}")

    # Assertions
    assert X_train.shape[0] == y_train.shape[0], "Train features/targets row mismatch"
    assert X_val.shape[0] == y_val.shape[0], "Val features/targets row mismatch"
    assert X_test.shape[0] == len(test_ids), "Test features/ids row mismatch"

    # Check if PCA reduced dimensions (Original is huge, PCA should be smaller)
    # 8 streams * large dims + 12 meta >> X_train.shape[1]
    # We expect X_train to be reasonably compact (e.g., < 4000 cols depending on variance)
    assert X_train.shape[1] > 12, "Features should include more than just metadata"

    print("Data processing verification passed.")

    # ==========================================
    # 5. Ensemble Modeling Demonstration
    # ==========================================
    print("\n[5] Demonstrating Stacking Ensemble (CV, Fit, Predict)...")

    ensemble = StackingEnsemble()

    # 5.1 Cross-Validation (Level 1 OOF + Meta Learner Training)
    print("Running Cross-Validation...")
    oof_rmse = ensemble.train_cv(X_train, y_train)
    print(f"CV OOF RMSE: {oof_rmse:.4f}")
    assert oof_rmse > 0, "RMSE should be positive"

    # 5.2 Final Fitting (Retraining on full train set)
    print("Retraining on full training set...")
    ensemble.fit_final(X_train, y_train)

    # 5.3 Prediction on Validation Set
    print("Predicting on Validation set...")
    val_preds = ensemble.predict(X_val)
    assert len(val_preds) == len(y_val), "Prediction length mismatch"
    assert np.all(
        (val_preds >= 1.0) & (val_preds <= 100.0)
    ), "Predictions out of valid range [1, 100]"

    val_rmse = np.sqrt(np.mean((y_val - val_preds) ** 2))
    print(f"Validation RMSE (using retrained model): {val_rmse:.4f}")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    print("\n[6] Generating Submission...")
    ensemble.generate_submission(X_test, test_ids)

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created"

    # Verify submission content
    sub_df = pd.read_csv(submission_path)
    print(f"Submission file created with {len(sub_df)} rows.")
    print(sub_df.head())

    assert list(sub_df.columns) == ["Id", "Pawpularity"], "Incorrect submission columns"
    assert len(sub_df) == len(test_ids), "Submission row count mismatch"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
