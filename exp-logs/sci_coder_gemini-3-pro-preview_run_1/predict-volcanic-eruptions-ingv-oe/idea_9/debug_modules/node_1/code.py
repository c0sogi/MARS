import os
import shutil
import numpy as np
import pandas as pd
import torch
import sys

# Import library modules
from library.config import Config
from library.utils import seed_everything, calc_mae
from library.feature_engineering import process_dataset
from library.vision_processing import process_vision_dataset
from library.dataset import VolcanoDataset, get_dataloaders
from library.models import LGBMRegressorWrapper, EfficientNet10Ch, RidgeStacker
from library.engine import Engine


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration for Fast Demonstration
    # --------------------------------------------------------------------------
    print(">>> Configuring environment for fast demonstration...")

    # Enable Debug Mode to process only a small subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10  # Process only 10 samples per dataset

    # Reduce Cross-Validation Folds
    Config.N_FOLDS = 2

    # Reduce Model Hyperparameters for Speed
    Config.CNN_TRAIN_PARAMS["epochs"] = 1
    Config.CNN_TRAIN_PARAMS["batch_size"] = 2
    Config.CNN_TRAIN_PARAMS["patience"] = 1

    # LightGBM Speedup
    Config.LGB_PARAMS["n_estimators"] = 10
    Config.LGB_PARAMS["early_stopping_rounds"] = 5
    Config.LGB_PARAMS["verbose"] = -1

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Clean working directory to ensure fresh execution
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # 2. Verify Feature Engineering (Tabular Branch)
    # --------------------------------------------------------------------------
    print("\n>>> Testing Feature Engineering (Tabular)...")
    # Force generation from scratch
    X_tab, y_tab = process_dataset("train", load_cached_data=False)

    # Validation
    assert isinstance(X_tab, pd.DataFrame), "Tabular features should be a DataFrame"
    assert (
        len(X_tab) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} rows"
    assert y_tab is not None, "Targets should not be None for training data"
    assert len(y_tab) == Config.DEBUG_SAMPLE_SIZE, "Target length mismatch"
    assert "sensor_beamformed_mean" in X_tab.columns, "Beamformed feature missing"

    print(f"Tabular Data Shape: {X_tab.shape}")
    print("Tabular feature engineering verification passed.")

    # --------------------------------------------------------------------------
    # 3. Verify Vision Processing (Spectrogram Branch)
    # --------------------------------------------------------------------------
    print("\n>>> Testing Vision Processing...")
    # Force generation from scratch
    X_vis, y_vis, ids_vis = process_vision_dataset("train", load_cached_data=False)

    # Validation
    assert isinstance(X_vis, np.ndarray), "Vision data should be a numpy array"
    # Shape: (N, Channels, Height, Width) -> (10, 10, 224, 224)
    expected_shape = (Config.DEBUG_SAMPLE_SIZE, 10, 224, 224)
    assert (
        X_vis.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {X_vis.shape}"
    assert len(y_vis) == Config.DEBUG_SAMPLE_SIZE, "Target length mismatch"

    print(f"Vision Data Shape: {X_vis.shape}")
    print("Vision processing verification passed.")

    # --------------------------------------------------------------------------
    # 4. Verify Dataset and DataLoader
    # --------------------------------------------------------------------------
    print("\n>>> Testing Dataset & DataLoader...")
    ds = VolcanoDataset(X_vis, y_vis, ids_vis)
    dl = torch.utils.data.DataLoader(ds, batch_size=2)

    # Fetch one batch
    batch_imgs, batch_targets = next(iter(dl))

    # Validation
    assert batch_imgs.shape == (2, 10, 224, 224), "Incorrect batch image shape"
    assert batch_targets.shape == (2,), "Incorrect batch target shape"
    assert batch_imgs.dtype == torch.float32, "Images should be float32"

    print("DataLoader verification passed.")

    # --------------------------------------------------------------------------
    # 5. Verify Models
    # --------------------------------------------------------------------------

    # A. LightGBM
    print("\n>>> Testing LightGBM Wrapper...")
    lgbm = LGBMRegressorWrapper()
    # Split small data for fit
    split_idx = Config.DEBUG_SAMPLE_SIZE // 2
    lgbm.fit(
        X_tab.iloc[:split_idx],
        y_tab[:split_idx],
        X_tab.iloc[split_idx:],
        y_tab[split_idx:],
    )
    preds_lgbm = lgbm.predict(X_tab.iloc[:2])
    assert len(preds_lgbm) == 2, "LGBM prediction shape mismatch"
    print("LightGBM verification passed.")

    # B. EfficientNet
    print("\n>>> Testing EfficientNet Model...")
    cnn = EfficientNet10Ch()
    cnn.eval()
    with torch.no_grad():
        # Pass the batch fetched earlier
        out = cnn(batch_imgs)

    assert out.shape == (2, 1), f"Expected CNN output (2, 1), got {out.shape}"
    print("EfficientNet verification passed.")

    # C. Ridge Stacker
    print("\n>>> Testing Ridge Stacker...")
    stacker = RidgeStacker()
    # Create dummy OOF predictions (N_samples, N_models)
    dummy_X = np.random.rand(10, 2)
    dummy_y = np.random.rand(10)
    stacker.fit(dummy_X, dummy_y)
    stack_preds = stacker.predict(dummy_X[:2])
    assert len(stack_preds) == 2, "Stacker prediction shape mismatch"
    print("Ridge Stacker verification passed.")

    # --------------------------------------------------------------------------
    # 6. Verify Full Engine Execution
    # --------------------------------------------------------------------------
    print("\n>>> Testing Engine (Full Pipeline Integration)...")

    # Instantiate Engine
    engine = Engine()

    # Run the pipeline
    # This will:
    # 1. Load/Generate features for Train, Val, and Test (respecting DEBUG limit)
    # 2. Train LGBM (Branch A)
    # 3. Train EfficientNet (Branch B)
    # 4. Stack predictions
    # 5. Save submission
    engine.run()

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"\nSubmission generated with {len(df_sub)} rows.")

    assert "segment_id" in df_sub.columns, "Submission missing segment_id"
    assert "time_to_eruption" in df_sub.columns, "Submission missing time_to_eruption"
    # In debug mode with disjoint test files, we expect some rows.
    # Since we process 10 test files (DEBUG_SAMPLE_SIZE), we expect 10 rows.
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} predictions"

    print("Engine pipeline verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
