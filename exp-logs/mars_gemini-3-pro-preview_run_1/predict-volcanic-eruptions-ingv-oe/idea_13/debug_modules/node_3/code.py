import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.feature_engineering import generate_features
from library.data_loader import get_data_loaders
from library.models_vision import ScalarConditionedEfficientNet
from library.models_tabular import LightGBMTrainer
from library.training import run_training

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_config():
    """
    Sets up a temporary environment in ./working/demo_run and overrides
    Config parameters to ensure the demo runs quickly on a small subset of data.
    """
    print(">>> Setting up demo configuration...")

    # 1. Create Demo Directory Structure
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR)

    METADATA_DIR = os.path.join(DEMO_DIR, "metadata")
    os.makedirs(METADATA_DIR)

    # 2. Create Mini Metadata Files
    # We read the first few rows of the real metadata and save them to our demo dir.
    # This forces the pipeline to process only these few files.

    def create_mini_meta(src_rel_path, dst_filename, n_rows):
        src_path = os.path.join(Config.METADATA_DIR, src_rel_path)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Original metadata not found at {src_path}")

        df = pd.read_csv(src_path)
        df_mini = df.head(n_rows).copy()

        dst_path = os.path.join(METADATA_DIR, dst_filename)
        df_mini.to_csv(dst_path, index=False)
        return dst_path

    # Create subsets: 12 train, 8 val, 5 test
    Config.TRAIN_METADATA = create_mini_meta("train.csv", "train.csv", 12)
    Config.VAL_METADATA = create_mini_meta("val.csv", "val.csv", 8)
    Config.TEST_METADATA = create_mini_meta("test.csv", "test.csv", 5)

    # 3. Redirect Output and Cache Paths to Demo Directory
    Config.IDEA_DIR = os.path.join(DEMO_DIR, "idea_demo")
    os.makedirs(Config.IDEA_DIR, exist_ok=True)

    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Update cache paths to point to the new demo directory
    Config.CACHE_TRAIN_FEATURES = os.path.join(
        Config.IDEA_DIR, "train_features.parquet"
    )
    Config.CACHE_VAL_FEATURES = os.path.join(Config.IDEA_DIR, "val_features.parquet")
    Config.CACHE_TEST_FEATURES = os.path.join(Config.IDEA_DIR, "test_features.parquet")
    Config.CACHE_GLOBAL_MAX = os.path.join(
        Config.IDEA_DIR, "global_max_spectrogram.npy"
    )

    Config.CACHE_SPECTROGRAMS_TRAIN = os.path.join(
        Config.IDEA_DIR, "spectrograms_train"
    )
    Config.CACHE_SPECTROGRAMS_VAL = os.path.join(Config.IDEA_DIR, "spectrograms_val")
    Config.CACHE_SPECTROGRAMS_TEST = os.path.join(Config.IDEA_DIR, "spectrograms_test")

    # Ensure cache directories exist
    for d in [
        Config.CACHE_SPECTROGRAMS_TRAIN,
        Config.CACHE_SPECTROGRAMS_VAL,
        Config.CACHE_SPECTROGRAMS_TEST,
    ]:
        os.makedirs(d, exist_ok=True)

    # 4. Override Hyperparameters for Speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 12  # Match the metadata subset size
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.N_FOLDS = 2  # Only 2 folds for CV
    Config.PATIENCE = 1

    # LightGBM Speedup
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["early_stopping_rounds"] = 5
    Config.LGBM_PARAMS["verbose"] = -1

    # Check Device
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Running on CPU.")
        Config.DEVICE = "cpu"

    print(">>> Configuration updated for demo execution.")


def test_feature_engineering():
    print("\n>>> Testing Feature Engineering...")

    # Trigger feature generation.
    # This will read our mini metadata, compute global max, and generate spectrograms/features.
    df_train, df_val, df_test = generate_features(load_cached=False)

    # Validation
    assert len(df_train) == 12, f"Expected 12 train samples, got {len(df_train)}"
    assert len(df_val) == 8, f"Expected 8 val samples, got {len(df_val)}"
    assert len(df_test) == 5, f"Expected 5 test samples, got {len(df_test)}"

    # Check for scalar injection columns
    scalar_cols = [c for c in df_train.columns if c.startswith("scalar_")]
    assert (
        len(scalar_cols) == Config.SCALAR_INPUT_DIM
    ), f"Expected {Config.SCALAR_INPUT_DIM} scalar features, found {len(scalar_cols)}"

    # Check for target variable
    assert (
        "time_to_eruption" in df_train.columns
    ), "Target column missing in train features"

    # Verify spectrogram files were created
    files_created = os.listdir(Config.CACHE_SPECTROGRAMS_TRAIN)
    assert (
        len(files_created) == 12
    ), "Spectrogram files not generated for all train samples"

    print("Feature Engineering logic verified.")
    return df_train, df_val, df_test


def test_vision_pipeline(df_train, df_val, df_test):
    print("\n>>> Testing Vision Pipeline...")

    # 1. Test Data Loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        df_train, df_val, df_test, batch_size=Config.BATCH_SIZE, num_workers=0
    )

    # Fetch a single batch
    specs, scalars, targets = next(iter(train_loader))

    # Verify Shapes
    # Spectrogram: (Batch, Channels, Height, Width)
    expected_spec_shape = (
        Config.BATCH_SIZE,
        Config.IN_CHANNELS,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    )
    assert (
        specs.shape == expected_spec_shape
    ), f"Spectrogram shape mismatch. Expected {expected_spec_shape}, got {specs.shape}"

    # Scalars: (Batch, Scalar_Dim)
    assert scalars.shape == (
        Config.BATCH_SIZE,
        Config.SCALAR_INPUT_DIM,
    ), f"Scalar shape mismatch. Expected {(Config.BATCH_SIZE, Config.SCALAR_INPUT_DIM)}, got {scalars.shape}"

    # Targets: (Batch, 1)
    assert targets.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Target shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {targets.shape}"

    print("Data Loader shapes verified.")

    # 2. Test Model Forward & Backward Pass
    # We use pretrained=False here to avoid download overhead during this specific check,
    # though the main training loop uses pretrained=True.
    model = ScalarConditionedEfficientNet(pretrained=False)
    model.to(Config.DEVICE)

    specs = specs.to(Config.DEVICE)
    scalars = scalars.to(Config.DEVICE)
    targets = targets.to(Config.DEVICE)

    # Forward
    outputs = model(specs, scalars)
    assert outputs.shape == (Config.BATCH_SIZE, 1), "Model output shape is incorrect"

    # Backward (Check gradient flow)
    criterion = nn.L1Loss()
    loss = criterion(outputs, targets)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Vision Model Forward/Backward pass verified.")


def test_tabular_pipeline(df_train, df_val, df_test):
    print("\n>>> Testing Tabular Pipeline...")

    trainer = LightGBMTrainer()

    # Train on the mini dataset
    # We use a dummy fold_id just for file naming
    model = trainer.train(df_train, df_val, fold_id=99)

    # Predict on test set
    preds = trainer.predict(df_test, model=model)

    # Verify predictions
    assert isinstance(preds, np.ndarray), "Predictions should be a numpy array"
    assert len(preds) == len(
        df_test
    ), f"Expected {len(df_test)} predictions, got {len(preds)}"

    print("Tabular Model training and prediction verified.")


def test_full_integration():
    print("\n>>> Testing Full Training Pipeline (run_training)...")

    # This function orchestrates the entire process:
    # 1. Loads data (uses cache we generated)
    # 2. Runs K-Fold CV for LightGBM
    # 3. Runs K-Fold CV for EfficientNet
    # 4. Trains Meta-Learner
    # 5. Generates Submission
    run_training()

    # Verify Submission File
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check submission format
    assert "segment_id" in df_sub.columns and "time_to_eruption" in df_sub.columns
    assert len(df_sub) == 5, f"Expected 5 rows in submission, got {len(df_sub)}"

    print(
        f"Full pipeline completed successfully. Submission generated at {Config.SUBMISSION_PATH}"
    )


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(42)

    # 1. Setup Environment and Config
    setup_demo_config()

    # 2. Verify Feature Engineering (Tabular + Spectrograms)
    df_train, df_val, df_test = test_feature_engineering()

    # 3. Verify Vision Branch Components
    test_vision_pipeline(df_train, df_val, df_test)

    # 4. Verify Tabular Branch Components
    test_tabular_pipeline(df_train, df_val, df_test)

    # 5. Verify Full Orchestration
    test_full_integration()

    print("\n>>> All demonstrations passed successfully.")
