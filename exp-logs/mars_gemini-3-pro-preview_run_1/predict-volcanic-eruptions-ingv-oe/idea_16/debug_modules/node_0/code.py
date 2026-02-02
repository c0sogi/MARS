import os
import pandas as pd
import numpy as np
import torch
import shutil
import warnings

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.feature_engineering import FeatureExtractor
from library.spectrogram_generator import SpectrogramProcessor
from library.dataset import VolcanoDataset
from library.model_vision import VolcanoEfficientNet
from library.engine_vision import VisionTrainer
from library.engine_tabular import TabularTrainer
from library.meta_learner import StackingEnsemble


def create_mini_metadata(n_train=20, n_val=10, n_test=10):
    """
    Creates smaller versions of the metadata files to speed up the demonstration.
    """
    print("Creating mini metadata files...")

    # Paths to original metadata
    orig_train = "./metadata/train.csv"
    orig_val = "./metadata/val.csv"
    orig_test = "./metadata/test.csv"

    # Destination paths
    mini_train_path = "./working/mini_train.csv"
    mini_val_path = "./working/mini_val.csv"
    mini_test_path = "./working/mini_test.csv"

    # Sample and save
    pd.read_csv(orig_train).head(n_train).to_csv(mini_train_path, index=False)
    pd.read_csv(orig_val).head(n_val).to_csv(mini_val_path, index=False)
    pd.read_csv(orig_test).head(n_test).to_csv(mini_test_path, index=False)

    return mini_train_path, mini_val_path, mini_test_path


def patch_config(mini_train_path, mini_val_path, mini_test_path):
    """
    Modifies the Config class in-place to use mini datasets and reduced hyperparameters.
    """
    print("Patching configuration for fast execution...")

    # 1. Update Directories
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_submission"

    # Create these directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Update Metadata Paths
    Config.TRAIN_METADATA_PATH = mini_train_path
    Config.VAL_METADATA_PATH = mini_val_path
    Config.TEST_METADATA_PATH = mini_test_path

    # 3. Update Derived Cache Paths (must match new WORKING_DIR)
    Config.CACHE_DIR = Config.WORKING_DIR
    Config.TRAIN_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "train_features.parquet"
    )
    Config.VAL_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "test_features.parquet"
    )

    Config.SPECTROGRAM_CACHE_DIR = os.path.join(Config.WORKING_DIR, "spectrograms_demo")
    os.makedirs(Config.SPECTROGRAM_CACHE_DIR, exist_ok=True)

    # 4. Reduce Hyperparameters for Speed
    Config.N_FOLDS = 2
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4

    # LightGBM Speedup
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["early_stopping_rounds"] = 5
    Config.LGBM_PARAMS["num_leaves"] = 8  # Reduce complexity

    print("Configuration patched.")


def test_feature_engineering():
    print("\n--- Testing Feature Engineering (Tabular) ---")
    extractor = FeatureExtractor()

    # Process the mini train dataset
    # We force load_cached_data=False to demonstrate generation
    df_feats = extractor.process_dataset(
        Config.TRAIN_METADATA_PATH, Config.TRAIN_FEATURES_PATH, load_cached_data=False
    )

    # Verify output
    assert isinstance(df_feats, pd.DataFrame)
    assert len(df_feats) == 20  # We selected 20 rows for mini train
    assert Config.SEGMENT_ID_COL in df_feats.columns
    assert Config.TARGET_COL in df_feats.columns
    # Check for some expected feature columns (e.g., sensor_1_mean)
    assert "sensor_1_mean" in df_feats.columns

    print("Feature Engineering verified.")


def test_tabular_trainer():
    print("\n--- Testing Tabular Trainer (LightGBM) ---")
    trainer = TabularTrainer()

    # Run CV
    # This will generate OOF and Test predictions in WORKING_DIR
    trainer.run_cv(
        load_cached_data=True
    )  # Use the cache we just generated in test_feature_engineering

    # Verify Outputs
    oof_path = os.path.join(Config.WORKING_DIR, "tabular_oof.csv")
    test_path = os.path.join(Config.WORKING_DIR, "tabular_test.csv")

    assert os.path.exists(oof_path), "Tabular OOF file missing"
    assert os.path.exists(test_path), "Tabular Test file missing"

    df_oof = pd.read_csv(oof_path)
    # Total samples in OOF = train (20) + val (10) = 30
    assert len(df_oof) == 30, f"Expected 30 OOF predictions, got {len(df_oof)}"

    print("Tabular Trainer verified.")


def test_spectrogram_generation():
    print("\n--- Testing Spectrogram Generation ---")
    # This is implicitly tested by VisionTrainer, but let's verify the processor class directly
    processor = SpectrogramProcessor()

    # Generate for mini test set
    processor.generate_dataset(Config.TEST_METADATA_PATH, load_cached_data=False)

    # Check if files exist
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    sample_seg_id = df_test.iloc[0][Config.SEGMENT_ID_COL]
    expected_file = os.path.join(Config.SPECTROGRAM_CACHE_DIR, f"{sample_seg_id}.npy")

    assert os.path.exists(
        expected_file
    ), f"Spectrogram file {expected_file} not generated"

    # Check shape
    data = np.load(expected_file)
    # Shape: (10 sensors, 224, 224)
    assert data.shape == (10, 224, 224), f"Incorrect spectrogram shape: {data.shape}"

    print("Spectrogram Generation verified.")


def test_vision_model_and_dataset():
    print("\n--- Testing Vision Model and Dataset ---")

    # 1. Dataset
    # We use the mini test set for which we just generated spectrograms
    ds = VolcanoDataset(Config.TEST_METADATA_PATH, is_test=True)
    img, seg_id = ds[0]

    assert isinstance(img, torch.Tensor)
    assert img.shape == (10, 224, 224)
    assert isinstance(seg_id, int) or isinstance(seg_id, np.integer)

    # 2. Model
    model = VolcanoEfficientNet(pretrained=False)  # False for speed/no-download check
    model.eval()

    # Forward pass with a batch of size 2
    dummy_input = torch.stack([ds[0][0], ds[1][0]])
    with torch.no_grad():
        output = model(dummy_input)

    # Output shape should be (Batch, 1)
    assert output.shape == (2, 1)

    print("Vision Model and Dataset verified.")


def test_vision_trainer():
    print("\n--- Testing Vision Trainer (EfficientNet) ---")
    trainer = VisionTrainer()

    # Run CV
    # This ensures spectrograms for train/val are generated, trains for 1 epoch, generates OOF
    trainer.run_cv()

    # Verify Outputs
    oof_path = os.path.join(Config.WORKING_DIR, "vision_oof.csv")
    test_path = os.path.join(Config.WORKING_DIR, "vision_test.csv")

    assert os.path.exists(oof_path), "Vision OOF file missing"
    assert os.path.exists(test_path), "Vision Test file missing"

    df_oof = pd.read_csv(oof_path)
    assert len(df_oof) == 30, f"Expected 30 OOF predictions, got {len(df_oof)}"

    print("Vision Trainer verified.")


def test_meta_learner():
    print("\n--- Testing Meta Learner (Stacking) ---")
    stacker = StackingEnsemble()

    # 1. Train Meta Model
    mae = stacker.train_meta_model()
    assert mae >= 0, "MAE should be non-negative"

    # 2. Predict Final Submission
    stacker.predict()

    # Verify Submission
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(sub_path), "Final submission file missing"

    df_sub = pd.read_csv(sub_path)
    assert len(df_sub) == 10, f"Expected 10 test predictions, got {len(df_sub)}"
    assert Config.SEGMENT_ID_COL in df_sub.columns
    assert Config.TARGET_COL in df_sub.columns

    print("Meta Learner verified.")


def main():
    # Ensure reproducibility
    seed_everything(42)

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # 1. Prepare Data
    mini_train, mini_val, mini_test = create_mini_metadata()

    # 2. Configure Environment
    patch_config(mini_train, mini_val, mini_test)

    # 3. Execute Pipeline Components
    test_feature_engineering()
    test_tabular_trainer()
    test_spectrogram_generation()
    test_vision_model_and_dataset()
    test_vision_trainer()
    test_meta_learner()

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
