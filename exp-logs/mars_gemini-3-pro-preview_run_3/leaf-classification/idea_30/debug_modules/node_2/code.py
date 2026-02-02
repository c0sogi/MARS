import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import random
import warnings

# Import library components
from library.config import Config
from library.feature_extraction import FeatureExtractor
from library.data_processing import DatasetManager
from library.pipeline import ModelFactory
from library.train import Trainer
from library.inference import InferenceEngine

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def setup_demo_data(base_dir):
    """
    Creates a small subset of the metadata to ensure the demo runs quickly.
    Selects 3 classes and a few samples per class.
    """
    print(f"Creating subset metadata in {base_dir}...")
    os.makedirs(base_dir, exist_ok=True)

    # Load original metadata
    df_train_orig = pd.read_csv(Config.TRAIN_CSV)
    df_val_orig = pd.read_csv(Config.VAL_CSV)
    df_test_orig = pd.read_csv(Config.TEST_CSV)

    # Select top 3 classes to ensure stratification works with small N
    top_classes = df_train_orig["species"].value_counts().head(3).index.tolist()
    print(f"Selected subset classes: {top_classes}")

    # Subset Train: 4 samples per class (Total 12)
    df_train_sub = (
        df_train_orig[df_train_orig["species"].isin(top_classes)]
        .groupby("species")
        .head(4)
        .reset_index(drop=True)
    )

    # Subset Val: 2 samples per class (Total 6)
    df_val_sub = (
        df_val_orig[df_val_orig["species"].isin(top_classes)]
        .groupby("species")
        .head(2)
        .reset_index(drop=True)
    )

    # Subset Test: Just take first 6 rows (Total 6)
    df_test_sub = df_test_orig.head(6).reset_index(drop=True)

    # Save to demo directory
    train_path = os.path.join(base_dir, "train.csv")
    val_path = os.path.join(base_dir, "val.csv")
    test_path = os.path.join(base_dir, "test.csv")

    df_train_sub.to_csv(train_path, index=False)
    df_val_sub.to_csv(val_path, index=False)
    df_test_sub.to_csv(test_path, index=False)

    return (
        train_path,
        val_path,
        test_path,
        len(df_train_sub),
        len(df_val_sub),
        len(df_test_sub),
    )


def patch_config(demo_dir, train_path, val_path, test_path):
    """
    Overrides Config attributes to use the demo directory and subset data.
    """
    print("Patching Config for demo execution...")
    Config.WORKING_DIR = os.path.join(demo_dir, "working")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    Config.TRAIN_CSV = train_path
    Config.VAL_CSV = val_path
    Config.TEST_CSV = test_path

    # Reduce folds for speed
    Config.N_FOLDS = 2

    # Ensure directories exist
    Config.make_dirs()


def verify_feature_extraction():
    """
    Verifies that FeatureExtractor runs and produces correct shapes.
    """
    print("\n--- Verifying Feature Extraction ---")
    # Load the subset dataframe
    df_train = pd.read_csv(Config.TRAIN_CSV)

    extractor = FeatureExtractor()

    # Force extraction (load_cached_data=False) to test the model inference
    # Note: This uses the actual images from ./input/images based on the file_path in df
    features = extractor.extract_features(
        df_train, dataset_name="demo_train", load_cached_data=False
    )

    # Assertions
    n_samples = len(df_train)
    # DINO: (N, 12, 1024)
    assert features["dino_features"].shape == (
        n_samples,
        12,
        1024,
    ), f"DINO shape mismatch. Expected ({n_samples}, 12, 1024), got {features['dino_features'].shape}"
    # ConvNeXt: (N, 12, 1536)
    assert features["conv_features"].shape == (
        n_samples,
        12,
        1536,
    ), f"ConvNeXt shape mismatch. Expected ({n_samples}, 12, 1536), got {features['conv_features'].shape}"
    # IDs
    assert len(features["ids"]) == n_samples

    print("Feature Extraction Verification Passed.")


def verify_data_processing():
    """
    Verifies DatasetManager loads data and performs manifold densification correctly.
    """
    print("\n--- Verifying Data Processing & Densification ---")
    dm = DatasetManager()

    # Load data (will use the cache generated in verify_feature_extraction for train)
    # Test features will be extracted here since we haven't run them yet
    data = dm.load_data(load_cached_data=True)

    # Check combined train+val size
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    expected_train_total = len(df_train) + len(df_val)

    assert (
        len(data["train"]["ids"]) == expected_train_total
    ), f"Combined train size mismatch. Expected {expected_train_total}, got {len(data['train']['ids'])}"

    # Test Densification (prepare_training_set)
    # Should convert N samples to 3N samples (3 centroids per image)
    X, y, ids = dm.prepare_training_set(data["train"])

    expected_rows = expected_train_total * 3
    # Feature dim: 1024 (DINO) + 1536 (Conv) + 192 (Tabular) = 2752
    expected_cols = 1024 + 1536 + 192

    assert X.shape == (
        expected_rows,
        expected_cols,
    ), f"Densified X shape mismatch. Expected ({expected_rows}, {expected_cols}), got {X.shape}"
    assert y.shape == (expected_rows,), "Label shape mismatch"
    assert ids.shape == (expected_rows,), "ID shape mismatch"

    print("Data Processing Verification Passed.")


def verify_pipeline_logic():
    """
    Verifies that the ModelFactory creates a valid pipeline.
    """
    print("\n--- Verifying Pipeline Construction ---")
    dm = DatasetManager()
    indices = dm.get_feature_indices()

    factory = ModelFactory()
    pipeline = factory.create_pipeline(indices)

    # Check steps
    step_names = [name for name, _ in pipeline.steps]
    assert "preprocessor" in step_names, "Pipeline missing preprocessor"
    assert "scaler" in step_names, "Pipeline missing scaler"
    assert "classifier" in step_names, "Pipeline missing classifier"

    print("Pipeline Construction Verification Passed.")


def run_full_training_inference():
    """
    Runs the Trainer and InferenceEngine to ensure end-to-end execution.
    """
    print("\n--- Running Full Training & Inference Loop ---")

    # 1. Training
    trainer = Trainer()
    trainer.run()

    # Check if models are saved
    model_dir = os.path.join(Config.WORKING_DIR, "models")
    assert os.path.exists(
        os.path.join(model_dir, "classes.pkl")
    ), "classes.pkl not found"
    assert os.path.exists(
        os.path.join(model_dir, "pipeline_fold_0.pkl")
    ), "Fold 0 model not found"

    # 2. Inference
    # We run inference explicitly to test the engine class, though Trainer also calls generate_submission
    inference = InferenceEngine()
    inference.run(load_cached_data=True)

    # Check submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with shape: {df_sub.shape}")

    # Check columns: id + 3 classes (since we subsetted to 3 classes)
    # Wait, the pipeline trains on the subset, so it only knows 3 classes.
    # The submission should have id + 3 columns.
    df_train_sub = pd.read_csv(Config.TRAIN_CSV)
    num_classes = df_train_sub["species"].nunique()

    assert (
        df_sub.shape[1] == num_classes + 1
    ), f"Submission column count mismatch. Expected {num_classes + 1}, got {df_sub.shape[1]}"

    # Check probabilities range
    probs = df_sub.iloc[:, 1:].values
    assert np.all(probs >= 0) and np.all(
        probs <= 1 + 1e-6
    ), "Probabilities out of range"

    print("End-to-End Execution Verification Passed.")


if __name__ == "__main__":
    seed_everything(42)

    # Define demo directory
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR)

    try:
        # 1. Setup Data
        train_p, val_p, test_p, n_train, n_val, n_test = setup_demo_data(
            os.path.join(DEMO_DIR, "metadata")
        )

        # 2. Patch Config
        patch_config(DEMO_DIR, train_p, val_p, test_p)

        # 3. Verify Components
        verify_feature_extraction()
        verify_data_processing()
        verify_pipeline_logic()

        # 4. Run Full Loop
        run_full_training_inference()

        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
