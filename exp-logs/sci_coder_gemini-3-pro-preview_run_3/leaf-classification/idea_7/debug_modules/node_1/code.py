import os
import shutil
import numpy as np
import pandas as pd
import torch

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, clip_probabilities, calculate_log_loss
from library.data_loader import LeafDataManager
from library.feature_engine import DualStreamExtractor
from library.cv_runner import StratifiedEnsembleTrainer
from library.inference_engine import EnsemblePredictor


def main():
    print("=== Starting Library Usage Demo ===")

    # ------------------------------------------------------------------------
    # 1. Configuration & Setup
    # ------------------------------------------------------------------------
    print("\n[1] Configuring Environment...")

    # Set seed for reproducibility
    seed_everything(42)

    # Override Config for the demo to run quickly and safely in the working dir
    Config.N_FOLDS = 2  # Reduce folds to 2 for speed
    Config.CACHE_DIR = "./working/demo_cache"
    Config.SUBMISSION_PATH = "./working/submission_demo.csv"

    # Update derived paths in Config to point to the new cache directory
    # (Since these are class attributes initialized at import, we update them manually)
    Config.CACHE_TRAIN_DINO = os.path.join(Config.CACHE_DIR, "train_dino_features.npy")
    Config.CACHE_TRAIN_CONV = os.path.join(Config.CACHE_DIR, "train_conv_features.npy")
    Config.CACHE_TEST_DINO = os.path.join(Config.CACHE_DIR, "test_dino_features.npy")
    Config.CACHE_TEST_CONV = os.path.join(Config.CACHE_DIR, "test_conv_features.npy")
    Config.CACHE_TRAIN_TABULAR = os.path.join(Config.CACHE_DIR, "train_tabular.npy")
    Config.CACHE_TEST_TABULAR = os.path.join(Config.CACHE_DIR, "test_tabular.npy")
    Config.CACHE_TRAIN_LABELS = os.path.join(Config.CACHE_DIR, "train_labels.npy")
    Config.CACHE_TEST_IDS = os.path.join(Config.CACHE_DIR, "test_ids.npy")
    Config.CACHE_CLASSES = os.path.join(Config.CACHE_DIR, "classes.npy")
    Config.PIPELINE_PATH = os.path.join(Config.CACHE_DIR, "pipeline_fold_{fold}.pkl")

    # Clean up previous demo run if exists
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    print(f"Cache Directory: {Config.CACHE_DIR}")
    print(f"N_FOLDS: {Config.N_FOLDS}")

    # ------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # ------------------------------------------------------------------------
    print("\n[2] Verifying Utilities...")

    # Test clip_probabilities
    probs = np.array([[-0.5, 0.5, 1.5], [0.0, 1.0, 0.0]])
    clipped = clip_probabilities(probs)
    assert clipped.min() >= 1e-15, "Probabilities not clipped correctly (min)"
    assert clipped.max() <= (1.0 - 1e-15), "Probabilities not clipped correctly (max)"
    print(" - clip_probabilities: OK")

    # Test calculate_log_loss
    y_true = np.array([0, 1])
    # Preds: High prob for correct class
    y_pred = np.array([[0.9, 0.1], [0.2, 0.8]])
    loss = calculate_log_loss(y_true, y_pred)
    assert loss < 0.5, f"Log loss should be low for good predictions, got {loss}"
    print(f" - calculate_log_loss: {loss:.4f} (OK)")

    # ------------------------------------------------------------------------
    # 3. Data Loading
    # ------------------------------------------------------------------------
    print("\n[3] Loading Data...")
    dm = LeafDataManager()

    # Load Train
    # Note: load_cached_data=False forces reading from metadata CSVs
    X_tab_train, y_train, paths_train, ids_train = dm.load_train_data(
        load_cached_data=False
    )

    # Load Test
    X_tab_test, ids_test, paths_test = dm.load_test_data(load_cached_data=False)

    # Validations
    print(f" - Train Samples: {len(X_tab_train)}")
    print(f" - Test Samples: {len(X_tab_test)}")

    assert X_tab_train.shape[1] == 192, "Incorrect tabular feature count (Train)"
    assert X_tab_test.shape[1] == 192, "Incorrect tabular feature count (Test)"
    assert len(y_train) == len(
        X_tab_train
    ), "Mismatch between train features and labels"
    assert len(paths_train) == len(
        X_tab_train
    ), "Mismatch between train features and paths"

    # ------------------------------------------------------------------------
    # 4. Feature Extraction (Dual Stream)
    # ------------------------------------------------------------------------
    print("\n[4] Extracting Features (This uses the GPU)...")
    extractor = DualStreamExtractor()

    # Extract Train Features
    # Note: This will process images. With ~900 images and A100, this is fast.
    dino_train, conv_train = extractor.get_train_features(load_cached_data=False)

    # Extract Test Features
    dino_test, conv_test = extractor.get_test_features(load_cached_data=False)

    # Validations
    # DINOv2 Large embedding dim = 1024
    # ConvNeXt Large embedding dim = 1536
    print(f" - DINO Train Shape: {dino_train.shape}")
    print(f" - ConvNeXt Train Shape: {conv_train.shape}")

    assert (
        dino_train.shape[1] == 1024
    ), f"Expected DINO dim 1024, got {dino_train.shape[1]}"
    assert (
        conv_train.shape[1] == 1536
    ), f"Expected ConvNeXt dim 1536, got {conv_train.shape[1]}"
    assert len(dino_train) == len(X_tab_train), "DINO train count mismatch"
    assert len(conv_train) == len(X_tab_train), "ConvNeXt train count mismatch"

    assert len(dino_test) == len(X_tab_test), "DINO test count mismatch"

    # ------------------------------------------------------------------------
    # 5. Model Training (Cross-Validation)
    # ------------------------------------------------------------------------
    print("\n[5] Running Cross-Validation...")
    trainer = StratifiedEnsembleTrainer()

    # Run CV
    fold_scores = trainer.cross_validate(dino_train, conv_train, X_tab_train, y_train)

    # Validations
    print(f" - Fold Scores: {fold_scores}")
    assert (
        len(fold_scores) == Config.N_FOLDS
    ), f"Expected {Config.N_FOLDS} scores, got {len(fold_scores)}"

    # Verify pipeline files were created
    for i in range(Config.N_FOLDS):
        p_path = Config.PIPELINE_PATH.format(fold=i)
        assert os.path.exists(
            p_path
        ), f"Pipeline file for fold {i} not found at {p_path}"

    # ------------------------------------------------------------------------
    # 6. Inference & Submission
    # ------------------------------------------------------------------------
    print("\n[6] Generating Submission...")
    predictor = EnsemblePredictor()

    # Create submission
    predictor.create_submission(dino_test, conv_test, X_tab_test, ids_test)

    # Validations
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f" - Submission Shape: {df_sub.shape}")

    # Expected shape: (99 test samples, 1 id col + 99 classes)
    expected_cols = 1 + 99
    assert df_sub.shape[0] == 99, f"Expected 99 rows, got {df_sub.shape[0]}"
    assert (
        df_sub.shape[1] == expected_cols
    ), f"Expected {expected_cols} columns, got {df_sub.shape[1]}"

    # Check if probabilities are valid
    prob_cols = [c for c in df_sub.columns if c != "id"]
    probs = df_sub[prob_cols].values
    assert probs.min() >= 0.0, "Negative probabilities found"
    assert probs.max() <= 1.0, "Probabilities > 1.0 found"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
