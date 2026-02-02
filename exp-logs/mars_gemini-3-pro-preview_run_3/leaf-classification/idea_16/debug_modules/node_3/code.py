import os
import pandas as pd
import numpy as np
import pickle
import shutil

# Import provided library components
from library.config import Config
from library.utils import seed_everything, clip_probabilities
from library.trainer import KFoldTrainer


def main():
    # ==========================================
    # 1. Setup Demo Data (Subset)
    # ==========================================
    print("Setting up demo data...")

    # Load original metadata
    train_meta_path = "./metadata/train.csv"
    test_meta_path = "./metadata/test.csv"

    if not os.path.exists(train_meta_path) or not os.path.exists(test_meta_path):
        raise FileNotFoundError("Metadata files not found in ./metadata/")

    df_train = pd.read_csv(train_meta_path)
    df_test = pd.read_csv(test_meta_path)

    # Select top 3 classes to ensure we have enough samples for LDA
    # (LDA requires samples per class > 1, and we need enough for splits)
    top_classes = df_train["species"].value_counts().head(3).index.tolist()
    print(f"Selected demo classes: {top_classes}")

    # Filter training data
    df_demo_train = df_train[df_train["species"].isin(top_classes)].copy()

    # Select a small subset of test data (5 images)
    df_demo_test = df_test.head(5).copy()

    # Define paths for demo artifacts
    demo_dir = "./working/demo_run"
    os.makedirs(demo_dir, exist_ok=True)

    demo_train_path = os.path.join(demo_dir, "demo_train.csv")
    demo_test_path = os.path.join(demo_dir, "demo_test.csv")

    # Save demo CSVs
    df_demo_train.to_csv(demo_train_path, index=False)
    df_demo_test.to_csv(demo_test_path, index=False)

    print(f"Demo train set: {len(df_demo_train)} samples")
    print(f"Demo test set: {len(df_demo_test)} samples")

    # ==========================================
    # 2. Configure Library for Demo
    # ==========================================
    print("\nConfiguring library parameters...")

    # Override Config attributes to use demo paths and settings
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.MODEL_DIR = os.path.join(demo_dir, "models")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    Config.TRAIN_CSV = demo_train_path
    Config.TEST_CSV = demo_test_path

    # Adjust hyperparameters for the small dataset
    Config.N_FOLDS = 2
    Config.N_CLASSES = len(top_classes)
    # LDA components must be <= min(n_classes-1, n_features)
    # With 3 classes, max components is 2.
    Config.LDA_COMPONENTS = Config.N_CLASSES - 1

    # Re-run setup to create the new directories
    Config.setup()

    # ==========================================
    # 3. Verify Utility Functions
    # ==========================================
    print("\nVerifying utilities...")
    seed_everything(42)

    # Test probability clipping
    test_probs = np.array([-0.1, 0.0, 0.5, 1.0, 1.1])
    clipped = clip_probabilities(test_probs)
    epsilon = 1e-15
    assert np.all(clipped >= epsilon), "Probabilities not clipped to min."
    assert np.all(clipped <= 1.0 - epsilon), "Probabilities not clipped to max."
    print("Utilities verified.")

    # ==========================================
    # 4. Run Training Pipeline
    # ==========================================
    print("\nInitializing Trainer...")
    trainer = KFoldTrainer()

    print("Starting K-Fold Training (Feature Extraction + Model Fitting)...")
    # load_cached_data=False forces the pipeline to run feature extraction
    # on our new demo dataset instead of looking for existing cache.
    trainer.train_kfold_ensemble(load_cached_data=False)

    # Verify models were saved
    print("Verifying model artifacts...")
    for fold in range(Config.N_FOLDS):
        model_path = os.path.join(Config.MODEL_DIR, f"model_fold_{fold}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model for fold {fold} was not created at {model_path}"
            )

        # Load model to check integrity
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            # Check if classes_ attribute matches our demo classes
            assert (
                len(model.classes_) == Config.N_CLASSES
            ), f"Model classes count mismatch. Expected {Config.N_CLASSES}, got {len(model.classes_)}"

    # ==========================================
    # 5. Run Inference Pipeline
    # ==========================================
    print("\nGenerating Submission...")
    trainer.generate_submission(load_cached_data=False)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not created at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check dimensions: Rows = test samples, Cols = id + n_classes
    expected_rows = len(df_demo_test)
    expected_cols = 1 + Config.N_CLASSES

    assert (
        len(df_sub) == expected_rows
    ), f"Submission rows mismatch. Expected {expected_rows}, got {len(df_sub)}"
    assert (
        len(df_sub.columns) == expected_cols
    ), f"Submission cols mismatch. Expected {expected_cols}, got {len(df_sub.columns)}"

    # Check if probabilities are within valid range
    prob_cols = df_sub.columns[1:]
    probs = df_sub[prob_cols].values
    assert np.all(
        (probs >= 0) & (probs <= 1)
    ), "Submission contains invalid probabilities."

    print("\nSUCCESS: Full pipeline demonstration completed.")
    print(f"Submission saved to: {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
