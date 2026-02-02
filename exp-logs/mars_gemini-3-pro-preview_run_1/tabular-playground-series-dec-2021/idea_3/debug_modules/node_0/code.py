import os
import shutil
import pandas as pd
import numpy as np
import torch
import warnings

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.data_factory import DataFactory
from library.arch_resnet import TabularResNet
from library.trainer_xgb import train_xgboost_model
from library.trainer_resnet import train_neural_network
from library.ensemble_runner import EnsembleRunner

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_environment():
    """
    Creates a small subset of data and patches the Config class
    to ensure the demo runs quickly.
    """
    print("Setting up demo environment...")

    # Define demo paths
    demo_base = "./working/demo_env"
    demo_input = os.path.join(demo_base, "input")
    demo_cache = os.path.join(demo_base, "cache")
    demo_submission = os.path.join(demo_base, "submission")

    os.makedirs(demo_input, exist_ok=True)
    os.makedirs(demo_cache, exist_ok=True)
    os.makedirs(demo_submission, exist_ok=True)

    # 1. Create Data Subsets (1000 rows each)
    # We read from the actual metadata files provided in the environment
    orig_train_path = "./metadata/train.csv"
    orig_val_path = "./metadata/val.csv"
    orig_test_path = "./metadata/test.csv"

    # Helper to save subset
    def save_subset(src, dst, n=1000):
        if os.path.exists(src):
            df = pd.read_csv(src, nrows=n)
            df.to_csv(dst, index=False)
        else:
            # Fallback for local testing if metadata doesn't exist
            print(f"Warning: {src} not found. Creating dummy data.")
            # Create dummy based on expected schema
            cols = ["Id", "Cover_Type"] + [f"Feat_{i}" for i in range(54)]
            if "test" in dst:
                cols.remove("Cover_Type")
            df = pd.DataFrame(np.random.randn(n, len(cols)), columns=cols)
            df["Id"] = np.arange(n)
            if "Cover_Type" in df.columns:
                df["Cover_Type"] = np.random.randint(1, 8, n)
            df.to_csv(dst, index=False)

    save_subset(orig_train_path, os.path.join(demo_input, "train.csv"))
    save_subset(orig_val_path, os.path.join(demo_input, "val.csv"))
    save_subset(orig_test_path, os.path.join(demo_input, "test.csv"))

    # 2. Patch Config
    # We modify the Config class attributes directly so all library modules see the changes
    Config.TRAIN_PATH = os.path.join(demo_input, "train.csv")
    Config.VAL_PATH = os.path.join(demo_input, "val.csv")
    Config.TEST_PATH = os.path.join(demo_input, "test.csv")

    Config.WORKING_DIR = demo_cache
    Config.TRAIN_PROCESSED_PATH = os.path.join(demo_cache, "train_processed.parquet")
    Config.VAL_PROCESSED_PATH = os.path.join(demo_cache, "val_processed.parquet")
    Config.TEST_PROCESSED_PATH = os.path.join(demo_cache, "test_processed.parquet")

    Config.SUBMISSION_DIR = demo_submission
    Config.SUBMISSION_PATH = os.path.join(demo_submission, "submission.csv")

    # Reduce Hyperparameters for Speed
    Config.N_FOLDS = 2

    # XGBoost
    Config.XGB_FIT_PARAMS["num_boost_round"] = 5
    Config.XGB_FIT_PARAMS["early_stopping_rounds"] = 2
    Config.XGB_FIT_PARAMS["verbose_eval"] = False

    # Neural Network
    Config.NN_PARAMS["epochs"] = 1
    Config.NN_PARAMS["batch_size"] = 32
    Config.NN_PARAMS["hidden_dims"] = [32, 16]  # Smaller network

    # Meta Learner
    Config.META_PARAMS["max_iter"] = 10

    print("Demo environment setup complete.")


def verify_data_factory():
    print("\n--- Verifying DataFactory ---")

    # Force reload from scratch (ignore cache initially)
    train_df, val_df, test_df, test_ids = DataFactory.load_and_engineer_data(
        load_cached_data=False
    )

    # Assertions
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Check if engineered features exist
    expected_features = [
        "Euclidean_Distance_To_Hydrology",
        "Relative_Elevation",
        "Aspect_Sin",
        "Aspect_Cos",
    ]
    for feat in expected_features:
        if feat not in train_df.columns:
            raise AssertionError(
                f"Feature engineering failed: {feat} missing from train_df"
            )

    # Check that Id column was dropped from train/val but kept/extracted for test
    if Config.ID_COL in train_df.columns:
        raise AssertionError("Id column should be dropped from training data")

    assert len(test_ids) == len(test_df), "Mismatch in Test IDs length"
    print("DataFactory verification passed.")
    return train_df.shape[1] - 1  # Input dim (minus target)


def verify_resnet_arch(input_dim):
    print("\n--- Verifying TabularResNet Architecture ---")

    batch_size = 4
    num_classes = Config.NUM_CLASSES

    model = TabularResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=[32, 16],
        dropout_rate=0.1,
        use_batch_norm=True,
    )

    # Create dummy input
    x = torch.randn(batch_size, input_dim)

    # Forward pass
    model.eval()
    with torch.no_grad():
        out = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    if out.shape != (batch_size, num_classes):
        raise AssertionError(
            f"Model output shape mismatch. Expected {(batch_size, num_classes)}, got {out.shape}"
        )

    print("TabularResNet verification passed.")


def verify_trainers():
    print("\n--- Verifying Trainers (XGBoost & ResNet) ---")

    # 1. XGBoost
    print("Running XGBoost Trainer...")
    booster, predict_fn = train_xgboost_model(load_cached_data=True)
    assert booster is not None, "XGBoost booster is None"

    # 2. ResNet
    print("Running ResNet Trainer...")
    # This runs the full training loop defined in trainer_resnet.py (patched to 1 epoch)
    test_probs = train_neural_network(load_cached_data=True)

    assert isinstance(
        test_probs, np.ndarray
    ), "ResNet predictions should be numpy array"
    assert (
        test_probs.shape[1] == Config.NUM_CLASSES
    ), f"ResNet output classes mismatch. Got {test_probs.shape[1]}"

    print("Trainers verification passed.")


def verify_ensemble_runner():
    print("\n--- Verifying EnsembleRunner (Integration Test) ---")

    runner = EnsembleRunner()
    runner.run_kfold_stacking()

    # Check submission
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError(f"Submission file not found at {Config.SUBMISSION_PATH}")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with shape: {df_sub.shape}")
    print(df_sub.head())

    expected_cols = [Config.ID_COL, Config.TARGET_COL]
    if list(df_sub.columns) != expected_cols:
        raise AssertionError(
            f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"
        )

    print("EnsembleRunner verification passed.")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # 1. Setup Demo Environment (Patch Config & Create Data)
    setup_demo_environment()

    # 2. Verify Data Pipeline
    # We get the input dimension here to pass to the architecture check
    input_dim = verify_data_factory()

    # 3. Verify Neural Network Architecture
    verify_resnet_arch(input_dim)

    # 4. Verify Individual Trainers
    verify_trainers()

    # 5. Verify Full Ensemble Pipeline
    verify_ensemble_runner()

    print("\nAll demonstrations and verifications completed successfully.")
