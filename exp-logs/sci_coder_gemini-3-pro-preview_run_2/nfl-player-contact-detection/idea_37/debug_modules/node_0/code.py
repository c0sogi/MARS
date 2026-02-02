import os
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.metrics import matthews_corrcoef

# Import library components
from library.config import Config
from library.utils import seed_everything, compute_mcc
from library.models import SPIRVNet, FocalLoss
from library.train import run_training
from library.inference import run_inference


def create_subset_metadata(n_samples=2000):
    """
    Creates small subsets of the metadata files to speed up the demonstration.
    """
    print(f"\n[Demo] Creating metadata subsets with {n_samples} samples...")

    # Define paths for mini metadata
    mini_train_path = os.path.join(Config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(Config.WORKING_DIR, "mini_val.csv")
    mini_test_path = os.path.join(Config.WORKING_DIR, "mini_test.csv")

    # Load original metadata
    df_train = pd.read_csv(Config.METADATA_TRAIN)
    df_val = pd.read_csv(Config.METADATA_VAL)
    df_test = pd.read_csv(Config.METADATA_TEST)

    # Sample and save
    # Use a fixed random state for reproducibility
    df_train.sample(n=min(len(df_train), n_samples), random_state=42).to_csv(
        mini_train_path, index=False
    )
    df_val.sample(n=min(len(df_val), n_samples // 2), random_state=42).to_csv(
        mini_val_path, index=False
    )
    df_test.sample(n=min(len(df_test), n_samples // 2), random_state=42).to_csv(
        mini_test_path, index=False
    )

    return mini_train_path, mini_val_path, mini_test_path


def verify_model_components():
    """
    Unit tests for Model and Loss function.
    """
    print("\n[Demo] Verifying model components...")

    # 1. Verify FocalLoss
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    logits = torch.randn(10, 1, requires_grad=True)
    targets = torch.randint(0, 2, (10, 1)).float()

    loss = loss_fn(logits, targets)
    assert loss.dim() == 0, "FocalLoss should return a scalar"
    assert not torch.isnan(loss), "FocalLoss returned NaN"

    # Backward pass check
    loss.backward()
    assert logits.grad is not None, "Gradients not computed for FocalLoss"

    # 2. Verify SPIRVNet Architecture
    # Dimensions based on config defaults
    kin_dim = 50  # Arbitrary for test
    vis_dim = 10  # Arbitrary for test
    model = SPIRVNet(input_dim_kin=kin_dim, input_dim_vis=vis_dim)
    model.eval()

    dummy_kin = torch.randn(32, kin_dim)
    dummy_vis = torch.randn(32, vis_dim)

    output = model(dummy_kin, dummy_vis)

    assert output.shape == (
        32,
        1,
    ), f"Model output shape mismatch. Expected (32, 1), got {output.shape}"
    print("Model and Loss verification passed.")


def main():
    # 1. Setup Environment
    seed_everything(42)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Prepare Data Subsets
    mini_train, mini_val, mini_test = create_subset_metadata(n_samples=2000)

    # 3. Override Configuration for Speed
    print("\n[Demo] Overriding Config for fast execution...")
    Config.METADATA_TRAIN = mini_train
    Config.METADATA_VAL = mini_val
    Config.METADATA_TEST = mini_test

    # Point caches to new locations to force re-computation on new subsets
    Config.CACHE_TRAIN_FEATURES = os.path.join(
        Config.WORKING_DIR, "mini_train_features.parquet"
    )
    Config.CACHE_VAL_FEATURES = os.path.join(
        Config.WORKING_DIR, "mini_val_features.parquet"
    )
    Config.CACHE_TEST_FEATURES = os.path.join(
        Config.WORKING_DIR, "mini_test_features.parquet"
    )

    # Reduce training parameters
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 256
    Config.NUM_WORKERS = 2  # Reduce overhead for small data

    # 4. Verify Components
    verify_model_components()

    # 5. Run Training Pipeline
    print("\n[Demo] Starting Training Pipeline...")
    # load_cached_data=False ensures we process the new mini datasets
    run_training(epochs=Config.EPOCHS, load_cached_data=False)

    # Verify training artifacts
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    assert os.path.exists(Config.SCALER_PATH), "Scalers were not saved."

    # 6. Run Inference Pipeline
    print("\n[Demo] Starting Inference Pipeline...")
    run_inference(load_cached_data=False)

    # 7. Verify Submission
    print("\n[Demo] Verifying Submission...")
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    df_test_meta = pd.read_csv(Config.METADATA_TEST)

    # Check length matches test metadata
    assert len(df_sub) == len(
        df_test_meta
    ), f"Submission length mismatch. Expected {len(df_test_meta)}, got {len(df_sub)}"

    # Check columns
    assert list(df_sub.columns) == [
        "contact_id",
        "contact",
    ], f"Invalid submission columns: {df_sub.columns}"

    # Check values are binary
    unique_vals = df_sub["contact"].unique()
    assert all(
        v in [0, 1] for v in unique_vals
    ), f"Predictions must be binary. Found: {unique_vals}"

    print("\n[Demo] Success! All steps completed and verified.")


if __name__ == "__main__":
    main()
