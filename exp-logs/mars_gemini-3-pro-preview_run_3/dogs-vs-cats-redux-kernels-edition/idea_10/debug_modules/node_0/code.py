import os
import pandas as pd
import torch
import shutil
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_dataloaders
from library.model_factory import create_model
from library.trainer import run_swa_training
from library.inference import run_inference


def create_mini_metadata(source_path, dest_path, n_samples=20):
    """
    Creates a smaller version of the metadata CSV for rapid testing.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source metadata not found: {source_path}")

    df = pd.read_csv(source_path)
    # Sample minimal amount
    df_mini = df.head(n_samples).copy()
    df_mini.to_csv(dest_path, index=False)
    return len(df_mini)


def main():
    print("Starting execution of the demonstration script...")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed and Testing
    # -------------------------------------------------------------------------
    # Define a separate working directory for this demo
    DEMO_WORKING_DIR = "./working/demo_run"
    if os.path.exists(DEMO_WORKING_DIR):
        shutil.rmtree(DEMO_WORKING_DIR)
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

    # Override Config paths and settings
    Config.WORKING_DIR = DEMO_WORKING_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_WORKING_DIR, "submission.csv")

    # Use a lightweight model for the demo to save memory and time
    # We keep the key 'resnet50' so downstream logic works, but use 'resnet18' implementation
    Config.MODEL_SPECS = {
        "resnet50": {
            "timm_name": "resnet18",
            "img_size": 224,
            "batch_size": 4,  # Small batch size for the mini dataset
        }
    }

    # Training settings
    Config.NUM_EPOCHS = 1
    Config.USE_SWA = False  # Disable SWA to save time
    Config.DEBUG = False  # We handle sampling manually via mini-CSVs

    # -------------------------------------------------------------------------
    # 2. Prepare Mini Datasets
    # -------------------------------------------------------------------------
    print("Preparing mini datasets...")
    mini_train_path = os.path.join(DEMO_WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(DEMO_WORKING_DIR, "mini_val.csv")
    mini_test_path = os.path.join(DEMO_WORKING_DIR, "mini_test.csv")

    n_train = create_mini_metadata(Config.TRAIN_METADATA, mini_train_path, n_samples=20)
    n_val = create_mini_metadata(Config.VAL_METADATA, mini_val_path, n_samples=10)
    n_test = create_mini_metadata(Config.TEST_METADATA, mini_test_path, n_samples=10)

    # Point Config to these new files
    Config.TRAIN_METADATA = mini_train_path
    Config.VAL_METADATA = mini_val_path
    Config.TEST_METADATA = mini_test_path

    print(f"Mini datasets created: Train={n_train}, Val={n_val}, Test={n_test}")

    # -------------------------------------------------------------------------
    # 3. Reproducibility
    # -------------------------------------------------------------------------
    seed_everything(Config.SEED)

    # -------------------------------------------------------------------------
    # 4. Validate Data Loading
    # -------------------------------------------------------------------------
    print("Validating DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        "resnet50", load_cached_data=False
    )

    # Check Train Loader
    images, labels = next(iter(train_loader))
    assert images.shape == (
        4,
        3,
        224,
        224,
    ), f"Unexpected train image shape: {images.shape}"
    assert labels.shape == (4,), f"Unexpected train label shape: {labels.shape}"
    print("DataLoaders initialized successfully.")

    # -------------------------------------------------------------------------
    # 5. Validate Model Creation
    # -------------------------------------------------------------------------
    print("Validating Model Creation...")
    model = create_model(
        "resnet50", pretrained=False
    )  # False to avoid download overhead if any
    assert isinstance(model, torch.nn.Module), "Model is not a torch.nn.Module"

    # Check if MultiSampleDropoutHead was applied (Config default is True)
    if Config.USE_MULTI_SAMPLE_DROPOUT:
        # For ResNet, it replaces model.fc
        assert (
            model.fc.__class__.__name__ == "MultiSampleDropoutHead"
        ), "Head replacement failed"

    print("Model created successfully.")

    # -------------------------------------------------------------------------
    # 6. Validate Training Pipeline
    # -------------------------------------------------------------------------
    print("Running Training (1 Epoch)...")
    # run_swa_training handles the loop, saving best model, etc.
    trained_model = run_swa_training(model, train_loader, val_loader, "resnet50")

    # Verify output file exists
    best_model_path = os.path.join(Config.WORKING_DIR, "resnet50_best.pth")
    assert os.path.exists(best_model_path), "Best model file was not saved."
    print("Training completed successfully.")

    # -------------------------------------------------------------------------
    # 7. Validate Inference Pipeline
    # -------------------------------------------------------------------------
    print("Running Inference...")
    # run_inference iterates over Config.MODEL_SPECS, loads weights, predicts, and saves CSV
    run_inference()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(submission_df)} rows.")

    # Checks
    assert (
        len(submission_df) == n_test
    ), f"Submission row count mismatch. Expected {n_test}, got {len(submission_df)}"
    assert (
        "id" in submission_df.columns and "label" in submission_df.columns
    ), "Missing columns in submission"

    # Check probability range
    preds = submission_df["label"].values
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions out of probability range [0, 1]"

    print("\nAll demonstration steps completed successfully!")
    print(f"Final submission saved to: {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
