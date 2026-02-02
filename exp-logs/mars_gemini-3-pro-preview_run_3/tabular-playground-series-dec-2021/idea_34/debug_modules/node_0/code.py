import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import library components
from library.config import Config
from library.utils import seed_everything, save_submission
from library.data_loader import get_dataloaders, process_data
from library.model import TriBranchWDCNet
from library.train import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_environment():
    """
    Overrides Config settings to use a temporary directory and small hyperparameters
    for a fast demonstration.
    """
    print("Setting up demo environment...")

    # Define demo paths
    demo_dir = os.path.join(Config.WORKING_DIR, "demo_execution")
    os.makedirs(demo_dir, exist_ok=True)

    # Override Global Config
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    Config.MODEL_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Override Data Paths (points to where we will save dummy data)
    Config.TRAIN_PATH = os.path.join(demo_dir, "train.parquet")
    Config.VAL_PATH = os.path.join(demo_dir, "val.parquet")
    Config.TEST_PATH = os.path.join(demo_dir, "test.parquet")

    # Override Processed Paths to force re-processing for this demo
    Config.TRAIN_PROCESSED_PATH = os.path.join(
        Config.CACHE_DIR, "train_processed.parquet"
    )
    Config.VAL_PROCESSED_PATH = os.path.join(Config.CACHE_DIR, "val_processed.parquet")
    Config.TEST_PROCESSED_PATH = os.path.join(
        Config.CACHE_DIR, "test_processed.parquet"
    )

    # Override Hyperparameters for Speed
    Config.BATCH_SIZE = 16
    Config.EPOCHS = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data
    Config.HIDDEN_DIM = 32  # Smaller model for demo
    Config.NUM_RESNET_BLOCKS = 1
    Config.NUM_CROSS_LAYERS = 1

    # Re-run setup to create new dirs
    Config.setup()

    return demo_dir


def generate_dummy_data(num_samples=100):
    """
    Generates a synthetic dataframe matching the Forest Cover Type schema
    and saves it to the configured paths.
    """
    print(f"Generating {num_samples} synthetic samples...")

    # 1. Create Feature Columns
    data = {
        Config.ID_COL: np.arange(num_samples),
        Config.TARGET_COL: np.random.randint(1, 8, size=num_samples),  # Classes 1-7
        "Elevation": np.random.normal(3000, 100, num_samples),
        "Aspect": np.random.uniform(0, 360, num_samples),
        "Slope": np.random.uniform(0, 90, num_samples),
        "Horizontal_Distance_To_Hydrology": np.random.uniform(0, 500, num_samples),
        "Vertical_Distance_To_Hydrology": np.random.uniform(-100, 100, num_samples),
        "Horizontal_Distance_To_Roadways": np.random.uniform(0, 5000, num_samples),
        "Hillshade_9am": np.random.uniform(0, 255, num_samples),
        "Hillshade_Noon": np.random.uniform(0, 255, num_samples),
        "Hillshade_3pm": np.random.uniform(0, 255, num_samples),
        "Horizontal_Distance_To_Fire_Points": np.random.uniform(0, 5000, num_samples),
    }

    # Add Wilderness Areas (4 binary columns)
    for i in range(1, 5):
        data[f"Wilderness_Area{i}"] = np.random.randint(0, 2, size=num_samples)

    # Add Soil Types (40 binary columns)
    for i in range(1, 41):
        data[f"Soil_Type{i}"] = np.random.randint(0, 2, size=num_samples)

    df = pd.DataFrame(data)

    # Split into Train/Val/Test
    # Train: 60%, Val: 20%, Test: 20%
    n_train = int(num_samples * 0.6)
    n_val = int(num_samples * 0.2)

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()

    # Test set shouldn't have target
    test_df = test_df.drop(columns=[Config.TARGET_COL])

    # Save to parquet
    train_df.to_parquet(Config.TRAIN_PATH, index=False)
    val_df.to_parquet(Config.VAL_PATH, index=False)
    test_df.to_parquet(Config.TEST_PATH, index=False)

    print("Synthetic data saved.")


def test_data_pipeline():
    """
    Demonstrates and validates the data loading and processing pipeline.
    """
    print("\n=== Testing Data Pipeline ===")

    # 1. Load Data
    # load_cached_data=False forces processing from the raw parquet files we just made
    train_loader, val_loader, test_loader, input_dim = get_dataloaders(
        load_cached_data=False
    )

    # 2. Verify Dimensions
    print(f"Detected Input Dimension: {input_dim}")

    # Check Train Batch
    features, targets, ids = next(iter(train_loader))
    print(
        f"Train Batch Shapes - X: {features.shape}, y: {targets.shape}, Id: {ids.shape}"
    )

    assert features.shape[0] == Config.BATCH_SIZE
    assert features.shape[1] == input_dim
    assert targets.shape[0] == Config.BATCH_SIZE
    # Targets should be 0-6 (internal) for 1-7 (external)
    assert targets.min() >= 0 and targets.max() <= 6

    # Check Test Batch
    t_features, t_targets, t_ids = next(iter(test_loader))
    # Test targets are dummy zeros
    assert torch.all(t_targets == 0)

    print("Data Pipeline Verified.")
    return train_loader, val_loader, test_loader, input_dim


def test_model_architecture(input_dim):
    """
    Demonstrates model instantiation and forward pass.
    """
    print("\n=== Testing Model Architecture ===")

    device = torch.device("cpu")  # Use CPU for simple demo
    model = TriBranchWDCNet(input_dim=input_dim, num_classes=7)
    model.to(device)

    # Create dummy input
    dummy_input = torch.randn(Config.BATCH_SIZE, input_dim).to(device)

    # Forward pass
    output = model(dummy_input)
    print(f"Model Output Shape: {output.shape}")

    assert output.shape == (Config.BATCH_SIZE, 7)
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    print("Model Architecture Verified.")
    return model


def test_training_loop(model, train_loader, val_loader):
    """
    Demonstrates the training process using the Trainer class.
    """
    print("\n=== Testing Training Loop ===")

    device = torch.device(Config.DEVICE)
    model.to(device)

    trainer = Trainer(model, train_loader, val_loader, device)

    # Run fit (Config.EPOCHS is set to 2)
    trainer.fit(epochs=Config.EPOCHS)

    # Check if model file was created
    assert os.path.exists(Config.MODEL_PATH), "Model file was not saved."
    print(f"Model successfully saved to {Config.MODEL_PATH}")

    # Check if best accuracy was recorded
    print(f"Best Validation Accuracy: {trainer.best_val_acc}")
    assert trainer.best_val_acc >= 0.0

    print("Training Loop Verified.")
    return trainer


def test_inference_and_submission(model, test_loader):
    """
    Demonstrates inference and submission file generation.
    """
    print("\n=== Testing Inference and Submission ===")

    device = torch.device(Config.DEVICE)
    model.eval()

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for data, _, ids in test_loader:
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)

            # Convert back to 1-based indexing for submission
            predicted = predicted + 1

            all_preds.extend(predicted.cpu().numpy())
            all_ids.extend(ids.cpu().numpy())

    # Save submission
    save_submission(all_preds, all_ids)

    # Verify File
    assert os.path.exists(Config.SUBMISSION_FILE)
    df_sub = pd.read_csv(Config.SUBMISSION_FILE)

    print("Submission File Head:")
    print(df_sub.head())

    assert list(df_sub.columns) == [Config.ID_COL, Config.TARGET_COL]
    assert len(df_sub) == len(all_ids)
    assert df_sub[Config.TARGET_COL].min() >= 1
    assert df_sub[Config.TARGET_COL].max() <= 7

    print("Inference and Submission Verified.")


def main():
    # 1. Reproducibility
    seed_everything(Config.SEED)

    # 2. Setup Environment
    demo_dir = setup_demo_environment()

    try:
        # 3. Generate Data
        generate_dummy_data(num_samples=200)

        # 4. Data Pipeline
        train_loader, val_loader, test_loader, input_dim = test_data_pipeline()

        # 5. Model
        model = test_model_architecture(input_dim)

        # 6. Training
        trainer = test_training_loop(model, train_loader, val_loader)

        # 7. Inference
        # Reload best model weights (Trainer does this automatically at end of fit, but good to verify)
        model.load_state_dict(torch.load(Config.MODEL_PATH))
        test_inference_and_submission(model, test_loader)

        print("\nAll demonstrations completed successfully.")

    finally:
        # Cleanup (Optional: remove demo directory to keep workspace clean)
        # shutil.rmtree(demo_dir)
        pass


if __name__ == "__main__":
    main()
