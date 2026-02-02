import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library files
from library.config import Config
from library.data_utils import process_data, get_dataloaders, set_seed
from library.model import (
    VectorCrossLayer,
    VectorDCN,
    ResNetBlock,
    ParallelVectorDCNResNet,
    generate_submission,
)
from library.train_utils import run_training


def create_mock_data(num_rows=1000):
    """
    Creates a small mock dataset matching the schema required by Config
    and saves it to parquet files in a temporary working directory.
    """
    print("Creating mock dataset for demonstration...")

    # 1. Define schema based on Config
    cont_cols = Config.CONTINUOUS_COLS
    # Wilderness_Area1 to 4
    wild_cols = [f"Wilderness_Area{i}" for i in range(1, 5)]
    # Soil_Type1 to 40
    soil_cols = [f"Soil_Type{i}" for i in range(1, 41)]

    all_feature_cols = cont_cols + wild_cols + soil_cols

    # 2. Generate random data
    data = {}

    # ID Column
    data[Config.ID_COL] = np.arange(num_rows)

    # Continuous columns: Random floats
    for col in cont_cols:
        data[col] = np.random.randn(num_rows).astype(np.float32) * 100
        # Ensure positive values for distances/elevations where logical to avoid sqrt errors
        # (though the engineering handles squares, so negatives are fine mathematically)
        if "Distance" in col or "Elevation" in col:
            data[col] = np.abs(data[col])

    # Binary columns: Random 0 or 1
    for col in wild_cols + soil_cols:
        data[col] = np.random.randint(0, 2, size=num_rows)

    # Target Column: Integers 1-7
    data[Config.TARGET_COL] = np.random.randint(1, 8, size=num_rows)

    df = pd.DataFrame(data)

    # 3. Split into Train, Val, Test
    # Train: 60%, Val: 20%, Test: 20%
    n_train = int(num_rows * 0.6)
    n_val = int(num_rows * 0.2)

    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train : n_train + n_val].copy()
    df_test = df.iloc[n_train + n_val :].copy()

    # Test set shouldn't have the target (usually), but the loader handles it.
    # The provided dataset class checks if y is None.
    # The metadata/test.parquet usually has IDs and features.
    df_test = df_test.drop(columns=[Config.TARGET_COL])

    # 4. Save to disk
    mock_dir = "./working/demo_data"
    os.makedirs(mock_dir, exist_ok=True)

    train_path = os.path.join(mock_dir, "train.parquet")
    val_path = os.path.join(mock_dir, "val.parquet")
    test_path = os.path.join(mock_dir, "test.parquet")

    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)
    df_test.to_parquet(test_path, index=False)

    print(f"Mock data saved to {mock_dir}")
    return train_path, val_path, test_path


def setup_demo_config(train_path, val_path, test_path):
    """
    Overrides Config paths and hyperparameters for the demo.
    """
    print("Configuring experiment settings for speed...")

    # Override paths
    Config.TRAIN_PATH = train_path
    Config.VAL_PATH = val_path
    Config.TEST_PATH = test_path

    # Use a separate working directory for the demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_execution/submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Override Hyperparameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 32
    Config.HIDDEN_DIM = 64  # Smaller model
    Config.RESNET_BLOCKS = 1

    # Ensure directories exist
    Config.setup()


def verify_data_pipeline():
    """
    Verifies data loading, feature engineering, and processing.
    """
    print("\n=== Verifying Data Pipeline ===")

    # Force processing from scratch (ignore cache if any)
    train_X, train_y, val_X, val_y, test_X, test_ids = process_data(
        load_cached_data=False
    )

    # Assertions
    print(f"Train X shape: {train_X.shape}")
    print(f"Train y shape: {train_y.shape}")

    assert len(train_X) == len(train_y), "Mismatch in training features and labels"
    assert not np.isnan(train_X).any(), "NaN values found in processed training data"

    # Check if feature engineering added columns
    # Base continuous (10) + Engineered (Aspect_Sin, Aspect_Cos, Hydro_Euclidean, Hydro_Elevation, Mean_Amenities = 5)
    # + Binary (4 + 40 = 44) => Total approx 59 columns expected
    expected_min_cols = 10 + 44
    assert (
        train_X.shape[1] >= expected_min_cols
    ), f"Expected at least {expected_min_cols} features, got {train_X.shape[1]}"

    # Verify DataLoaders
    train_loader, val_loader, test_loader, input_dim, ids = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    batch_X, batch_y = next(iter(train_loader))
    print(f"DataLoader Batch X: {batch_X.shape}, Batch y: {batch_y.shape}")

    assert batch_X.shape[0] == Config.BATCH_SIZE, "DataLoader batch size mismatch"
    assert batch_X.shape[1] == input_dim, "DataLoader feature dimension mismatch"

    return train_loader, val_loader, test_loader, input_dim, ids


def verify_model_components(input_dim):
    """
    Verifies individual model layers and the full architecture.
    """
    print("\n=== Verifying Model Architecture ===")

    batch_size = 4
    dummy_input = torch.randn(batch_size, input_dim)

    # 1. Verify VectorCrossLayer
    print("Testing VectorCrossLayer...")
    vcl = VectorCrossLayer(input_dim)
    out_vcl = vcl(dummy_input, dummy_input)
    assert out_vcl.shape == dummy_input.shape, "VectorCrossLayer output shape mismatch"

    # 2. Verify VectorDCN
    print("Testing VectorDCN...")
    dcn = VectorDCN(input_dim, num_layers=2)
    out_dcn = dcn(dummy_input)
    assert out_dcn.shape == dummy_input.shape, "VectorDCN output shape mismatch"

    # 3. Verify ResNetBlock
    print("Testing ResNetBlock...")
    hidden_dim = Config.HIDDEN_DIM
    res_block = ResNetBlock(hidden_dim, dropout_rate=0.1)
    dummy_hidden = torch.randn(batch_size, hidden_dim)
    out_res = res_block(dummy_hidden)
    assert out_res.shape == dummy_hidden.shape, "ResNetBlock output shape mismatch"

    # 4. Verify Full Model
    print("Testing ParallelVectorDCNResNet...")
    model = ParallelVectorDCNResNet(
        input_dim=input_dim,
        num_classes=Config.NUM_CLASSES,
        hidden_dim=Config.HIDDEN_DIM,
        resnet_blocks=Config.RESNET_BLOCKS,
    )

    # Forward pass
    logits = model(dummy_input)
    print(f"Model Output Shape: {logits.shape}")

    # Expected output: (batch_size, num_classes)
    assert logits.shape == (
        batch_size,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected {(batch_size, Config.NUM_CLASSES)}, got {logits.shape}"

    return model


def verify_training_loop(model, train_loader, val_loader):
    """
    Verifies the training process using the provided utility.
    """
    print("\n=== Verifying Training Loop ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model = model.to(device)

    # Run training for limited epochs (configured in setup_demo_config)
    trained_model = run_training(
        model,
        train_loader,
        val_loader,
        device,
        epochs=Config.EPOCHS,
        learning_rate=1e-3,
        patience=2,
    )

    # Check if best model was saved
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model file was not saved"
    print("Training completed and best model saved.")

    return trained_model, device


def verify_submission(model, test_loader, test_ids, device):
    """
    Verifies submission file generation.
    """
    print("\n=== Verifying Submission Generation ===")

    generate_submission(model, test_loader, test_ids, device)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")
    print(f"Submission columns: {df_sub.columns.tolist()}")

    assert Config.ID_COL in df_sub.columns, "Id column missing in submission"
    assert Config.TARGET_COL in df_sub.columns, "Target column missing in submission"
    assert len(df_sub) == len(test_ids), "Submission row count mismatch"

    # Check values are valid class labels (1-7)
    preds = df_sub[Config.TARGET_COL]
    assert preds.min() >= 1 and preds.max() <= 7, "Predictions out of valid range (1-7)"

    print("Submission verification successful.")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    # 1. Create Mock Data
    train_path, val_path, test_path = create_mock_data(num_rows=1000)

    # 2. Configure Environment
    setup_demo_config(train_path, val_path, test_path)

    # 3. Verify Data Pipeline
    train_loader, val_loader, test_loader, input_dim, test_ids = verify_data_pipeline()

    # 4. Verify Model Architecture
    model = verify_model_components(input_dim)

    # 5. Verify Training Loop
    trained_model, device = verify_training_loop(model, train_loader, val_loader)

    # 6. Verify Submission
    verify_submission(trained_model, test_loader, test_ids, device)

    print("\nAll demonstrations and verifications passed successfully!")
