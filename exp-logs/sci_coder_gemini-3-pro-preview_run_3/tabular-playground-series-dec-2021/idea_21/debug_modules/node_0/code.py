import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import library components
import library.config as config
import library.data as data_lib
import library.model as model_lib
import library.train as train_lib
import library.utils as utils_lib


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup
    utils_lib.seed_everything(config.SEED)
    device = utils_lib.get_device()
    print(f"Device: {device}")

    # ==========================================
    # 2. Model Verification
    # ==========================================
    print("\n[1/4] Verifying Model Architecture...")

    batch_size = 32
    # Create dummy input: (Batch, Input_Dim)
    # Input dim is sum of continuous and binary features
    dummy_input = torch.randn(batch_size, config.INPUT_DIM).to(device)

    model = model_lib.DeepVectorDCNResNet(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
    ).to(device)

    # Forward pass
    logits = model(dummy_input)

    # Assertions
    assert logits.shape == (
        batch_size,
        config.NUM_CLASSES,
    ), f"Output shape mismatch. Expected {(batch_size, config.NUM_CLASSES)}, got {logits.shape}"
    print("  - Forward pass successful. Output shape verified.")

    # Backward pass check
    target = torch.randint(0, config.NUM_CLASSES, (batch_size,)).to(device)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, target)
    loss.backward()

    # Check if gradients are populated
    assert model.classifier.weight.grad is not None, "Gradient computation failed."
    print("  - Backward pass successful. Gradients computed.")

    # ==========================================
    # 3. Feature Engineering Verification
    # ==========================================
    print("\n[2/4] Verifying Feature Engineering Logic...")

    # Create a minimal dataframe with necessary columns for engineering
    df_dummy = pd.DataFrame(
        {
            "Aspect": [0, 90, 180, 270],
            "Horizontal_Distance_To_Hydrol": [3, 4, 0, 10],
            "Vertical_Distance_To_Hydrolog": [4, 3, 0, 10],
            "Elevation": [100, 200, 300, 400],
            "Horizontal_Distance_To_Roadwa": [10, 10, 10, 10],
            "Horizontal_Distance_To_Fire_P": [20, 20, 20, 20],
        }
    )

    # Apply engineering
    df_processed = data_lib.feature_engineering(df_dummy)

    # Check specific engineered features
    # 1. Aspect_Sin: sin(0) = 0, sin(90) = 1
    assert config.FEAT_ASPECT_SIN in df_processed.columns
    assert np.isclose(
        df_processed.loc[1, config.FEAT_ASPECT_SIN], 1.0
    ), "Aspect_Sin calculation incorrect"

    # 2. Euclidean_Distance_To_Hydrology: sqrt(3^2 + 4^2) = 5
    assert config.FEAT_EUCLIDEAN_HYDRO in df_processed.columns
    assert np.isclose(
        df_processed.loc[0, config.FEAT_EUCLIDEAN_HYDRO], 5.0
    ), "Euclidean distance calculation incorrect"

    print("  - Feature engineering logic verified.")

    # ==========================================
    # 4. Pipeline Integration (Mini-Dataset)
    # ==========================================
    print("\n[3/4] Setting up Mini-Dataset for Pipeline Integration...")

    # Define paths for mini dataset
    working_dir = "./working/demo_pipeline"
    os.makedirs(working_dir, exist_ok=True)

    mini_train_path = os.path.join(working_dir, "mini_train.parquet")
    mini_val_path = os.path.join(working_dir, "mini_val.parquet")
    mini_test_path = os.path.join(working_dir, "mini_test.parquet")
    mini_cache_dir = os.path.join(working_dir, "cache")

    # Generate synthetic data matching the schema
    num_samples = 200

    # Columns: Id, Cover_Type, Raw Continuous, Raw Binary
    cols = [config.ID_COL] + config.RAW_CONTINUOUS_FEATURES + config.RAW_BINARY_FEATURES

    # Create random data
    data_dict = {config.ID_COL: np.arange(num_samples)}
    for col in config.RAW_CONTINUOUS_FEATURES:
        data_dict[col] = np.random.randn(num_samples) * 100 + 2000  # Random continuous
    for col in config.RAW_BINARY_FEATURES:
        data_dict[col] = np.random.randint(0, 2, num_samples)  # Random binary

    # Add target for train/val
    data_dict[config.TARGET_COL] = np.random.choice(config.CLASS_LABELS, num_samples)

    df_mini = pd.DataFrame(data_dict)

    # Split and save
    df_mini.iloc[:100].to_parquet(mini_train_path)
    df_mini.iloc[100:150].to_parquet(mini_val_path)
    # Test set doesn't have target
    df_mini.iloc[150:].drop(columns=[config.TARGET_COL]).to_parquet(mini_test_path)

    print(f"  - Mini-dataset created at {working_dir}")

    # --- Monkey-Patching Library Paths ---
    # We must patch the variables in the loaded modules to point to our mini data
    data_lib.TRAIN_DATA_PATH = mini_train_path
    data_lib.VAL_DATA_PATH = mini_val_path
    data_lib.TEST_DATA_PATH = mini_test_path
    data_lib.CACHE_DIR = mini_cache_dir

    # Also patch config just in case it's referenced elsewhere (though data_lib imports specific vars)
    config.TRAIN_DATA_PATH = mini_train_path
    config.VAL_DATA_PATH = mini_val_path
    config.TEST_DATA_PATH = mini_test_path
    config.CACHE_DIR = mini_cache_dir

    # Ensure cache dir exists
    os.makedirs(mini_cache_dir, exist_ok=True)

    print("\n[4/4] Running Training Pipeline on Mini-Dataset...")

    # 1. Get DataLoaders (this triggers process_data)
    # Set load_cached_data=False to ensure we process our new mini files
    train_loader, val_loader, test_loader, test_ids = data_lib.get_dataloaders(
        batch_size=16,
        num_workers=0,  # Avoid multiprocessing overhead for small demo
        load_cached_data=False,
    )

    assert len(train_loader) > 0, "Train loader is empty"
    print("  - DataLoaders initialized successfully.")

    # 2. Initialize Trainer
    trainer = train_lib.Trainer(model, train_loader, val_loader, device)

    # 3. Fit (Run for 1 epoch only for speed)
    print("  - Starting training (1 epoch)...")
    trainer.fit(epochs=1, patience=1)

    # 4. Predict
    print("  - Generating predictions...")
    df_submission = trainer.predict(test_loader, test_ids)

    # 5. Verify Submission
    assert (
        len(df_submission) == 50
    ), f"Submission length mismatch. Expected 50, got {len(df_submission)}"
    assert config.ID_COL in df_submission.columns
    assert config.TARGET_COL in df_submission.columns

    # Check if predictions are valid class labels
    preds = df_submission[config.TARGET_COL].unique()
    valid_labels = set(config.CLASS_LABELS)
    assert all(
        p in valid_labels for p in preds
    ), "Submission contains invalid class labels"

    print("  - Submission generated and verified.")
    print("  - Sample output:")
    print(df_submission.head())

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
