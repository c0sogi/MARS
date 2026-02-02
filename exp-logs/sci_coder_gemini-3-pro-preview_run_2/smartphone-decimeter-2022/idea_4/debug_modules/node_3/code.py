import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library import config
from library import dataset
from library import model
from library import engine
from library import utils


def run_pipeline():
    print("Initializing Pipeline...")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed
    # ---------------------------------------------------------
    # Enable debug mode to use a small subset of trips (defined in library/dataset.py)
    config.DEBUG = True

    # Reduce training epochs and patience for demonstration
    config.TRAIN_PARAMS["epochs"] = 2
    config.TRAIN_PARAMS["patience"] = 1

    # Adjust batch size and workers for the environment
    config.TRAIN_PARAMS["batch_size"] = 256
    config.TRAIN_PARAMS["num_workers"] = 2

    # Ensure reproducibility
    torch.manual_seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)

    print(f"Configuration:")
    print(f"  Debug Mode: {config.DEBUG}")
    print(f"  Epochs: {config.TRAIN_PARAMS['epochs']}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # ---------------------------------------------------------
    # 2. Data Processing (Train & Validation)
    # ---------------------------------------------------------
    print("\n--- Step 1: Processing Train/Val Data ---")

    # Process Train Data
    # load_cached_data=False forces re-processing to demonstrate the logic
    train_df = dataset.preprocess_data(
        config.TRAIN_METADATA_PATH, mode="train", load_cached_data=False
    )

    # Process Validation Data
    val_df = dataset.preprocess_data(
        config.VAL_METADATA_PATH, mode="val", load_cached_data=False
    )

    # Verify DataFrames are not empty
    assert not train_df.empty, "Processed training DataFrame is empty."
    assert not val_df.empty, "Processed validation DataFrame is empty."
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # Compute Scaler Statistics (Normalization)
    scaler_stats = dataset.get_scaler_stats(
        train_df,
        config.INPUT_FEATURES,
        config.CACHE_FILES["scaler"],
        load_cached_data=False,
    )

    # Verify scaler stats contain all features
    for feature in config.INPUT_FEATURES:
        assert feature in scaler_stats, f"Missing scaler stats for {feature}"

    # Instantiate Datasets
    # This prepares the sliding windows
    train_dataset = dataset.GNSSWindowDataset(
        train_df, config.WINDOW_SIZE, scaler_stats, mode="train"
    )

    val_dataset = dataset.GNSSWindowDataset(
        val_df, config.WINDOW_SIZE, scaler_stats, mode="val"
    )

    print(f"Train windows: {len(train_dataset)}, Val windows: {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN_PARAMS["batch_size"],
        shuffle=True,
        num_workers=config.TRAIN_PARAMS["num_workers"],
        pin_memory=config.TRAIN_PARAMS["pin_memory"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.TRAIN_PARAMS["batch_size"],
        shuffle=False,
        num_workers=config.TRAIN_PARAMS["num_workers"],
        pin_memory=config.TRAIN_PARAMS["pin_memory"],
    )

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print("\n--- Step 2: Initializing Model ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = model.ContextAware1DCNN(
        input_dim=config.MODEL_PARAMS["input_dim"],
        context_dim=config.MODEL_PARAMS["context_dim"],
        window_size=config.WINDOW_SIZE,
        conv_channels=config.MODEL_PARAMS["conv_channels"],
        kernel_size=config.MODEL_PARAMS["kernel_size"],
        fc_hidden=config.MODEL_PARAMS["fc_hidden"],
        dropout=config.MODEL_PARAMS["dropout"],
        output_dim=config.MODEL_PARAMS["output_dim"],
    ).to(device)

    # Verify model structure (simple forward pass with dummy data)
    dummy_window = torch.randn(
        2, config.MODEL_PARAMS["input_dim"], config.WINDOW_SIZE
    ).to(device)
    dummy_context = torch.randn(2, config.MODEL_PARAMS["context_dim"]).to(device)
    with torch.no_grad():
        dummy_out = net(dummy_window, dummy_context)
    assert dummy_out.shape == (
        2,
        2,
    ), f"Model output shape mismatch. Expected (2, 2), got {dummy_out.shape}"
    print("Model initialized and verified.")

    # ---------------------------------------------------------
    # 4. Training
    # ---------------------------------------------------------
    print("\n--- Step 3: Training Model ---")

    # train_model handles the training loop, validation, scheduler, and saving best model
    trained_model = engine.train_model(
        net, train_loader, val_loader, device, config.TRAIN_PARAMS
    )

    assert os.path.exists(config.CACHE_FILES["model"]), "Best model file was not saved."

    # ---------------------------------------------------------
    # 5. Inference (Test Set)
    # ---------------------------------------------------------
    print("\n--- Step 4: Generating Submission ---")

    # Process Test Data
    test_df = dataset.preprocess_data(
        config.TEST_METADATA_PATH, mode="test", load_cached_data=False
    )

    assert not test_df.empty, "Processed test DataFrame is empty."

    # Instantiate Test Dataset
    test_dataset = dataset.GNSSWindowDataset(
        test_df, config.WINDOW_SIZE, scaler_stats, mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.TRAIN_PARAMS["batch_size"],
        shuffle=False,
        num_workers=config.TRAIN_PARAMS["num_workers"],
        pin_memory=config.TRAIN_PARAMS["pin_memory"],
    )

    # Generate Submission
    # This function predicts residuals, converts back to LLA, and saves CSV
    engine.generate_submission(trained_model, test_loader, test_df, device)

    # ---------------------------------------------------------
    # 6. Final Verification
    # ---------------------------------------------------------
    print("\n--- Step 5: Final Verification ---")

    if os.path.exists(config.SUBMISSION_PATH):
        sub_df = pd.read_csv(config.SUBMISSION_PATH)
        print(f"Submission loaded. Shape: {sub_df.shape}")

        # Check required columns
        required_cols = [
            "tripId",
            "UnixTimeMillis",
            "LatitudeDegrees",
            "LongitudeDegrees",
        ]
        missing_cols = [c for c in required_cols if c not in sub_df.columns]

        if missing_cols:
            raise AssertionError(f"Submission missing columns: {missing_cols}")

        # Check for NaNs
        if sub_df.isnull().any().any():
            raise AssertionError("Submission contains NaN values.")

        # Check row count matches input test metadata (accounting for debug sampling)
        # In debug mode, test_df is sampled, so submission should match test_df length
        assert len(sub_df) == len(
            test_df
        ), f"Submission length ({len(sub_df)}) does not match processed test data length ({len(test_df)})"

        print("Verification successful. Pipeline complete.")
    else:
        raise FileNotFoundError(
            f"Submission file not found at {config.SUBMISSION_PATH}"
        )


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print(f"\nPipeline Failed: {e}")
        raise e
