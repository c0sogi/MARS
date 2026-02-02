import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.data_processor import DataProcessor, NFLContactDataset
from library.model import TRGCN
from library.trainer import Trainer
from library.inference import InferenceEngine


def run_demo():
    print("--- Starting Library Demo ---")

    # 1. Configure for Speed (Demo Mode)
    # We modify the Config class attributes directly to run a fast, small-scale test.
    print("Configuring for fast execution...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 500  # Process only 500 samples
    Config.EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 16  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Clean up any previous cache to ensure we process the debug subset
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 2. Data Processing
    print("\n--- Testing DataProcessor ---")
    processor = DataProcessor()

    # Process Train
    # load_cached_data=False forces regeneration using the DEBUG subset
    X_train, y_train, ids_train = processor.process_data(
        split="train", load_cached_data=False
    )
    print(f"Train Data Shape: {X_train.shape}")

    # Process Validation
    X_val, y_val, ids_val = processor.process_data(
        split="validation", load_cached_data=False
    )
    print(f"Val Data Shape: {X_val.shape}")

    # Process Test
    X_test, y_test, ids_test = processor.process_data(
        split="test", load_cached_data=False
    )
    print(f"Test Data Shape: {X_test.shape}")

    # Assertions to verify data processing logic
    assert (
        len(X_train) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} train samples, got {len(X_train)}"
    assert X_train.shape[1] == Config.WINDOW_SIZE, "Incorrect temporal window size."
    assert (
        X_train.shape[2] == Config.NUM_FEATURES_PER_TIMESTEP
    ), "Incorrect feature dimension."
    assert not np.isnan(X_train).any(), "Processed data contains NaNs."

    # 3. Dataset and DataLoader
    print("\n--- Creating DataLoaders ---")
    train_dataset = NFLContactDataset(X_train, y_train, ids_train)
    val_dataset = NFLContactDataset(X_val, y_val, ids_val)
    test_dataset = NFLContactDataset(X_test, contact_ids=ids_test)

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # 4. Model Initialization
    print("\n--- Initializing TRGCN Model ---")
    model = TRGCN(
        input_dim=Config.NUM_FEATURES_PER_TIMESTEP,
        window_size=Config.WINDOW_SIZE,
        cnn_filters=32,  # Reduced for demo
        hidden_dim=64,  # Reduced for demo
    ).to(Config.DEVICE)

    # Verify Forward Pass
    dummy_input = torch.randn(
        2, Config.WINDOW_SIZE, Config.NUM_FEATURES_PER_TIMESTEP
    ).to(Config.DEVICE)
    # Ensure the last feature (is_ground) is binary-like for the gating logic test
    dummy_input[:, :, -1] = (
        torch.randint(0, 2, (2, Config.WINDOW_SIZE)).float().to(Config.DEVICE)
    )

    with torch.no_grad():
        dummy_output = model(dummy_input)

    assert dummy_output.shape == (
        2,
        1,
    ), f"Expected output shape (2, 1), got {dummy_output.shape}"
    print("Model forward pass verified.")

    # 5. Training
    print("\n--- Starting Training Loop ---")
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    trainer = Trainer(model, optimizer, device=Config.DEVICE)

    best_mcc = trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS, patience=1)

    # Verify model saving
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model file was not saved."
    print("Training complete and model saved.")

    # 6. Inference and Submission
    print("\n--- Running Inference ---")
    inference_engine = InferenceEngine(model, device=Config.DEVICE)

    # Load best weights
    inference_engine.load_weights(best_model_path)

    # Optimize Threshold (using validation set)
    # Using a larger step for speed in this demo
    optimal_threshold = inference_engine.optimize_threshold(val_loader, step=0.1)
    assert 0.0 < optimal_threshold < 1.0, "Optimal threshold out of expected range."

    # Generate Submission
    submission_path = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")
    inference_engine.predict_test(
        test_loader, threshold=optimal_threshold, output_path=submission_path
    )

    # Verify Submission File
    assert os.path.exists(submission_path), "Submission file not created."
    df_sub = pd.read_csv(submission_path)

    assert (
        "contact_id" in df_sub.columns and "contact" in df_sub.columns
    ), "Submission missing required columns."
    assert len(df_sub) == len(
        ids_test
    ), f"Submission row count mismatch. Expected {len(ids_test)}, got {len(df_sub)}."
    assert (
        df_sub["contact"].isin([0, 1]).all()
    ), "Submission contains non-binary predictions."

    print("\n--- Demo Completed Successfully ---")
    print(f"Submission generated at: {submission_path}")
    print(f"Rows in submission: {len(df_sub)}")


if __name__ == "__main__":
    run_demo()
