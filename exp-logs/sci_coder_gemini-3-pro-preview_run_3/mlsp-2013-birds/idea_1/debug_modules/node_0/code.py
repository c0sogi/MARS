import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import provided library modules
from library import config, dataset, model, trainer


def run_demo():
    print("Initializing demonstration...")

    # ==========================================
    # 0. Configuration & Setup
    # ==========================================
    # Set seeds for reproducibility
    dataset.set_seed(config.RANDOM_SEED)

    # Override configuration for speed in this demo
    config.NUM_EPOCHS = 2
    config.BATCH_SIZE = 16
    config.HIDDEN_DIM = 32  # Smaller model for speed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================================
    # 1. Data Loading Demonstration
    # ==========================================
    print("\n[Step 1] Testing Data Loading...")

    # Load dataloaders
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        load_cached_data=False
    )

    # Verify Train Loader
    train_batch = next(iter(train_loader))
    features, labels = train_batch

    print(f"Train batch features shape: {features.shape}")
    print(f"Train batch labels shape: {labels.shape}")

    # Assertions for data integrity
    assert (
        features.shape[1] == config.INPUT_DIM
    ), f"Expected input dim {config.INPUT_DIM}, got {features.shape[1]}"
    assert (
        labels.shape[1] == config.NUM_CLASSES
    ), f"Expected num classes {config.NUM_CLASSES}, got {labels.shape[1]}"
    assert features.dtype == torch.float32, "Features should be float32"
    assert labels.dtype == torch.float32, "Labels should be float32"

    # Verify Test Loader (returns features and rec_ids)
    test_batch = next(iter(test_loader))
    t_features, t_rec_ids = test_batch
    assert t_features.shape[1] == config.INPUT_DIM, "Test features dim mismatch"
    assert t_rec_ids.dtype == torch.long, "Test rec_ids should be long/int"

    print("Data loading verification passed.")

    # ==========================================
    # 2. Model Instantiation & Forward Pass
    # ==========================================
    print("\n[Step 2] Testing Model Architecture...")

    # Instantiate model
    net = model.ShallowMLP(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE,
    ).to(device)

    # Test forward pass with a dummy batch
    dummy_input = torch.randn(config.BATCH_SIZE, config.INPUT_DIM).to(device)
    dummy_output = net(dummy_input)

    print(f"Model output shape: {dummy_output.shape}")

    # Assertions
    assert dummy_output.shape == (
        config.BATCH_SIZE,
        config.NUM_CLASSES,
    ), "Model output shape mismatch"
    assert not torch.isnan(dummy_output).any(), "Model output contains NaNs"

    print("Model architecture verification passed.")

    # ==========================================
    # 3. Training Loop Demonstration
    # ==========================================
    print("\n[Step 3] Testing Training Loop...")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    # Run a few epochs
    for epoch in range(config.NUM_EPOCHS):
        train_loss = trainer.train_epoch(
            net, train_loader, criterion, optimizer, device
        )
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}")

        # Assertions
        assert isinstance(train_loss, float), "Train loss should be a float"
        assert train_loss >= 0, "Train loss should be non-negative"
        assert not np.isnan(train_loss), "Train loss is NaN"

    print("Training loop verification passed.")

    # ==========================================
    # 4. Evaluation Demonstration
    # ==========================================
    print("\n[Step 4] Testing Evaluation...")

    val_loss, val_auc = trainer.evaluate(net, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")

    # Assertions
    assert isinstance(val_loss, float), "Val loss should be a float"
    assert isinstance(val_auc, float), "Val AUC should be a float"
    # AUC can be 0.0 if predictions are poor or classes absent, but usually between 0 and 1
    assert 0.0 <= val_auc <= 1.0, "AUC should be between 0 and 1"

    print("Evaluation verification passed.")

    # ==========================================
    # 5. Prediction & Submission Demonstration
    # ==========================================
    print("\n[Step 5] Testing Prediction and Submission Generation...")

    # Generate predictions
    test_predictions = trainer.predict(net, test_loader, device)

    # Check predictions dictionary
    assert isinstance(test_predictions, dict), "Predictions should be a dictionary"
    assert len(test_predictions) > 0, "Predictions dictionary is empty"

    # Check a single prediction entry
    first_rec_id = list(test_predictions.keys())[0]
    first_probs = test_predictions[first_rec_id]
    assert (
        len(first_probs) == config.NUM_CLASSES
    ), f"Prediction vector length mismatch. Expected {config.NUM_CLASSES}, got {len(first_probs)}"
    assert np.all(
        (first_probs >= 0) & (first_probs <= 1)
    ), "Probabilities should be between 0 and 1"

    # Save submission
    # Ensure submission dir exists (handled by config, but good to double check in logic flow)
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

    trainer.save_submission(test_predictions, config.SUBMISSION_PATH)

    # Verify file creation
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created"

    # Verify file content format
    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission file shape: {df_sub.shape}")
    print(f"Submission columns: {df_sub.columns.tolist()}")

    assert (
        "Id" in df_sub.columns and "Probability" in df_sub.columns
    ), "Submission file missing required columns"

    # Verify row count: 64 test samples * 19 classes = 1216 rows
    expected_rows = 64 * 19
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(df_sub)}"

    print("Prediction and submission verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
