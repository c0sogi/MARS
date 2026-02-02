import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Import library components
from library.config import Config
from library.data_utils import load_data, CoverTypeDataset
from library.model import ParallelDCNResNet
from library.train_eval import train_one_epoch, evaluate, predict


def setup_demo_config():
    """
    Overrides the default Config class attributes to create a lightweight
    execution environment for demonstration purposes.
    """
    # Enable Debug mode to use data subsets (10k train, 2k val/test)
    Config.DEBUG = True

    # Reduce training parameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 256

    # Set a specific working directory for this demo to avoid path conflicts
    Config.WORKING_DIR = "./working/demo_task"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Redirect cache paths to the demo directory
    Config.CACHE_TRAIN_X = os.path.join(Config.WORKING_DIR, "train_processed.npy")
    Config.CACHE_TRAIN_Y = os.path.join(Config.WORKING_DIR, "train_labels.npy")
    Config.CACHE_VAL_X = os.path.join(Config.WORKING_DIR, "val_processed.npy")
    Config.CACHE_VAL_Y = os.path.join(Config.WORKING_DIR, "val_labels.npy")
    Config.CACHE_TEST_X = os.path.join(Config.WORKING_DIR, "test_processed.npy")
    Config.CACHE_TEST_IDS = os.path.join(Config.WORKING_DIR, "test_ids.npy")

    # Redirect output paths
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure reproducibility
    Config.SEED = 999
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)


def validate_data_loading():
    """
    Demonstrates and validates the data loading and feature engineering pipeline.
    """
    print("\n=== 1. Validating Data Loading & Processing ===")

    # Force reload to trigger feature engineering logic
    train_X, train_y, val_X, val_y, test_X, test_ids = load_data(load_cached_data=False)

    # Validation: Check shapes based on DEBUG subsampling in data_utils.py
    # Debug mode slices: train[:10000], val[:2000], test[:2000]
    print(f"Train Data Shape: {train_X.shape}")
    print(f"Val Data Shape:   {val_X.shape}")
    print(f"Test Data Shape:  {test_X.shape}")

    assert train_X.shape[0] == 10000, "Debug mode training set size mismatch."
    assert val_X.shape[0] == 2000, "Debug mode validation set size mismatch."
    assert test_X.shape[0] == 2000, "Debug mode test set size mismatch."

    # Validation: Check consistency of features
    assert (
        train_X.shape[1] == val_X.shape[1] == test_X.shape[1]
    ), "Feature dimension mismatch across splits."

    # Validation: Check target values
    # Mapped classes should be between 0 and 5 (for 6 classes)
    assert (
        train_y.min() >= 0 and train_y.max() < Config.NUM_CLASSES
    ), "Target labels out of expected range."

    return train_X, train_y, val_X, val_y, test_X, test_ids


def validate_model_architecture(input_dim):
    """
    Demonstrates model instantiation and validates the forward pass.
    """
    print("\n=== 2. Validating Model Architecture ===")

    device = torch.device("cpu")  # Use CPU for simple shape check
    model = ParallelDCNResNet(input_dim=input_dim, num_classes=Config.NUM_CLASSES)
    model.to(device)
    model.eval()

    # Create a dummy batch
    batch_size = 4
    dummy_input = torch.randn(batch_size, input_dim).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Input Shape: {dummy_input.shape}")
    print(f"Model Output Shape: {output.shape}")

    # Validation: Output shape
    assert output.shape == (
        batch_size,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch."

    # Validation: Finite outputs
    assert torch.isfinite(output).all(), "Model produced NaN or Inf values."

    return model


def validate_training_loop(model, train_X, train_y, val_X, val_y):
    """
    Demonstrates the training and evaluation loop for one epoch.
    """
    print("\n=== 3. Validating Training & Evaluation Loop ===")

    device = torch.device(Config.DEVICE)
    model.to(device)

    # Create Datasets and Loaders
    train_dataset = CoverTypeDataset(train_X, train_y)
    val_dataset = CoverTypeDataset(val_X, val_y)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Setup Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Run Training Step
    print("Running training epoch...")
    loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Train Result -> Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    # Validation: Metrics range
    assert loss > 0, "Training loss should be positive."
    assert 0.0 <= acc <= 1.0, "Training accuracy out of range."

    # Run Evaluation Step
    print("Running evaluation...")
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Val Result   -> Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    # Validation: Metrics range
    assert val_loss > 0, "Validation loss should be positive."
    assert 0.0 <= val_acc <= 1.0, "Validation accuracy out of range."


def validate_inference_and_submission(model, test_X, test_ids):
    """
    Demonstrates inference on test data and submission file generation.
    """
    print("\n=== 4. Validating Inference & Submission Generation ===")

    device = torch.device(Config.DEVICE)
    test_dataset = CoverTypeDataset(test_X, None)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Run Prediction
    raw_preds = predict(model, test_loader, device)

    # Validation: Prediction count
    assert len(raw_preds) == len(
        test_ids
    ), "Number of predictions does not match number of test IDs."

    # Map predictions back to original class labels
    final_preds = [Config.INVERSE_CLASS_MAPPING[p] for p in raw_preds]

    # Create Submission DataFrame
    submission = pd.DataFrame({Config.ID_COL: test_ids, Config.TARGET_COL: final_preds})

    print("Sample Submission:")
    print(submission.head())

    # Save Submission
    submission.to_csv(Config.SUBMISSION_PATH, index=False)

    # Validation: File existence and content
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_check = pd.read_csv(Config.SUBMISSION_PATH)
    assert df_check.shape == (2000, 2), "Submission file shape mismatch."
    assert list(df_check.columns) == [
        Config.ID_COL,
        Config.TARGET_COL,
    ], "Submission columns mismatch."


if __name__ == "__main__":
    # 1. Configure
    setup_demo_config()

    # 2. Data
    train_X, train_y, val_X, val_y, test_X, test_ids = validate_data_loading()

    # 3. Model
    input_dim = train_X.shape[1]
    model = validate_model_architecture(input_dim)

    # 4. Train
    validate_training_loop(model, train_X, train_y, val_X, val_y)

    # 5. Inference
    validate_inference_and_submission(model, test_X, test_ids)

    print("\nAll pipeline components validated successfully.")
