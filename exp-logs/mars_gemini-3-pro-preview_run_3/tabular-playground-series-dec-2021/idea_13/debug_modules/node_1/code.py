import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.data_utils import process_data, TabularDataset
from library.model import ParallelLowRankDCNResNet
from library.train_utils import train_one_epoch, validate, EarlyStopping


def set_seed(seed=42):
    """Sets fixed random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("Starting Library Usage Demonstration...")
    set_seed(42)

    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    print("\n[1] Configuring parameters for fast demonstration...")

    # Override Config parameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 64
    Config.WORKING_DIR = "./working/demo_run"  # Separate dir for demo artifacts

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Epochs: {Config.EPOCHS}")

    # =========================================================================
    # 2. Data Processing & Slicing
    # =========================================================================
    print("\n[2] Processing and slicing data...")

    # Load and process full data (cached if available, otherwise generated)
    # We use the library function process_data
    X_train_full, y_train_full, X_val_full, y_val_full, X_test_full, test_ids_full = (
        process_data(load_cached_data=True)
    )

    # Verify data shapes
    print(f"    Full Train Shape: {X_train_full.shape}")
    assert (
        X_train_full.shape[1] == Config.INPUT_DIM
    ), f"Expected input dim {Config.INPUT_DIM}, got {X_train_full.shape[1]}"

    # Slice data to create a tiny dataset for rapid demonstration
    subset_size = 1000
    print(f"    Slicing to top {subset_size} samples for speed...")

    X_train_sub = X_train_full[:subset_size]
    y_train_sub = y_train_full[:subset_size]
    X_val_sub = X_val_full[:subset_size]
    y_val_sub = y_val_full[:subset_size]
    X_test_sub = X_test_full[:subset_size]

    # Instantiate Datasets using the library class
    train_dataset = TabularDataset(X_train_sub, y_train_sub)
    val_dataset = TabularDataset(X_val_sub, y_val_sub)
    test_dataset = TabularDataset(X_test_sub)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    print("    DataLoaders created successfully.")

    # =========================================================================
    # 3. Model Initialization
    # =========================================================================
    print("\n[3] Initializing ParallelLowRankDCNResNet model...")

    model = ParallelLowRankDCNResNet().to(device)

    # Verify model architecture output shape
    dummy_input = torch.randn(2, Config.INPUT_DIM).to(device)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    print(f"    Model Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Expected output shape (2, {Config.NUM_CLASSES}), got {dummy_output.shape}"

    # =========================================================================
    # 4. Training Loop Demonstration
    # =========================================================================
    print("\n[4] Running training loop demonstration...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    for epoch in range(Config.EPOCHS):
        # Use library function: train_one_epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Use library function: validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(
            f"    Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Assertions to ensure learning mechanics are working
        assert not np.isnan(train_loss), "Training loss is NaN"
        assert 0 <= train_acc <= 1, "Training accuracy out of bounds"
        assert 0 <= val_acc <= 1, "Validation accuracy out of bounds"

    print("    Training loop completed successfully.")

    # =========================================================================
    # 5. Early Stopping Verification
    # =========================================================================
    print("\n[5] Verifying EarlyStopping logic...")

    # Create a dummy model to save weights
    dummy_model = nn.Linear(10, 2)
    save_path = os.path.join(Config.WORKING_DIR, "early_stop_test.pth")

    # Initialize EarlyStopping with patience=2
    es = EarlyStopping(patience=2, verbose=True, path=save_path)

    # Simulate loss sequence: Decrease -> Increase -> Increase (Should trigger stop)
    losses = [0.5, 0.4, 0.45, 0.46]
    triggered = False

    print("    Simulating loss sequence: [0.5, 0.4, 0.45, 0.46] with patience=2")
    for i, loss in enumerate(losses):
        es(loss, dummy_model)
        if es.early_stop:
            print(f"    Early stopping triggered at step {i+1} (Loss: {loss})")
            triggered = True
            break

    assert triggered, "EarlyStopping failed to trigger on increasing loss sequence."
    assert os.path.exists(
        save_path
    ), "EarlyStopping failed to save the checkpoint file."
    print("    EarlyStopping logic verified.")

    # =========================================================================
    # 6. Inference Demonstration
    # =========================================================================
    print("\n[6] Running inference on test subset...")

    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            # Map 0-6 back to 1-7
            predicted = predicted + 1
            predictions.extend(predicted.cpu().numpy())

    predictions = np.array(predictions)

    print(f"    Predictions generated: {len(predictions)}")
    print(f"    Sample predictions: {predictions[:10]}")

    assert (
        len(predictions) == subset_size
    ), f"Expected {subset_size} predictions, got {len(predictions)}"
    assert np.all(
        (predictions >= 1) & (predictions <= 7)
    ), "Predictions contain invalid class labels"

    print("    Inference completed successfully.")
    print("\nAll demonstrations and verifications passed!")


if __name__ == "__main__":
    main()
