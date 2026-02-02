import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import (
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE,
    LEARNING_RATE,
    EPOCHS,
    PATIENCE,
    WORKING_DIR,
    SUBMISSION_PATH,
    DEBUG_SUBSET_SIZE,
    SEED,
)
from library.utils import seed_everything
from library.dataset import IceCubeDataset
from library.model import (
    GeometricPulseAggregator,
    CosineDistanceLoss,
    calculate_angular_error,
    predict_and_submit,
)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one epoch of training.

    Args:
        model: The PyTorch model.
        loader: DataLoader for the training set.
        optimizer: The optimizer instance.
        criterion: The loss function.
        device: The device (cpu or cuda) to run on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(features)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        batch_size = features.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for the validation set.
        criterion: The loss function.
        device: The device (cpu or cuda) to run on.

    Returns:
        tuple: (average_loss, average_angular_error)
    """
    model.eval()
    running_loss = 0.0
    running_angular_error = 0.0
    dataset_size = 0

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            outputs = model(features)
            loss = criterion(outputs, targets)

            # Calculate angular error for metrics
            ang_error = calculate_angular_error(outputs, targets)

            batch_size = features.size(0)
            running_loss += loss.item() * batch_size
            running_angular_error += ang_error * batch_size
            dataset_size += batch_size

    avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    avg_angular_error = (
        running_angular_error / dataset_size if dataset_size > 0 else 0.0
    )

    return avg_loss, avg_angular_error


def run_training(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    patience=PATIENCE,
    debug_subset_size=DEBUG_SUBSET_SIZE,
    save_dir=WORKING_DIR,
):
    """
    Orchestrates the training process, including data loading, training loop,
    validation, early stopping, and submission generation.
    """
    # 1. Setup
    seed_everything(SEED)
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "best_model.pth")

    print(f"Initializing training on device: {DEVICE}")

    # 2. Data Loading
    print("Loading datasets...")
    train_dataset = IceCubeDataset(
        metadata_path=TRAIN_META_PATH,
        mode="train",
        debug_subset_size=debug_subset_size,
    )
    val_dataset = IceCubeDataset(
        metadata_path=VAL_META_PATH,
        mode="val",
        debug_subset_size=debug_subset_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model & Optimizer
    model = GeometricPulseAggregator().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = CosineDistanceLoss()

    # 4. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training loop...")
    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)

        # Validate
        val_loss, val_ang_error = validate(model, val_loader, criterion, DEVICE)

        # Print Metrics
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Angular Error: {val_ang_error}"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Val Loss: {best_val_loss}")

    # 5. Submission Generation
    print("Generating submission...")

    # Load best model weights
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))

    # Prepare Test Loader
    test_dataset = IceCubeDataset(
        metadata_path=TEST_META_PATH,
        mode="test",
        debug_subset_size=debug_subset_size,  # Use subset if debugging, else full
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Generate and save predictions
    predict_and_submit(model, test_loader, DEVICE, SUBMISSION_PATH)
