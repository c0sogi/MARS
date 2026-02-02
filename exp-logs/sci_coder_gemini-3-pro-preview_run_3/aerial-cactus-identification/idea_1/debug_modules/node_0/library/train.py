import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.utils import seed_everything, get_device
from library.model import SimpleCNN, train_one_epoch, validate, predict
from library.dataset import create_dataloaders


def fit_model(
    model, train_loader, val_loader, criterion, optimizer, device, epochs, patience
):
    """
    Manages the training loop over multiple epochs with early stopping.

    Args:
        model: The neural network model.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        criterion: The loss function (Binary Cross Entropy).
        optimizer: The optimizer (Adam).
        device: The computing device (CPU or GPU).
        epochs: Maximum number of epochs to train.
        patience: Number of epochs to wait for improvement before stopping.

    Returns:
        model: The model with the best validation weights loaded.
    """
    best_auc = -1.0
    patience_counter = 0
    best_model_state = None

    print(f"Starting training for {epochs} epochs with patience {patience}...")

    for epoch in range(epochs):
        # Perform one epoch of training
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate the model
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val AUC: {val_auc}"
        )

        # Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Restore the best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with Val AUC: {best_auc}")

    return model


def train_cactus_classifier(
    epochs=20,
    batch_size=64,
    learning_rate=1e-3,
    patience=5,
    load_cached_data=True,
    input_dir="./input",
    metadata_dir="./metadata",
    submission_dir="./submission",
):
    """
    Main pipeline to setup data, train the model, and generate the submission file.

    Args:
        epochs (int): Maximum training epochs.
        batch_size (int): Batch size for data loaders.
        learning_rate (float): Learning rate for the optimizer.
        patience (int): Early stopping patience.
        load_cached_data (bool): Whether to load pre-processed data from cache.
        input_dir (str): Path to input images.
        metadata_dir (str): Path to metadata CSVs.
        submission_dir (str): Path to save the submission file.
    """
    # 1. Setup Environment
    seed_everything(42)
    device = get_device()
    print(f"Using device: {device}")

    # 2. Prepare Data
    # create_dataloaders handles loading metadata, caching processed arrays, and creating loaders
    train_loader, val_loader, test_loader, test_ids = create_dataloaders(
        batch_size=batch_size,
        input_dir=input_dir,
        metadata_dir=metadata_dir,
        load_cached_data=load_cached_data,
    )

    # 3. Initialize Model, Loss, and Optimizer
    model = SimpleCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Train the Model
    model = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        patience=patience,
    )

    # 5. Generate Predictions on Test Set
    print("Generating predictions for test set...")
    test_probs = predict(model, test_loader, device)

    # 6. Save Submission
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    submission_df = pd.DataFrame({"id": test_ids, "has_cactus": test_probs})

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
