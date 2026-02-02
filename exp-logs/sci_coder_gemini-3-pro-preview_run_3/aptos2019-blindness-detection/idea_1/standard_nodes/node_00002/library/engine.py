import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import RetinopathyDataset, get_transforms
from library.model import ResNet18Regression, train_one_epoch, validate, predict
from library.utils import seed_everything


def get_dataloaders(
    train_csv_path=Config.TRAIN_METADATA_PATH,
    val_csv_path=Config.VAL_METADATA_PATH,
    test_csv_path=Config.TEST_METADATA_PATH,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
    debug_size=Config.DEBUG_SIZE,
):
    """
    Creates DataLoaders for train, validation, and test sets using metadata.
    """
    # Load metadata DataFrames
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Handle debug mode
    if debug:
        train_df = train_df.head(debug_size)
        val_df = val_df.head(debug_size)
        test_df = test_df.head(debug_size)

    # Initialize Datasets
    train_dataset = RetinopathyDataset(train_df, transform=get_transforms("train"))
    val_dataset = RetinopathyDataset(val_df, transform=get_transforms("val"))
    test_dataset = RetinopathyDataset(test_df, transform=get_transforms("test"))

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_df


def train(
    train_loader,
    val_loader,
    epochs=Config.NUM_EPOCHS,
    lr=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    patience=Config.PATIENCE,
    device=Config.DEVICE,
    save_path=Config.MODEL_PATH,
):
    """
    Main training loop with Early Stopping and full-precision logging.
    """
    seed_everything(Config.SEED)

    # Initialize Model
    model = ResNet18Regression(pretrained=True)
    model.to(device)

    # Define Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    patience_counter = 0

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_qwk = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val QWK: {val_qwk}")

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Load best model weights
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print("Loaded best model weights.")

    return model


def evaluate(model, val_loader, device=Config.DEVICE):
    """
    Evaluates the model and prints metrics.
    """
    criterion = nn.MSELoss()
    loss, qwk = validate(model, val_loader, criterion, device)
    print(f"Evaluation - Loss: {loss}")
    print(f"Evaluation - QWK: {qwk}")
    return loss, qwk


def generate_submission(
    model,
    test_loader,
    test_df,
    device=Config.DEVICE,
    output_path=Config.SUBMISSION_PATH,
):
    """
    Generates predictions for the test set and saves the submission file.
    """
    # Generate integer predictions
    preds = predict(model, test_loader, device=device)

    # Create submission DataFrame
    submission = pd.DataFrame({"id_code": test_df["id_code"], "diagnosis": preds})

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
