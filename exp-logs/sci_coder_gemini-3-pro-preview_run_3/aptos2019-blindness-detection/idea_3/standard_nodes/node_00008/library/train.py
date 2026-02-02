import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from library.dataset import get_dataloaders
from library.model import (
    RetinopathyModel,
    train_one_epoch,
    validate,
    generate_submission,
)
from library.utils import seed_everything


def train(
    epochs=25,
    batch_size=8,
    learning_rate=1e-4,
    patience=6,
    debug_subset_size=None,
    save_dir="./working/idea_3",
    submission_path="./submission/submission.csv",
):
    """
    Main training function for Diabetic Retinopathy prediction.

    Args:
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for training and validation.
        learning_rate (float): Learning rate for the optimizer.
        patience (int): Early stopping patience.
        debug_subset_size (int, optional): If set, limits the dataset size for debugging.
        save_dir (str): Directory to save model checkpoints.
        submission_path (str): Path to save the submission CSV.
    """
    # 1. Setup
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")

    print(f"Device: {device}")

    # 2. Data Loading
    # Using 512x512 resolution as per strategy to capture fine details
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, image_size=512, load_cached_data=True
    )

    # Debugging: Subset datasets if requested
    if debug_subset_size is not None:
        print(f"Debugging mode: Subsetting datasets to {debug_subset_size} samples.")

        train_ds = train_loader.dataset
        val_ds = val_loader.dataset

        # Ensure we don't exceed dataset length
        t_indices = list(range(min(len(train_ds), debug_subset_size)))
        v_indices = list(range(min(len(val_ds), debug_subset_size)))

        train_subset = Subset(train_ds, t_indices)
        val_subset = Subset(val_ds, v_indices)

        # Recreate loaders with subsets
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        # Note: We typically do not subset the test loader as submission requires all predictions.

    # 3. Model Initialization
    model = RetinopathyModel(pretrained=True)
    model.to(device)

    # 4. Optimization Setup
    # Regression loss for ordinal data
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda")

    # Scheduler: Monitor QWK (maximize)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # 5. Training Loop
    best_qwk = -float("inf")
    epochs_no_improve = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()

        # Train Step
        train_loss, train_qwk = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        # Validation Step
        val_loss, val_qwk = validate(model, val_loader, criterion, device)

        duration = time.time() - start_time

        # Print Metrics (Full Precision)
        print(f"Epoch {epoch+1}/{epochs} [{duration:.2f}s]")
        print(f"Train Loss: {train_loss} QWK: {train_qwk}")
        print(f"Val Loss: {val_loss} QWK: {val_qwk}")

        # Scheduler Step
        scheduler.step(val_qwk)

        # Checkpointing & Early Stopping
        if val_qwk > best_qwk:
            print(f"Validation QWK improved ({best_qwk} -> {val_qwk}). Saving model...")
            best_qwk = val_qwk
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement. Patience {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    # 6. Submission
    print("Loading best model for submission...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: No best model found. Using current model state.")

    generate_submission(model, test_loader, device, output_path=submission_path)
