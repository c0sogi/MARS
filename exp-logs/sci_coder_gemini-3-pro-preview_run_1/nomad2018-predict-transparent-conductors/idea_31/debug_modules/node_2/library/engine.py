import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import DEVICE, TRAINING_PARAMS, WORKING_DIR, SUBMISSION_PATH, SEED
from library.data import get_data_loaders
from library.architecture import LCEWDS


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        atomic_feats = batch["atomic_features"].to(device)
        global_feats = batch["global_features"].to(device)
        batch_indices = batch["batch_indices"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        outputs = model(atomic_feats, global_feats, batch_indices)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            atomic_feats = batch["atomic_features"].to(device)
            global_feats = batch["global_features"].to(device)
            batch_indices = batch["batch_indices"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(atomic_feats, global_feats, batch_indices)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * targets.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def run_training(load_cached_data=True):
    """
    Orchestrates the training process including data loading, model initialization,
    training loop, validation, scheduling, and early stopping.
    """
    set_seed(SEED)

    # Get DataLoaders
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=TRAINING_PARAMS["batch_size"], load_cached_data=load_cached_data
    )

    # Initialize Model
    model = LCEWDS().to(DEVICE)

    # Loss and Optimizer
    # Targets are log1p transformed, so MSE on them corresponds to MSLE on original scale
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAINING_PARAMS["learning_rate"],
        weight_decay=TRAINING_PARAMS["weight_decay"],
    )

    # Scheduler
    # Cite debug_lesson_2: Remove Deprecated verbose Argument from PyTorch Scheduler Constructors
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=TRAINING_PARAMS["factor"],
        patience=5,  # Reduce LR if no improvement for 5 epochs
        min_lr=TRAINING_PARAMS["min_lr"],
    )

    # Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pt")

    print(f"Starting training on {DEVICE}...")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    for epoch in range(TRAINING_PARAMS["epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)

        # Scheduler step
        scheduler.step(val_loss)

        # Print metrics (full precision)
        print(
            f"Epoch {epoch+1}/{TRAINING_PARAMS['epochs']} - "
            f"Train Loss (MSE): {train_loss} - "
            f"Val Loss (MSE): {val_loss} - "
            f"Val RMSLE: {np.sqrt(val_loss)}"
        )

        # Early Stopping and Model Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= TRAINING_PARAMS["patience"]:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Training complete. Best Val Loss: {best_val_loss}")

    # Load best model for submission generation
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    return model, test_loader


def generate_submission(model, test_loader):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()
    results = []

    print("Generating predictions for test set...")

    with torch.no_grad():
        for batch in test_loader:
            atomic_feats = batch["atomic_features"].to(DEVICE)
            global_feats = batch["global_features"].to(DEVICE)
            batch_indices = batch["batch_indices"].to(DEVICE)
            ids = batch["ids"].cpu().numpy()

            # Forward pass
            outputs = model(atomic_feats, global_feats, batch_indices)

            # Inverse transform: exp(x) - 1
            # Targets were log1p(y), so y = expm1(target)
            predictions = torch.expm1(outputs).cpu().numpy()

            for i, sample_id in enumerate(ids):
                formation_energy = predictions[i][0]
                bandgap_energy = predictions[i][1]

                # Ensure non-negative if physical constraint dictates
                formation_energy = max(0.0, formation_energy)
                bandgap_energy = max(0.0, bandgap_energy)

                results.append(
                    {
                        "id": sample_id,
                        "formation_energy_ev_natom": formation_energy,
                        "bandgap_energy_ev": bandgap_energy,
                    }
                )

    # Create DataFrame
    submission_df = pd.DataFrame(results)

    # Sort by ID just in case
    submission_df = submission_df.sort_values("id")

    # Save to CSV
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(submission_df.head())
