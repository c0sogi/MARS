import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import random

from library.config import Config
from library.model import MNPADSModel
from library.data import get_data_loaders


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
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
        atomic_features = batch["atomic_features"].to(device)
        global_features = batch["global_features"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        outputs = model(atomic_features, global_features, mask)
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
            atomic_features = batch["atomic_features"].to(device)
            global_features = batch["global_features"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(atomic_features, global_features, mask)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * targets.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.
    Applies inverse log transformation (expm1) to predictions.
    """
    model.eval()
    results = []

    print("Generating predictions for test set...")
    with torch.no_grad():
        for batch in test_loader:
            atomic_features = batch["atomic_features"].to(device)
            global_features = batch["global_features"].to(device)
            mask = batch["mask"].to(device)
            ids = batch["ids"]  # List of IDs

            # Forward pass (log scale)
            outputs = model(atomic_features, global_features, mask)

            # Inverse transform: exp(x) - 1
            predictions = torch.expm1(outputs).cpu().numpy()

            for i, id_val in enumerate(ids):
                # predictions[i] is [formation_energy, bandgap_energy]
                results.append(
                    {
                        "id": id_val,
                        "formation_energy_ev_natom": predictions[i][0],
                        "bandgap_energy_ev": predictions[i][1],
                    }
                )

    df_submission = pd.DataFrame(results)
    # Ensure column order matches sample submission
    df_submission = df_submission[
        ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
    ]

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training():
    """
    Main execution function for the training pipeline.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data
    # load_cached_data=True will look for npz files in working/idea_35/
    # If not found, it will process from scratch using ./metadata/ and ./input/
    train_loader, val_loader, test_loader = get_data_loaders(load_cached_data=True)

    # 3. Model
    model = MNPADSModel(config=Config).to(device)

    # 4. Optimization
    # Since targets are log1p transformed in data.py, MSE loss represents MSLE directly.
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.MIN_LR,
        verbose=True,
    )

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_loss)

        # Logging (Full precision)
        # Note: Val Loss here is Mean Squared Error on Log-Transformed targets.
        # This is equivalent to Mean Squared Logarithmic Error (MSLE).
        # The competition metric is Root Mean Squared Logarithmic Error (RMSLE).
        rmsle = np.sqrt(val_loss)
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss (MSLE): {train_loss} | Val Loss (MSLE): {val_loss} | Val RMSLE: {rmsle}"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  New best model saved! (Val Loss: {val_loss})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Submission
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    generate_submission(model, test_loader, device, Config.SUBMISSION_OUTPUT_PATH)
