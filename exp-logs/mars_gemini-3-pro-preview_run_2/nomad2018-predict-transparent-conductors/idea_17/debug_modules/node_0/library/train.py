import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.data import get_dataloaders
from library.model import LRCGCNN
from library.utils import set_seed, StandardScaler


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        batch = batch.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch)

        # Compute loss
        # Targets are already standardized in the loader
        loss = criterion(outputs, batch.y)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item() * batch.num_graphs

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Forward pass
            outputs = model(batch)

            # Compute loss
            loss = criterion(outputs, batch.y)

            running_loss += loss.item() * batch.num_graphs

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def generate_submission(model, loader, scaler, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.eval()
    ids_list = []
    preds_list = []

    print("Generating submission...")

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Forward pass (outputs are standardized)
            outputs = model(batch)

            # Inverse transform to get original scale
            # scaler handles tensors and device internally
            preds_original = scaler.inverse_transform(outputs)

            # Collect results
            ids_list.extend(batch.material_id.cpu().numpy())
            preds_list.append(preds_original.cpu().numpy())

    # Concatenate predictions
    preds_array = np.concatenate(preds_list, axis=0)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {
            "id": ids_list,
            "formation_energy_ev_natom": preds_array[:, 0],
            "bandgap_energy_ev": preds_array[:, 1],
        }
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(load_cached_data=True):
    """
    Orchestrates the training pipeline.
    """
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    print("Preparing DataLoaders...")
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # Initialize Model
    # Note: Input dimensions are defined in Config
    model = LRCGCNN(
        atom_fea_len=Config.ATOM_FEA_LEN,
        h_fea_len=Config.H_FEA_LEN,
        n_conv=Config.N_CONV,
        n_h=Config.N_H,
        n_rbf=Config.N_RBF,
        radius=Config.RADIUS,
    ).to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Loss Function
    criterion = nn.MSELoss()

    # Training Loop variables
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate
        val_loss = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{Config.EPOCHS} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            os.makedirs(os.path.dirname(Config.MODEL_CHECKPOINT), exist_ok=True)
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)
            # print(f"  New best model saved! Val Loss: {val_loss}")
        else:
            patience_counter += 1
            # print(f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print(f"Training complete. Best Validation Loss: {best_val_loss}")

    # Load best model for submission
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT, map_location=device))

    # Generate Submission
    generate_submission(model, test_loader, scaler, device, Config.SUBMISSION_FILE)
