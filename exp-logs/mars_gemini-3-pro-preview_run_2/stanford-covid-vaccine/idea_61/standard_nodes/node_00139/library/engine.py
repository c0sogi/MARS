import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, MCRMSE
from library.loss import AnchoredMCRMSELoss
from library.data import get_loaders
from library.model import ADSRN


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for inputs, targets, partner_indices in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        partner_indices = partner_indices.to(device)

        optimizer.zero_grad()

        # Forward pass returns (final_pred, aux_pred)
        y_2, y_1 = model(inputs, partner_indices)

        # Compute loss for both passes
        # AnchoredMCRMSELoss calculates loss over full sequence length (0-107)
        loss_main = criterion(y_2, targets)
        loss_aux = criterion(y_1, targets)

        # Combined loss
        loss = loss_main + Config.AUX_LOSS_WEIGHT * loss_aux

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using the competition metric.
    """
    model.eval()
    metric = MCRMSE()  # Defaults to scored columns [0, 1, 3] and seq_scored=68

    with torch.no_grad():
        for inputs, targets, partner_indices in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            partner_indices = partner_indices.to(device)

            # Inference only requires the final prediction y_2
            y_2, _ = model(inputs, partner_indices)

            # Update metric (handles slicing internally)
            metric.update(y_2, targets)

    return metric.compute()


class Engine:
    @staticmethod
    def run_training():
        """
        Main training loop with early stopping and model saving.
        """
        seed_everything(Config.SEED)
        device = torch.device(Config.DEVICE)

        # Get DataLoaders
        train_loader, val_loader, _ = get_loaders(load_cached_data=True)

        # Initialize Model
        model = ADSRN().to(device)

        # Optimizer & Scheduler
        optimizer = optim.AdamW(model.parameters(), lr=Config.LR)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Loss Function
        criterion = AnchoredMCRMSELoss()

        # Training State
        best_score = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

        print(f"Starting training on device: {device}")

        for epoch in range(Config.EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_score = validate(model, val_loader, device)

            # Step scheduler
            scheduler.step(val_score)

            # Print full precision metrics
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {train_loss} - Val MCRMSE: {val_score}"
            )

            # Checkpoint
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"Best Validation Score: {best_score}")
        return best_model_path

    @staticmethod
    def generate_submission(model_path):
        """
        Generates predictions for the test set and saves the submission file.
        """
        seed_everything(Config.SEED)
        device = torch.device(Config.DEVICE)

        # Get Test Loader
        _, _, test_loader = get_loaders(load_cached_data=True)

        # Load Model
        model = ADSRN().to(device)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        all_preds = []

        print("Generating predictions...")
        with torch.no_grad():
            for inputs, targets, partner_indices in test_loader:
                inputs = inputs.to(device)
                partner_indices = partner_indices.to(device)

                # Predict
                y_2, _ = model(inputs, partner_indices)
                all_preds.append(y_2.cpu().numpy())

        # Concatenate all batches: Shape (N_samples, 107, 5)
        preds_array = np.concatenate(all_preds, axis=0)

        # Retrieve IDs from the dataset
        ids = test_loader.dataset.ids

        # --- Format for Submission ---
        n_samples, seq_len, n_channels = preds_array.shape

        # Flatten predictions: (N_samples * 107, 5)
        preds_flat = preds_array.reshape(-1, n_channels)

        # Create 'id_seqpos' column
        # Repeat IDs for each sequence position: [id1, id1... (107 times), id2...]
        ids_repeated = np.repeat(ids, seq_len)
        # Tile sequence positions: [0, 1, ... 106, 0, 1, ... 106]
        seqpos_tiled = np.tile(np.arange(seq_len), n_samples)

        id_seqpos_list = [f"{i}_{s}" for i, s in zip(ids_repeated, seqpos_tiled)]

        # Define columns in correct order
        target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

        # Create DataFrame
        df_preds = pd.DataFrame(preds_flat, columns=target_cols)
        df_preds.insert(0, "id_seqpos", id_seqpos_list)

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        df_preds.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
