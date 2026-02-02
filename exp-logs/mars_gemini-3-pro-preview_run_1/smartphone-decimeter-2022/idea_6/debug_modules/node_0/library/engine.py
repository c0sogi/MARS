import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import library.config as C
import library.utils as U


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device):
        """
        Initializes the Trainer.

        Args:
            model: The PyTorch model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            optimizer: Optimizer for model parameters.
            device: Torch device (cpu or cuda).
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        # reduction='none' allows us to manually apply the padding mask later
        self.criterion = nn.L1Loss(reduction="none")

    def train_one_epoch(self):
        """
        Trains the model for one epoch.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_count = 0

        for batch in self.train_loader:
            # Move data to device
            features = batch["features"].to(self.device)  # (B, L, InputDim)
            targets = batch["targets"].to(self.device)  # (B, L, OutputDim)
            phone_idx = batch["phone_idx"].to(self.device)  # (B,)
            mask = batch["padding_mask"].to(
                self.device
            )  # (B, L), True indicates padding

            # Prepare inputs for model: (Batch, Channels, Length)
            features = features.permute(0, 2, 1)

            # Forward pass
            outputs = self.model(features, phone_idx)  # (B, OutputDim, L)

            # Align outputs with targets: (Batch, Length, OutputDim)
            outputs = outputs.permute(0, 2, 1)

            # Calculate unreduced loss
            loss_unreduced = self.criterion(outputs, targets)  # (B, L, OutputDim)

            # Expand mask for output dimensions to match loss shape
            # mask is (B, L), need (B, L, OutputDim)
            mask_expanded = mask.unsqueeze(-1).expand_as(loss_unreduced)

            # Select only valid elements (where mask is False)
            valid_loss = loss_unreduced[~mask_expanded]

            if valid_loss.numel() > 0:
                loss = valid_loss.mean()

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Accumulate weighted loss
                total_loss += loss.item() * valid_loss.numel()
                total_count += valid_loss.numel()

        return total_loss / total_count if total_count > 0 else 0.0

    def evaluate(self):
        """
        Evaluates the model on the validation set.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        total_count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch["features"].to(self.device)
                targets = batch["targets"].to(self.device)
                phone_idx = batch["phone_idx"].to(self.device)
                mask = batch["padding_mask"].to(self.device)

                features = features.permute(0, 2, 1)
                outputs = self.model(features, phone_idx)
                outputs = outputs.permute(0, 2, 1)

                loss_unreduced = self.criterion(outputs, targets)
                mask_expanded = mask.unsqueeze(-1).expand_as(loss_unreduced)
                valid_loss = loss_unreduced[~mask_expanded]

                if valid_loss.numel() > 0:
                    total_loss += valid_loss.item() * valid_loss.numel()
                    total_count += valid_loss.numel()

        return total_loss / total_count if total_count > 0 else 0.0

    def save_model(self, path):
        """Saves the model state dict to the specified path."""
        torch.save(self.model.state_dict(), path)

    def fit(
        self, epochs=C.NUM_EPOCHS, patience=C.EARLY_STOPPING_PATIENCE, save_path=None
    ):
        """
        Main training loop with early stopping.

        Args:
            epochs (int): Maximum number of epochs.
            patience (int): Early stopping patience.
            save_path (str): Path to save the best model.
        """
        if save_path is None:
            save_path = os.path.join(C.WORKING_DIR, "model_weights.pth")

        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.evaluate()

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(save_path)
                print(f"  New best model saved to {save_path}")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Val Loss: {best_val_loss}")


def generate_submission(model, test_loader, device):
    """
    Generates predictions for the test set, reconstructs coordinates,
    and saves the results to a CSV file.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for the test set (must be shuffle=False).
        device: Torch device.
    """
    print("Generating submission...")
    model.eval()

    # Ensure submission directory exists
    os.makedirs(C.SUBMISSION_DIR, exist_ok=True)

    trip_idx = 0

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            phone_idx = batch["phone_idx"].to(device)
            lengths = batch["lengths"]
            wls_pos = batch["wls_pos"].numpy()  # (B, L, 3) -> Lat, Lon, Alt

            # Forward pass
            features = features.permute(0, 2, 1)
            outputs = model(features, phone_idx)  # (B, 2, L)
            outputs = outputs.permute(0, 2, 1).cpu().numpy()  # (B, L, 2)

            batch_size = features.size(0)

            for b in range(batch_size):
                length = lengths[b]

                # Extract valid predictions and WLS positions for the current sequence
                # Output 0: North (dLat), Output 1: East (dLon)
                pred_n = outputs[b, :length, 0]
                pred_e = outputs[b, :length, 1]

                curr_wls = wls_pos[b, :length, :]
                wls_lat = curr_wls[:, 0]
                wls_lon = curr_wls[:, 1]
                wls_alt = curr_wls[:, 2]

                # Reconstruct Geodetic Coordinates (Lat, Lon) from ENU residuals
                # We assume Up deviation is 0 for reconstruction
                pred_lat, pred_lon, _ = U.enu_to_geodetic(
                    pred_e, pred_n, np.zeros_like(pred_e), wls_lat, wls_lon, wls_alt
                )

                # Assign predictions back to the corresponding dataframe in the dataset
                # We rely on the deterministic order of the loader (shuffle=False)
                if trip_idx < len(test_loader.dataset.sequences):
                    df = test_loader.dataset.sequences[trip_idx]

                    # Sanity check for length alignment
                    if len(df) != length:
                        print(
                            f"Warning: Length mismatch for trip {trip_idx}. DF: {len(df)}, Pred: {length}"
                        )
                        # Truncate to minimum length to avoid errors
                        min_len = min(len(df), length)
                        df = df.iloc[:min_len]
                        pred_lat = pred_lat[:min_len]
                        pred_lon = pred_lon[:min_len]

                    df["LatitudeDegrees"] = pred_lat
                    df["LongitudeDegrees"] = pred_lon

                trip_idx += 1

    # Concatenate all processed sequences into a single DataFrame
    all_preds = pd.concat(test_loader.dataset.sequences, ignore_index=True)

    # Select required columns for submission
    submission = all_preds[
        ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    ]

    # Save to CSV
    output_path = os.path.join(C.SUBMISSION_DIR, "submission.csv")
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
