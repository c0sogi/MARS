import os
import time
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.model import cosine_similarity_loss
from library.utils import vector_to_azimuth_zenith


class Engine:
    """
    Engine class to handle training, validation, and inference for the IceCube DGCN model.
    """

    def __init__(self, model, device, optimizer, scheduler=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_loss = float("inf")
        self.best_epoch = -1

    def train_one_epoch(self, loader):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_graphs = 0

        for batch in loader:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            # batch.x: Node features, batch.batch: Graph assignment
            pred = self.model(batch)

            # Compute loss
            # batch.y is [batch_size, 3] unit vectors
            loss = cosine_similarity_loss(pred, batch.y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Step scheduler (OneCycleLR requires stepping every batch)
            if self.scheduler is not None:
                self.scheduler.step()

            # Accumulate loss (weighted by batch size)
            batch_size = batch.num_graphs
            total_loss += loss.item() * batch_size
            num_graphs += batch_size

        avg_loss = total_loss / num_graphs if num_graphs > 0 else 0.0
        return avg_loss

    def validate(self, loader):
        """
        Evaluates the model on the validation set.
        Computes Loss and Mean Angular Error.
        """
        self.model.eval()
        total_loss = 0.0
        total_angular_error = 0.0
        num_graphs = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                # Forward pass
                pred = self.model(batch)

                # Compute Loss
                loss = cosine_similarity_loss(pred, batch.y)

                # Accumulate Loss
                batch_size = batch.num_graphs
                total_loss += loss.item() * batch_size

                # Compute Metric: Mean Angular Error
                # 1. Normalize predictions and targets to unit vectors
                pred_norm = torch.nn.functional.normalize(pred, p=2, dim=1)
                target_norm = torch.nn.functional.normalize(batch.y, p=2, dim=1)

                # 2. Compute dot product (cosine of angle)
                dot_prod = torch.sum(pred_norm * target_norm, dim=1)

                # 3. Clamp to [-1, 1] to avoid numerical errors with acos
                dot_prod = torch.clamp(dot_prod, -1.0, 1.0)

                # 4. Compute angle in radians
                angles = torch.acos(dot_prod)

                # Accumulate angular error
                total_angular_error += angles.sum().item()
                num_graphs += batch_size

        avg_loss = total_loss / num_graphs if num_graphs > 0 else 0.0
        avg_mae = total_angular_error / num_graphs if num_graphs > 0 else 0.0

        return avg_loss, avg_mae

    def fit(self, train_loader, val_loader, epochs, patience, save_path):
        """
        Runs the full training loop with early stopping.
        """
        print(f"Starting training for {epochs} epochs with patience {patience}...")

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_mae = self.validate(val_loader)

            elapsed = time.time() - start_time

            # Print metrics with full precision
            print(f"Epoch {epoch}/{epochs} | Time: {elapsed:.2f}s")
            print(f"  Train Loss: {train_loss}")
            print(f"  Val Loss:   {val_loss}")
            print(f"  Val MAE:    {val_mae}")

            # Early Stopping and Checkpointing
            if val_loss < self.best_loss:
                print(
                    f"  Validation loss improved from {self.best_loss} to {val_loss}. Saving model..."
                )
                self.best_loss = val_loss
                self.best_epoch = epoch
                patience_counter = 0

                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(
            f"Training complete. Best Val Loss: {self.best_loss} at Epoch {self.best_epoch}"
        )

    def predict(self, test_loader, output_path):
        """
        Generates predictions for the test set and saves to CSV.
        """
        print("Generating predictions for test set...")
        self.model.eval()

        event_ids_list = []
        azimuth_list = []
        zenith_list = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)

                # Forward pass
                pred = self.model(batch)

                # Convert predicted vectors to Azimuth and Zenith
                # pred is [batch_size, 3]
                az, zen = vector_to_azimuth_zenith(pred)

                # Collect results
                event_ids_list.extend(batch.event_id.cpu().numpy())
                azimuth_list.extend(az.cpu().numpy())
                zenith_list.extend(zen.cpu().numpy())

        # Create DataFrame
        df_submission = pd.DataFrame(
            {"event_id": event_ids_list, "azimuth": azimuth_list, "zenith": zenith_list}
        )

        # Ensure event_id is integer
        df_submission["event_id"] = df_submission["event_id"].astype(int)

        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
