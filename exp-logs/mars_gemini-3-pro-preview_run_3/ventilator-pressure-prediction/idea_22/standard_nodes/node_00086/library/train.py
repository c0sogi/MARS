import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import sys

from library import config, data, model, utils


# =============================================================================
# Loss Function
# =============================================================================
class MaskedL1Loss(nn.Module):
    """
    Computes Mean Absolute Error (L1 Loss) only for the inspiratory phase.
    The inspiratory phase is defined where u_out == 0.
    """

    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, u_out):
        """
        Args:
            pred: (Batch, Seq_Len)
            target: (Batch, Seq_Len)
            u_out: (Batch, Seq_Len) - 0 for inspiratory, 1 for expiratory
        """
        # Create mask: 1 where u_out == 0, else 0
        mask = 1 - u_out

        # Calculate absolute error
        error = torch.abs(pred - target)

        # Apply mask
        masked_error = error * mask

        # Compute mean over valid elements
        # Add a small epsilon to denominator to avoid division by zero (though unlikely in batches)
        loss = masked_error.sum() / (mask.sum() + 1e-8)

        return loss


# =============================================================================
# Trainer Class
# =============================================================================
class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.best_val_loss = float("inf")

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0

        for batch in self.train_loader:
            # Move data to device
            x = batch["x"].to(self.device)
            u_out = batch["u_out"].to(self.device)
            y = batch["y"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(x)

            # Compute loss
            loss = self.criterion(preds, y, u_out)

            # Backward pass
            loss.backward()

            # Gradient Clipping (Mandatory for stability)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), config.CLIP_GRAD_NORM
            )

            # Optimizer step
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                x = batch["x"].to(self.device)
                u_out = batch["u_out"].to(self.device)
                y = batch["y"].to(self.device)

                preds = self.model(x)
                loss = self.criterion(preds, y, u_out)

                running_loss += loss.item()

        return running_loss / len(self.val_loader)

    def fit(self, epochs):
        print(f"Starting training for {epochs} epochs on device: {self.device}")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            # Print full precision metrics
            print(
                f"Epoch {epoch}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
            )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), config.MODEL_SAVE_PATH)
                print(f"New best model saved with Val Loss: {val_loss}")


# =============================================================================
# Main Execution Function
# =============================================================================
def run_training(debug=False, load_cached_data=True):
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    utils.seed_everything(config.SEED)
    device = utils.get_device()

    # 2. Data Loading
    # Pass arguments to prepare_data to handle caching and debug mode
    train_loader, val_loader, test_loader = data.prepare_data(
        load_cached_data=load_cached_data, debug=debug
    )

    # 3. Model Initialization
    net = model.CWDHNet()
    net.to(device)

    # 4. Optimizer and Loss
    # Using AdamW as implied by weight decay in config
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    criterion = MaskedL1Loss()

    # 5. Training
    trainer = Trainer(net, train_loader, val_loader, criterion, optimizer, device)

    # Determine epochs based on debug flag
    epochs = 2 if debug else config.EPOCHS
    trainer.fit(epochs)

    # 6. Inference and Submission
    print("Generating predictions for submission...")

    # Load best model
    net.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    net.eval()

    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            # We don't need u_out for prediction, just x
            preds = net(x)

            # Flatten predictions (Batch, Seq_Len) -> (Batch * Seq_Len)
            # Move to CPU and numpy
            preds_flat = preds.view(-1).cpu().numpy()
            predictions.append(preds_flat)

    # Concatenate all batches
    all_predictions = np.concatenate(predictions)

    # 7. Create Submission File
    # We need to map predictions back to IDs.
    # The data loader preserves the order of data.prepare_data, which sorts by breath_id and id.
    # We load the raw test metadata to get the IDs in the correct order.

    test_meta_path = config.TEST_PATH
    if not os.path.exists(test_meta_path):
        # Fallback to input directory if metadata not found (though it should be there)
        test_meta_path = os.path.join(config.INPUT_DIR, "test.csv")

    print(f"Loading test metadata from {test_meta_path} for ID mapping...")
    test_df = pd.read_csv(test_meta_path)

    # Apply the same sorting as in data.prepare_data
    # "df = df.sort_values(by=[config.BREATH_ID_COL, config.ID_COL])"
    test_df = test_df.sort_values(by=[config.BREATH_ID_COL, config.ID_COL])

    # Verify length matches
    if len(all_predictions) != len(test_df):
        raise ValueError(
            f"Prediction count {len(all_predictions)} does not match test set size {len(test_df)}"
        )

    # Assign predictions
    test_df["pressure"] = all_predictions

    # Prepare final submission dataframe: id, pressure
    submission_df = test_df[[config.ID_COL, "pressure"]].copy()

    # Sort by ID as per sample submission format (usually strictly increasing ID)
    submission_df = submission_df.sort_values(by=config.ID_COL)

    # Save
    print(f"Saving submission to {config.SUBMISSION_PATH}...")
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print("Done.")
