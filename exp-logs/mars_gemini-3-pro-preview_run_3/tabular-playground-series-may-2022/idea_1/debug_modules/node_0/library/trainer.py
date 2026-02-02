import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import (
    DEVICE,
    LEARNING_RATE,
    EPOCHS,
    PATIENCE,
    WEIGHT_DECAY,
    WORKING_DIR,
    SUBMISSION_DIR,
    CONTINUOUS_FEATURES,
    BATCH_SIZE,
)
from library.utils import seed_everything, compute_auc
from library.data_processor import make_dataloaders
from library.model import EntityEmbeddingMLP


class Trainer:
    """
    Manages the training, validation, and inference lifecycle of the model.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        patience=PATIENCE,
        checkpoint_path=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.patience = patience
        self.checkpoint_path = checkpoint_path or os.path.join(
            WORKING_DIR, "best_model.pth"
        )

        self.best_auc = -float("inf")
        self.patience_counter = 0
        self.model.to(self.device)

    def train_epoch(self, train_loader):
        """Runs one epoch of training."""
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            cont_data = batch["continuous"].to(self.device)
            cat_data = batch["categorical"].to(self.device)
            targets = batch["target"].to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()
            outputs = self.model(cont_data, cat_data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * cont_data.size(0)

        return total_loss / len(train_loader.dataset)

    def validate(self, val_loader):
        """Runs validation and computes AUC."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                cont_data = batch["continuous"].to(self.device)
                cat_data = batch["categorical"].to(self.device)
                targets = batch["target"].to(self.device).unsqueeze(1)

                outputs = self.model(cont_data, cat_data)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item() * cont_data.size(0)
                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())

        avg_loss = total_loss / len(val_loader.dataset)

        # Concatenate predictions and targets for AUC calculation
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_targets).numpy()

        auc_score = compute_auc(y_true, y_pred)

        return avg_loss, auc_score

    def fit(self, train_loader, val_loader, epochs=EPOCHS):
        """Main training loop with Early Stopping."""
        print(f"Starting training on device: {self.device}")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_auc = self.validate(val_loader)

            # Print metrics with full precision
            print(
                f"Epoch {epoch}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
            )

            # Early Stopping and Checkpointing
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        print(f"Training complete. Best Val AUC: {self.best_auc}")

    def predict(self, test_loader):
        """Generates predictions using the best saved model."""
        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(
                torch.load(self.checkpoint_path, map_location=self.device)
            )
            print("Loaded best model for inference.")
        else:
            print("Warning: No checkpoint found. Using current model state.")

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                cont_data = batch["continuous"].to(self.device)
                cat_data = batch["categorical"].to(self.device)

                outputs = self.model(cont_data, cat_data)
                all_preds.append(outputs.cpu())

        return torch.cat(all_preds).numpy().flatten()


def run_training_pipeline(
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    load_cached_data=True,
):
    """
    Orchestrates the data loading, model initialization, training, and submission generation.
    """
    # 1. Setup
    seed_everything()

    # 2. Data Loading
    print("Preparing DataLoaders...")
    train_loader, val_loader, test_loader = make_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data
    )

    # 3. Model Initialization
    # Get vocab sizes from the dataset attached to the loader
    vocab_sizes = train_loader.dataset.vocab_sizes
    num_continuous = len(CONTINUOUS_FEATURES)

    model = EntityEmbeddingMLP(vocab_sizes=vocab_sizes, num_continuous=num_continuous)

    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.BCELoss()

    # 4. Training
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        patience=PATIENCE,
    )

    trainer.fit(train_loader, val_loader, epochs=epochs)

    # 5. Inference
    print("Generating predictions...")
    predictions = trainer.predict(test_loader)

    # 6. Submission
    # Retrieve IDs from the test dataset
    test_ids = test_loader.dataset.ids

    submission_df = pd.DataFrame({"id": test_ids, "target": predictions})

    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    return trainer.best_auc
