import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.dataset import get_dataloaders
from library.architecture import SwishGatedResFunnel, DEVICE, SEED

# Set fixed seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class Trainer:
    """
    Manages the training, validation, and inference lifecycle for the Swish-Gated ResFunnel model.
    """

    def __init__(self, model_class=SwishGatedResFunnel, device=DEVICE):
        self.device = device
        self.model = model_class().to(self.device)
        self.criterion = nn.BCELoss()
        self.best_model_path = "./working/best_model.pth"
        os.makedirs("./working", exist_ok=True)

    def _configure_optimizers(self, learning_rate=1e-3, weight_decay=1e-2):
        """
        Configures AdamW with Decoupled Weight Decay.
        Group 1: Decay applied (Weights of Linear, Embedding, Attention)
        Group 2: No Decay (Biases, LayerNorm, Positional Embeddings)
        """
        param_groups = [
            {"params": [], "weight_decay": weight_decay},
            {"params": [], "weight_decay": 0.0},
        ]

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Identify parameters that should not decay
            # 1. Biases
            # 2. Normalization parameters (weights/biases in LayerNorm)
            # 3. Positional Embeddings
            # 4. 1D parameters (often biases or scale factors)
            if (
                param.ndim <= 1
                or name.endswith(".bias")
                or "norm" in name
                or "pos_embed" in name
            ):
                param_groups[1]["params"].append(param)
            else:
                param_groups[0]["params"].append(param)

        optimizer = optim.AdamW(param_groups, lr=learning_rate)
        return optimizer

    def _train_epoch(self, loader, optimizer):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            num_x = batch["numerical"].to(self.device)
            cat_x = batch["categorical"].to(self.device)
            target = batch["target"].to(self.device)

            optimizer.zero_grad()
            output = self.model(num_x, cat_x)
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * num_x.size(0)

        return total_loss / len(loader.dataset)

    def _validate(self, loader):
        """
        Evaluates the model on the validation set and returns AUC.
        """
        self.model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in loader:
                num_x = batch["numerical"].to(self.device)
                cat_x = batch["categorical"].to(self.device)
                target = batch["target"]

                output = self.model(num_x, cat_x)
                val_preds.append(output.cpu().numpy())
                val_targets.append(target.numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)

        return roc_auc_score(val_targets, val_preds)

    def fit(self, epochs=40, batch_size=1024, patience=10, learning_rate=1e-3):
        """
        Executes the training pipeline with Early Stopping.
        """
        print(f"Starting training on {self.device}...")

        # Load Data
        train_loader, val_loader, _, _ = get_dataloaders(
            batch_size=batch_size, num_workers=4, load_cached_data=True
        )

        # Setup Optimizer and Scheduler
        optimizer = self._configure_optimizers(learning_rate=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        best_auc = 0.0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer)
            scheduler.step()

            val_auc = self._validate(val_loader)

            # Print full precision as requested
            print(f"Epoch {epoch}: Train Loss = {train_loss}, Val AUC = {val_auc}")

            # Early Stopping and Checkpointing
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(self.model.state_dict(), self.best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        print(f"Training complete. Best Val AUC: {best_auc}")

    def predict(self, batch_size=1024):
        """
        Loads the best model, generates predictions for the test set, and saves the submission file.
        """
        print("Starting inference on test set...")

        if not os.path.exists(self.best_model_path):
            raise FileNotFoundError("Best model not found. Run fit() first.")

        self.model.load_state_dict(
            torch.load(self.best_model_path, map_location=self.device)
        )
        self.model.eval()

        # Get test loader
        _, _, test_loader, test_ids = get_dataloaders(
            batch_size=batch_size, num_workers=4, load_cached_data=True
        )

        test_preds = []
        with torch.no_grad():
            for batch in test_loader:
                num_x = batch["numerical"].to(self.device)
                cat_x = batch["categorical"].to(self.device)
                output = self.model(num_x, cat_x)
                test_preds.append(output.cpu().numpy())

        test_preds = np.concatenate(test_preds).flatten()

        # Save submission
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        df_sub = pd.DataFrame({"id": test_ids, "target": test_preds})
        df_sub.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")


def run_training_pipeline(epochs=40, batch_size=1024, patience=10):
    """
    Helper function to instantiate the trainer and run the full pipeline.
    """
    trainer = Trainer()
    trainer.fit(epochs=epochs, batch_size=batch_size, patience=patience)
    trainer.predict(batch_size=batch_size)
