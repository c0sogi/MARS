import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything, get_optimizer_params
from library.data import get_dataloaders
from library.model import HybridResFunnel


class Trainer:
    """
    Manages the training, validation, and inference lifecycle of the
    Hybrid Post-Norm SwiGLU-ResFunnel Network.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)

        # Initialize Model
        self.model = HybridResFunnel(
            vocab_size=Config.VOCAB_SIZE,
            embedding_dim=Config.EMBEDDING_DIM,
            seq_len=Config.SEQUENCE_LENGTH,
            num_cont=Config.NUM_CONT_FEATURES,
            transformer_layers=Config.TRANSFORMER_LAYERS,
            transformer_heads=Config.TRANSFORMER_HEADS,
            transformer_dropout=Config.TRANSFORMER_DROPOUT,
            backbone_stages=Config.BACKBONE_STAGES,
            backbone_blocks=Config.BACKBONE_BLOCKS,
            backbone_dropout=Config.BACKBONE_DROPOUT,
            stochastic_depth_max=Config.STOCHASTIC_DEPTH_MAX,
        ).to(self.device)

        # Optimizer with decoupled weight decay
        optimizer_params = get_optimizer_params(
            self.model, weight_decay=Config.WEIGHT_DECAY
        )
        self.optimizer = torch.optim.AdamW(optimizer_params, lr=Config.LEARNING_RATE)

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=Config.SCHEDULER_STEP_SIZE,
            gamma=Config.SCHEDULER_GAMMA,
        )

        # Loss Function (Binary Cross Entropy)
        self.criterion = nn.BCELoss()

        # Metric Tracking
        self.best_auc = 0.0

    def train_one_epoch(self, train_loader):
        """
        Executes one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        num_samples = 0

        for cat, cont, target in train_loader:
            cat = cat.to(self.device)
            cont = cont.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(cat, cont).squeeze()
            loss = self.criterion(outputs, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * cat.size(0)
            num_samples += cat.size(0)

        avg_loss = total_loss / num_samples
        return avg_loss

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set using AUC.
        """
        self.model.eval()
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for cat, cont, target in val_loader:
                cat = cat.to(self.device)
                cont = cont.to(self.device)
                target = target.to(self.device)

                outputs = self.model(cat, cont).squeeze()

                all_preds.append(outputs.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        auc = roc_auc_score(all_targets, all_preds)
        return auc

    def fit(self):
        """
        Main training loop with Early Stopping and Checkpointing.
        """
        # Get DataLoaders
        train_loader, val_loader, test_loader = get_dataloaders(
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            debug=Config.DEBUG,
        )

        print(f"Starting training on {self.device} for {Config.EPOCHS} epochs.")

        # Early Stopping parameters
        patience = 10
        counter = 0

        for epoch in range(1, Config.EPOCHS + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_auc = self.validate(val_loader)

            # Step scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # Print metrics with full precision
            print(
                f"Epoch {epoch} | LR: {current_lr} | Train Loss: {train_loss} | Val AUC: {val_auc}"
            )

            # Checkpoint and Early Stopping Logic
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"New best model saved with AUC: {self.best_auc}")
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        print(f"Training complete. Best Val AUC: {self.best_auc}")
        return test_loader

    def predict(self, test_loader):
        """
        Generates predictions for the test set using the best saved model.
        """
        print("Loading best model for inference...")
        if not os.path.exists(Config.MODEL_SAVE_PATH):
            print("No model checkpoint found. Skipping prediction.")
            return

        self.model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
        )
        self.model.eval()

        all_preds = []

        with torch.no_grad():
            for cat, cont in test_loader:
                cat = cat.to(self.device)
                cont = cont.to(self.device)

                outputs = self.model(cat, cont).squeeze()

                # Handle potential scalar output for batch size 1
                if outputs.ndim == 0:
                    outputs = outputs.unsqueeze(0)

                all_preds.append(outputs.cpu().numpy())

        all_preds = np.concatenate(all_preds)

        # Prepare submission dataframe
        # We rely on test_metadata.csv for IDs to ensure alignment
        test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

        # If in debug mode, test_loader is truncated, so we must truncate metadata too
        if Config.DEBUG:
            test_meta = test_meta.iloc[: len(all_preds)]

        submission = pd.DataFrame({"id": test_meta["id"], "target": all_preds})

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    """
    Orchestrates the training and prediction pipeline.
    """
    # Setup directories
    Config.setup()

    # Reproducibility
    seed_everything(Config.SEED)

    # Initialize Trainer
    trainer = Trainer()

    # Train
    test_loader = trainer.fit()

    # Predict
    trainer.predict(test_loader)
