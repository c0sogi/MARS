import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from library.model import AsymmetricEfficientNet
from library.dataset import get_dataloader
from library.utils import seed_everything


class Trainer:
    def __init__(
        self, device="cuda" if torch.cuda.is_available() else "cpu", config=None
    ):
        """
        Trainer class for the MGMT promoter methylation prediction task.

        Args:
            device (str): Computation device ('cuda' or 'cpu').
            config (dict): Hyperparameters including learning_rate, weight_decay, etc.
        """
        self.device = device
        self.config = config if config else {}

        # Hyperparameters
        self.lr = self.config.get("learning_rate", 1e-4)
        self.weight_decay = self.config.get("weight_decay", 1e-2)
        self.checkpoint_dir = self.config.get("checkpoint_dir", "./working/idea_12")

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Model Initialization
        self.model = AsymmetricEfficientNet(num_classes=1, pretrained=True)
        self.model.to(self.device)

        # Optimizer: AdamW with aggressive weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Loss Function: BCEWithLogitsLoss
        self.criterion = nn.BCEWithLogitsLoss()

    def train_one_epoch(self, dataloader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for images, targets in dataloader:
            images = images.to(self.device, dtype=torch.float32)
            targets = targets.to(self.device, dtype=torch.float32)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss

    def evaluate(self, dataloader):
        """
        Evaluates the model on the validation set.
        Returns average loss and AUC score.
        """
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device, dtype=torch.float32)
                targets = targets.to(self.device, dtype=torch.float32)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * images.size(0)

                # Apply sigmoid for probabilities
                probs = torch.sigmoid(outputs)

                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(probs.cpu().numpy())

        val_loss = running_loss / len(dataloader.dataset)

        # Handle case where only one class is present in batch (though unlikely with full val set)
        try:
            val_auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            val_auc = 0.5

        return val_loss, val_auc

    def fit(self, train_loader, val_loader, epochs=10, patience=5):
        """
        Main training loop with Early Stopping.
        """
        best_auc = 0.0
        patience_counter = 0
        best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")

        print(f"Starting training on device: {self.device}")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_auc = self.evaluate(val_loader)

            print(
                f"Epoch {epoch}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val AUC: {val_auc}"
            )

            # Early Stopping Logic (Maximize AUC)
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                print(f"New best model saved with AUC: {val_auc}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        print(f"Training complete. Best Val AUC: {best_auc}")
        return best_auc

    def predict(self, test_loader, output_path="./submission/submission.csv"):
        """
        Generates predictions for the test set using Test-Time Augmentation (TTA).
        Saves the result to a CSV file.
        """
        best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        if not os.path.exists(best_model_path):
            print(
                "No trained model found. Using current model state (warning: might be untrained)."
            )
        else:
            print(f"Loading best model from {best_model_path}")
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )

        self.model.eval()

        results = []
        # Retrieve IDs from the dataset dataframe
        # Assuming test_loader is not shuffled, which is standard for test/inference
        test_ids = test_loader.dataset.df["BraTS21ID"].values

        # Index to track current ID
        current_idx = 0

        print("Starting inference with TTA (Original + HFlip + VFlip)...")

        with torch.no_grad():
            for images in test_loader:
                # Handle case where dataloader returns (images, targets) or just images
                if isinstance(images, (list, tuple)):
                    images = images[0]

                images = images.to(self.device, dtype=torch.float32)
                batch_size = images.size(0)

                # 1. Original Prediction
                out_orig = self.model(images)
                prob_orig = torch.sigmoid(out_orig)

                # 2. Horizontal Flip TTA (dim 3 is width)
                images_h = torch.flip(images, dims=[3])
                out_h = self.model(images_h)
                prob_h = torch.sigmoid(out_h)

                # 3. Vertical Flip TTA (dim 2 is height)
                images_v = torch.flip(images, dims=[2])
                out_v = self.model(images_v)
                prob_v = torch.sigmoid(out_v)

                # Average Probabilities
                avg_probs = (prob_orig + prob_h + prob_v) / 3.0
                avg_probs = avg_probs.cpu().numpy().flatten()

                # Map to IDs
                for i in range(batch_size):
                    if current_idx < len(test_ids):
                        results.append(
                            {
                                "BraTS21ID": test_ids[current_idx],
                                "MGMT_value": avg_probs[i],
                            }
                        )
                        current_idx += 1

        # Save submission
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        submission_df = pd.DataFrame(results)
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")


def run_training_pipeline(
    epochs=15, batch_size=32, input_root="./input", metadata_dir="./metadata"
):
    """
    Helper function to run the full training and inference pipeline.
    """
    seed_everything(42)

    # Load Metadata
    train_df = pd.read_csv(os.path.join(metadata_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(metadata_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(metadata_dir, "test.csv"))

    # Create DataLoaders
    # Note: cache_dir is set to ./working/idea_12 to persist ROI calculations
    train_loader = get_dataloader(
        train_df, phase="train", batch_size=batch_size, input_root=input_root
    )
    val_loader = get_dataloader(
        val_df, phase="valid", batch_size=batch_size, input_root=input_root
    )
    test_loader = get_dataloader(
        test_df, phase="test", batch_size=batch_size, input_root=input_root
    )

    # Initialize Trainer
    config = {
        "learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "checkpoint_dir": "./working/idea_12",
    }
    trainer = Trainer(config=config)

    # Fit Model
    trainer.fit(train_loader, val_loader, epochs=epochs, patience=5)

    # Predict on Test Set
    trainer.predict(test_loader, output_path="./submission/submission.csv")
