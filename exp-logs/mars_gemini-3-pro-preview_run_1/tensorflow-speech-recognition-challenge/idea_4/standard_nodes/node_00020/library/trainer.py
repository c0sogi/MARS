import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from library.config import Config
from library.utils import mixup_data, mixup_criterion


class Trainer:
    """
    Trainer class for Dilated EfficientNet-B2 Speech Command Recognition.
    Encapsulates training, validation, early stopping, and submission generation.
    """

    def __init__(self, model, train_loader, val_loader, config: Config):
        """
        Args:
            model (nn.Module): The neural network model.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            config (Config): Configuration object containing hyperparameters.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(config.device)

        # Move model to the appropriate device
        self.model.to(self.device)

        # Define Loss Function
        self.criterion = nn.CrossEntropyLoss()

        # Define Optimizer (AdamW)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Define Scheduler (Cosine Annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=config.min_lr
        )

        # Training State
        self.best_acc = 0.0
        self.patience = 10  # Epochs to wait before early stopping
        self.counter = 0
        self.best_model_path = os.path.join(self.config.working_dir, "best_model.pth")

    def train_epoch(self, epoch):
        """
        Executes one epoch of training with Mixup augmentation.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0

        for i, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Apply Mixup Augmentation
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, targets, self.config.mixup_alpha, self.device
            )

            # Forward Pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Compute Mixup Loss
            loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)

            # Backward Pass and Optimization
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        """
        Evaluates the model on the validation set.

        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward Pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()

                # Calculate Accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = running_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def fit(self):
        """
        Main training loop handling epochs, logging, checkpointing, and early stopping.
        """
        print(f"Starting training on {self.device} for {self.config.epochs} epochs.")

        for epoch in range(1, self.config.epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc = self.validate()

            # Step Scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch} | LR: {current_lr} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Acc: {val_acc}"
            )

            # Checkpoint and Early Stopping Logic
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.counter = 0
                # Save Best Model
                torch.save(self.model.state_dict(), self.best_model_path)
                print(
                    f"Validation accuracy improved to {val_acc}. Model saved to {self.best_model_path}"
                )
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(
                        f"Early stopping triggered. No improvement for {self.patience} epochs."
                    )
                    break

        print(f"Training finished. Best Validation Accuracy: {self.best_acc}")

    def generate_submission(self, test_loader):
        """
        Generates predictions for the test set using the best saved model.
        Saves the result to ./submission/submission.csv and the working directory.

        Args:
            test_loader (DataLoader): DataLoader for the test set.
        """
        print("Generating submission...")

        # Load Best Weights
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
            print(f"Loaded best model weights from {self.best_model_path}")
        else:
            print(
                "Warning: Best model checkpoint not found. Using current model state."
            )

        self.model.eval()
        predictions = []

        # Get ID to Label Map
        id2label = self.config.get_id_map()

        # Inference Loop
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)

                # Get predicted class indices
                _, predicted_ids = outputs.max(1)
                predictions.extend(predicted_ids.cpu().numpy())

        # Retrieve Filenames from Dataset
        # Assumes test_loader.dataset is a SpeechCommandDataset with a 'df' attribute
        if hasattr(test_loader.dataset, "df"):
            test_df = test_loader.dataset.df
            filenames = test_df["filepath"].apply(os.path.basename).tolist()
        else:
            raise AttributeError(
                "Test dataset does not have a 'df' attribute to retrieve filenames."
            )

        if len(predictions) != len(filenames):
            print(
                f"Warning: Mismatch between predictions ({len(predictions)}) and filenames ({len(filenames)})."
            )

        # Map IDs to Labels
        pred_labels = [id2label[p] for p in predictions]

        # Create DataFrame
        submission_df = pd.DataFrame({"fname": filenames, "label": pred_labels})

        # Define Output Paths
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        final_path = os.path.join(submission_dir, "submission.csv")
        working_path = os.path.join(self.config.working_dir, "submission.csv")

        # Save CSV
        submission_df.to_csv(final_path, index=False)
        submission_df.to_csv(working_path, index=False)

        print(f"Submission saved to {final_path}")
