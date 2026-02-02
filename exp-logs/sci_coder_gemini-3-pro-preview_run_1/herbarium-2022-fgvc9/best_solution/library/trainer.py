import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library import config
from library import utils


class Trainer:
    """
    Trainer class to handle model training, validation, and inference.
    """

    def __init__(self, model, device=config.DEVICE):
        """
        Args:
            model (nn.Module): The PyTorch model to train.
            device (torch.device): Device to run training on.
        """
        self.model = model
        self.device = device
        self.model.to(self.device)

        # Initialize Loss with Class Weights
        # Weights are calculated based on training data imbalance to optimize Macro F1
        class_weights = utils.calculate_class_weights()
        # Added label_smoothing to improve generalization on high-cardinality data
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device), label_smoothing=0.1
        )

        # Initialize Optimizer
        # Differential Learning Rates: Lower for backbone, higher for head
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "fc" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        params_list = [
            {"params": backbone_params, "lr": config.BACKBONE_LR},
            {"params": head_params, "lr": config.LEARNING_RATE},
        ]

        self.optimizer = optim.Adam(params_list, weight_decay=config.WEIGHT_DECAY)

        # Initialize Cosine Annealing Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.NUM_EPOCHS
        )

    def train_epoch(self, train_loader):
        """
        Performs one epoch of training.

        Args:
            train_loader (DataLoader): DataLoader for training data.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.

        Args:
            val_loader (DataLoader): DataLoader for validation data.

        Returns:
            tuple: (validation_loss, macro_f1_score)
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                batch_size = images.size(0)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                # Get predictions for F1 score calculation
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        epoch_loss = running_loss / dataset_size

        # Concatenate all batches
        if all_preds:
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
        else:
            all_preds = np.array([])
            all_labels = np.array([])

        macro_f1 = utils.calculate_macro_f1(all_labels, all_preds)

        return epoch_loss, macro_f1

    def fit(self, train_loader, val_loader, num_epochs=config.NUM_EPOCHS):
        """
        Main training loop with Early Stopping.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            num_epochs (int): Maximum number of epochs to train.
        """
        print(f"Starting training for {num_epochs} epochs on {self.device}...")

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_f1 = self.validate(val_loader)

            # Step the scheduler
            self.scheduler.step()

            # Print metrics with full precision
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss}")
            print(f"Val Loss: {val_loss}")
            print(f"Val Macro F1: {val_f1}")

            # Early Stopping Logic based on Validation Loss
            if val_loss < best_val_loss - config.MIN_DELTA:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), config.MODEL_CHECKPOINT_PATH)
                print(
                    f"Validation loss improved. Model saved to {config.MODEL_CHECKPOINT_PATH}"
                )
            else:
                patience_counter += 1
                print(
                    f"No improvement in validation loss. Patience: {patience_counter}/{config.PATIENCE}"
                )
                if patience_counter >= config.PATIENCE:
                    print("Early stopping triggered.")
                    break

        print("Training completed.")

    def predict(self, test_loader):
        """
        Generates predictions for the test set and saves to submission file.

        Args:
            test_loader (DataLoader): DataLoader for test data.
        """
        print("Starting inference on test set...")

        # Load the best model weights
        if os.path.exists(config.MODEL_CHECKPOINT_PATH):
            print(f"Loading model weights from {config.MODEL_CHECKPOINT_PATH}")
            self.model.load_state_dict(
                torch.load(config.MODEL_CHECKPOINT_PATH, map_location=self.device)
            )
        else:
            print("Warning: No checkpoint found. Using current model weights.")

        self.model.eval()

        # Get mapping to convert model indices back to category_ids
        _, idx_to_label = utils.get_class_mappings()

        ids_list = []
        predictions_list = []

        with torch.no_grad():
            for images, image_ids in test_loader:
                images = images.to(self.device)

                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                # Convert model indices to original category_ids
                mapped_preds = [idx_to_label[idx] for idx in preds]

                ids_list.extend(image_ids)
                predictions_list.extend(mapped_preds)

        # Create submission DataFrame
        submission_df = pd.DataFrame({"Id": ids_list, "Predicted": predictions_list})

        # Save to CSV
        output_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
