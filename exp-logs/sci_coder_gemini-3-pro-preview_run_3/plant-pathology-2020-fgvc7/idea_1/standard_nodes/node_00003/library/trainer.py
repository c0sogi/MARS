import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config
from library import utils, data, model


class Trainer:
    """
    Trainer class for the Apple Disease Detection model.
    Handles training, validation, early stopping, and submission generation.
    """

    def __init__(self, debug: bool = Config.DEBUG):
        """
        Initialize the Trainer.

        Args:
            debug (bool): If True, uses a subset of data for faster debugging.
        """
        self.debug = debug
        self.device = utils.get_device()
        self.best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

        # Initialize Model
        self.model = model.AppleDiseaseModel(
            model_name=Config.MODEL_NAME,
            pretrained=True,
            num_classes=Config.NUM_CLASSES,
            dropout_rate=Config.DROPOUT_RATE,
        ).to(self.device)

        # Calculate Class Weights for Loss Function
        # We need to load the training data here to calculate weights
        train_df = pd.read_csv(Config.TRAIN_CSV)
        if self.debug:
            train_df = train_df.sample(
                n=min(50, len(train_df)), random_state=Config.SEED
            ).reset_index(drop=True)

        class_weights = data.calculate_class_weights(train_df).to(self.device)

        # Loss, Optimizer, Scheduler
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler monitors validation AUC (max mode)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=1
        )

        # Early Stopping parameters
        self.patience = Config.PATIENCE
        self.best_auc = 0.0
        self.patience_counter = 0

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def validate(self, val_loader):
        """
        Runs validation and calculates ROC AUC.
        """
        self.model.eval()
        running_loss = 0.0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # TTA: Original (Cite {solution_lesson_node_00002})
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                running_loss += loss.item() * images.size(0)
                probs = torch.softmax(logits, dim=1)

                # TTA: Horizontal Flip
                images_hflip = torch.flip(images, [3])
                logits_hflip = self.model(images_hflip)
                probs_hflip = torch.softmax(logits_hflip, dim=1)

                # TTA: Vertical Flip
                images_vflip = torch.flip(images, [2])
                logits_vflip = self.model(images_vflip)
                probs_vflip = torch.softmax(logits_vflip, dim=1)

                # Average predictions
                avg_probs = (probs + probs_hflip + probs_vflip) / 3.0

                all_probs.append(avg_probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        val_loss = running_loss / len(val_loader.dataset)

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        # Calculate Mean Column-wise ROC AUC
        # multi_class='ovr' calculates AUC for each class against the rest and averages them
        try:
            val_auc = roc_auc_score(
                all_labels, all_probs, multi_class="ovr", average="macro"
            )
        except ValueError:
            # Handle edge cases in debug mode where not all classes might be present
            val_auc = 0.5

        return val_loss, val_auc

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        utils.set_seed(Config.SEED)
        train_loader, val_loader, _ = data.get_dataloaders(debug=self.debug)

        print(f"Starting training on {self.device}...")

        for epoch in range(1, Config.EPOCHS + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_auc = self.validate(val_loader)

            # Print metrics with full precision
            print(
                f"Epoch {epoch}/{Config.EPOCHS} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val AUC: {val_auc}"
            )

            # Update Scheduler
            self.scheduler.step(val_auc)

            # Early Stopping Check
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print("New best model saved.")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        print(f"Training finished. Best Val AUC: {self.best_auc}")

    def generate_submission(self):
        """
        Generates predictions for the test set using the best saved model.
        """
        print("Generating submission...")

        # Load best model weights
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
            print("Loaded best model checkpoint.")
        else:
            print(
                "Warning: Best model checkpoint not found. Using current model weights."
            )

        self.model.eval()

        # Get test loader
        _, _, test_loader = data.get_dataloaders(debug=self.debug)

        results = []

        with torch.no_grad():
            for images, image_ids in test_loader:
                images = images.to(self.device)

                logits = self.model(images)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

                # Append results
                for img_id, prob_vector in zip(image_ids, probs):
                    # Create a dictionary for the row
                    row = {"image_id": img_id}
                    # Map probabilities to class names
                    for idx, class_name in enumerate(Config.CLASSES):
                        row[class_name] = prob_vector[idx]
                    results.append(row)

        # Create DataFrame and save
        submission_df = pd.DataFrame(results)

        # Reorder columns to match sample submission format if necessary
        # Expected: image_id, healthy, multiple_diseases, rust, scab
        cols = ["image_id"] + Config.CLASSES
        submission_df = submission_df[cols]

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
