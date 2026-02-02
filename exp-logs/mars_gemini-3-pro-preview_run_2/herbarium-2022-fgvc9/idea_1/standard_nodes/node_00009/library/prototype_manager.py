import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from sklearn.metrics import f1_score
from library.config import SUBMISSION_DIR, NUM_EPOCHS, LEARNING_RATE, NUM_CLASSES
from library.model import PlantClassifier
from library.utils import get_device
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


class SupervisedTrainer:
    """
    A standard supervised trainer for the PlantClassifier.
    Replaces the PrototypeClassifier to allow fine-tuning of the backbone.
    """

    def __init__(self, num_classes=NUM_CLASSES):
        self.device = get_device()
        self.model = PlantClassifier(num_classes=num_classes).to(self.device)
        # Using Label Smoothing to help with high cardinality and noise
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=NUM_EPOCHS
        )
        self.scaler = GradScaler()

    def fit(self, train_loader):
        """
        Trains the model using standard backpropagation with AMP.
        """
        print(f"Starting training for {NUM_EPOCHS} epochs with AMP...")
        self.model.train()

        for epoch in range(NUM_EPOCHS):
            running_loss = 0.0
            correct = 0
            total = 0

            for i, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                # Mixed Precision Training
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            self.scheduler.step()
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            print(
                f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}% - LR: {self.scheduler.get_last_lr()[0]:.6f}"
            )

    def predict(self, loader, is_test=False):
        """
        Performs inference.
        """
        self.model.eval()
        all_preds = []
        all_aux = []

        with torch.no_grad():
            for batch in loader:
                images = batch[0].to(self.device)
                aux = batch[1]

                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().tolist())

                if torch.is_tensor(aux):
                    all_aux.extend(aux.tolist())
                else:
                    all_aux.extend(aux)

        return all_preds, all_aux

    def evaluate(self, val_loader):
        """
        Evaluates the model on the validation set and prints the Macro F1 score.
        """
        print("Evaluating on validation set...")
        preds, labels = self.predict(val_loader, is_test=False)

        # Calculate Macro F1
        score = f1_score(labels, preds, average="macro")
        print(f"Validation Macro F1 Score: {score}")
        return score

    def generate_submission(self, test_loader, idx_to_label=None):
        """
        Generates predictions for the test set and saves submission.csv.
        """
        print("Generating submission...")
        preds, image_ids = self.predict(test_loader, is_test=True)

        if idx_to_label is not None:
            preds = [idx_to_label[p] for p in preds]

        submission_df = pd.DataFrame({"Id": image_ids, "Predicted": preds})

        # Ensure Id is sorted or formatted correctly if needed, though sample submission
        # usually implies just matching IDs. The sample submission has 'Id' as int.
        # Our dataset returns image_id as string or int depending on metadata.
        # Based on sample_submission.csv, Id is int.
        submission_df["Id"] = submission_df["Id"].astype(int)

        output_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
