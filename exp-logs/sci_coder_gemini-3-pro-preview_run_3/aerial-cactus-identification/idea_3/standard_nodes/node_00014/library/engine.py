import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from library.utils import mixup_data, mixup_criterion
from library.config import DEVICE, MIXUP_ALPHA


def train_one_epoch(model, loader, criterion, optimizer, device, alpha=1.0):
    """
    Trains the model for one epoch using Mixup augmentation.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # Apply Mixup
        inputs, targets_a, targets_b, lam = mixup_data(images, labels, alpha, device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # Squeeze outputs to match label shape if necessary (B, 1) -> (B)
        outputs = outputs.squeeze()

        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            outputs = outputs.squeeze()

            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(outputs)

            all_targets.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    avg_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Calculate AUC
    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle case where only one class is present in batch (rare but possible in small debug runs)
        auc_score = 0.5

    return avg_loss, auc_score


class Trainer:
    """
    Manages the training process, including early stopping and model checkpointing.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        patience,
        save_path,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.save_path = save_path

        self.best_model_state = None
        self.best_score = -np.inf
        self.early_stop_counter = 0

    def fit(self, num_epochs):
        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                alpha=MIXUP_ALPHA,
            )

            val_loss, val_auc = validate(
                self.model, self.val_loader, self.criterion, self.device
            )

            # Step the scheduler
            if self.scheduler:
                self.scheduler.step()

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val AUC: {val_auc}"
            )

            # Early Stopping and Checkpointing (Maximize AUC)
            if val_auc > self.best_score:
                self.best_score = val_auc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.early_stop_counter = 0

                # Save best model to disk immediately
                torch.save(self.best_model_state, self.save_path)
            else:
                self.early_stop_counter += 1

            if self.early_stop_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"Training complete. Best Val AUC: {self.best_score}")

        # Load best weights into model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return self.best_score


def predict_with_tta(model, loader, device):
    """
    Generates predictions using Test-Time Augmentation (Original + Horizontal Flip + Vertical Flip).
    Returns a numpy array of probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # 1. Original
            out1 = torch.sigmoid(model(images).squeeze())

            # 2. Horizontal Flip
            images_h = torch.flip(images, dims=[3])
            out2 = torch.sigmoid(model(images_h).squeeze())

            # 3. Vertical Flip
            images_v = torch.flip(images, dims=[2])
            out3 = torch.sigmoid(model(images_v).squeeze())

            # Average predictions
            avg_out = (out1 + out2 + out3) / 3.0

            all_preds.append(avg_out.cpu().numpy())

    return np.concatenate(all_preds)


def generate_submission(models, test_loader, test_ids, output_path):
    """
    Generates the submission file by averaging predictions from an ensemble of models.

    Args:
        models (list): List of trained PyTorch models.
        test_loader (DataLoader): DataLoader for the test set.
        test_ids (np.ndarray): Array of test image IDs.
        output_path (str): Path to save the CSV file.
    """
    print(f"Generating predictions with ensemble of {len(models)} models using TTA...")

    ensemble_preds = np.zeros(len(test_ids))

    for i, model in enumerate(models):
        print(f"Inference for model {i+1}...")
        preds = predict_with_tta(model, test_loader, DEVICE)
        ensemble_preds += preds

    # Average over models
    ensemble_preds /= len(models)

    # Create DataFrame
    df = pd.DataFrame({"id": test_ids, "has_cactus": ensemble_preds})

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
