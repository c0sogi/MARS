import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import AverageMeter, alaska_weighted_auc


def train_one_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """
    Performs one epoch of training.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to run on.
        scheduler (LRScheduler, optional): Learning rate scheduler.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        # Model output shape is (Batch_Size, 1), squeeze to (Batch_Size,)
        outputs = model(images).squeeze(1)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Step scheduler after every batch if provided (e.g., OneCycleLR)
        if scheduler is not None:
            scheduler.step()

        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def validate(model, dataloader, criterion, device):
    """
    Validates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run on.

    Returns:
        tuple: (Average Loss, Weighted AUC Score)
    """
    model.eval()
    loss_meter = AverageMeter()

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

            loss_meter.update(loss.item(), images.size(0))

            # Apply sigmoid to get probabilities [0, 1]
            probs = torch.sigmoid(outputs)

            preds_list.extend(probs.cpu().numpy())
            targets_list.extend(labels.cpu().numpy())

    # Convert to numpy arrays for metric calculation
    preds_arr = np.array(preds_list)
    targets_arr = np.array(targets_list)

    # Calculate Weighted AUC
    try:
        auc_score = alaska_weighted_auc(targets_arr, preds_arr)
    except Exception as e:
        print(f"Warning: Could not calculate AUC. Error: {e}")
        auc_score = 0.0

    return loss_meter.avg, auc_score


def inference_tta(model, dataloader, device):
    """
    Performs inference with Test-Time Augmentation (Horizontal Flip).

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Test data loader (returns image, image_id).
        device (torch.device): Device to run on.

    Returns:
        pd.DataFrame: DataFrame containing 'Id' and 'Label' (probability).
    """
    model.eval()
    results = []

    with torch.no_grad():
        for images, image_ids in dataloader:
            images = images.to(device)

            # 1. Original Image Prediction
            out_orig = model(images).squeeze(1)
            prob_orig = torch.sigmoid(out_orig)

            # 2. Flipped Image Prediction (Horizontal Flip)
            # Input is (B, C, H, W), flip along width (dim 3)
            images_flip = torch.flip(images, dims=[3])
            out_flip = model(images_flip).squeeze(1)
            prob_flip = torch.sigmoid(out_flip)

            # 3. Average Probabilities
            avg_probs = (prob_orig + prob_flip) / 2.0

            # Store results
            avg_probs_np = avg_probs.cpu().numpy()

            for img_id, prob in zip(image_ids, avg_probs_np):
                results.append({"Id": img_id, "Label": prob})

    return pd.DataFrame(results)


class StegoEngine:
    """
    Engine class to handle training, validation, and submission.
    """

    def __init__(self, model, device, optimizer=None, scheduler=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Define Loss Function with Class Balancing
        # The dataset has 1 Cover : 3 Stego.
        # We weight the positive class (Stego) by 0.33 to balance the loss contribution.
        pos_weight_val = torch.tensor([Config.POS_WEIGHT], device=device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)

    def train_model(
        self, train_loader, val_loader, epochs=Config.NUM_EPOCHS, patience=5
    ):
        """
        Runs the full training loop with Early Stopping.

        Args:
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
            epochs (int): Maximum number of epochs.
            patience (int): Early stopping patience.
        """
        best_auc = -1.0
        patience_counter = 0

        print(f"Starting training on device: {self.device}")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Train
            train_loss = train_one_epoch(
                self.model,
                train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                self.scheduler,
            )

            # Validate
            val_loss, val_auc = validate(
                self.model, val_loader, self.criterion, self.device
            )

            # Print Metrics (Full Precision)
            print(f"Train Loss: {train_loss}")
            print(f"Val Loss: {val_loss}")
            print(f"Val Weighted AUC: {val_auc}")

            # Early Stopping Check
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                # Save Best Model
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                print(f"New best model saved to {Config.BEST_MODEL_PATH}")
            else:
                patience_counter += 1
                print(f"No improvement in AUC. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered. Training finished.")
                break

    def generate_submission(self, test_loader):
        """
        Loads the best model and generates the submission file.

        Args:
            test_loader (DataLoader): Test data loader.
        """
        print("\nGenerating submission...")

        # Load Best Model Weights
        if os.path.exists(Config.BEST_MODEL_PATH):
            print(f"Loading model weights from {Config.BEST_MODEL_PATH}")
            state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(
                "Warning: Best model checkpoint not found. Using current model weights."
            )

        # Run Inference
        df_submission = inference_tta(self.model, test_loader, self.device)

        # Sort by Id
        df_submission = df_submission.sort_values(by="Id")

        # Save to CSV
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
