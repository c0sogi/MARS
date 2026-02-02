import math
import sys
import time
import torch
import pandas as pd
import os
import numpy as np
from library.utils import Averager, format_prediction_string
from library.config import Config


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=3, min_delta=0.001, path=Config.MODEL_SAVE_PATH):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """Saves model when validation loss decrease."""
        torch.save(model.state_dict(), self.path)
        print(f"Validation loss decreased. Saving model to {self.path}")


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    """
    Trains the model for one epoch.
    """
    model.train()
    loss_hist = Averager()

    print(f"Epoch: {epoch+1}")

    for step, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)

        # Move targets to device
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_hist.send(loss_value)

        if step % print_freq == 0:
            print(
                f"Epoch: {epoch+1} Step: {step}/{len(data_loader)} Loss: {loss_value}"
            )

    return loss_hist.value


@torch.no_grad()
def evaluate_loss(model, data_loader, device):
    """
    Evaluates the model on the validation set by calculating loss.
    Note: torchvision Faster R-CNN only returns losses in train mode.
    We use train mode with no_grad to estimate validation loss.
    """
    model.train()  # Keep in train mode to get loss dictionary
    loss_hist = Averager()

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_hist.send(losses.item())

    return loss_hist.value


def inference(model, data_loader, device):
    """
    Runs inference on the test set and generates the submission dataframe.
    """
    model.eval()
    results = []

    print("Starting inference on test set...")

    with torch.no_grad():
        for images, targets in data_loader:
            # Move images to device
            images = list(image.to(device) for image in images)

            # Forward pass
            outputs = model(images)

            # Process batch
            for i, output in enumerate(outputs):
                image_id = targets[i]["image_id"]

                # Extract predictions
                boxes = output["boxes"]
                scores = output["scores"]
                labels = output["labels"]

                # Format prediction string
                pred_string = format_prediction_string(boxes, scores, labels)

                results.append({"image_id": image_id, "PredictionString": pred_string})

    df_submission = pd.DataFrame(results)
    return df_submission


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    patience=3,
):
    """
    Main training loop with early stopping.
    """
    early_stopping = EarlyStopping(patience=patience, path=Config.MODEL_SAVE_PATH)

    print(f"Starting training for {num_epochs} epochs on device: {device}")

    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

        # Validate
        val_loss = evaluate_loss(model, val_loader, device)

        # Scheduler Step
        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch #{epoch+1} Summary: Train Loss: {train_loss} Val Loss: {val_loss} LR: {current_lr}"
        )

        # Check Early Stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Load best model weights
    print(f"Loading best model weights from {Config.MODEL_SAVE_PATH}")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    return model
