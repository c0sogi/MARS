import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import AverageMeter, LaplaceLogLikelihoodLoss


def train_fn(dataloader, model, optimizer, device, scheduler=None):
    """
    Performs one epoch of training.

    Args:
        dataloader: PyTorch DataLoader for training data.
        model: The PGA-Net model.
        optimizer: The optimizer (AdamW).
        device: 'cuda' or 'cpu'.
        scheduler: Learning rate scheduler (optional).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()
    loss_fn = LaplaceLogLikelihoodLoss()

    for batch in dataloader:
        # Move inputs to device
        img_ax = batch["image_axial"].to(device)
        img_cor = batch["image_coronal"].to(device)
        tabular = batch["tabular"].to(device)
        meta = batch["meta"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass
        pred_fvc, pred_sigma = model(img_ax, img_cor, tabular, meta)

        # Compute loss
        loss = loss_fn(pred_fvc, pred_sigma, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        loss_meter.update(loss.item(), img_ax.size(0))

    return loss_meter.avg


def eval_fn(dataloader, model, device):
    """
    Evaluates the model on the validation set.

    Args:
        dataloader: PyTorch DataLoader for validation data.
        model: The PGA-Net model.
        device: 'cuda' or 'cpu'.

    Returns:
        float: Average loss (metric) for the validation set.
    """
    model.eval()
    loss_meter = AverageMeter()
    loss_fn = LaplaceLogLikelihoodLoss()

    with torch.no_grad():
        for batch in dataloader:
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            meta = batch["meta"].to(device)
            target = batch["target"].to(device)

            pred_fvc, pred_sigma = model(img_ax, img_cor, tabular, meta)

            loss = loss_fn(pred_fvc, pred_sigma, target)
            loss_meter.update(loss.item(), img_ax.size(0))

    return loss_meter.avg


def run_training(
    model, train_loader, val_loader, optimizer, scheduler, device, num_epochs, patience
):
    """
    Orchestrates the training loop with early stopping.

    Args:
        model: The neural network.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device to train on.
        num_epochs: Maximum number of epochs.
        patience: Early stopping patience.

    Returns:
        str: Path to the best saved model.
    """
    best_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training on device: {device}")

    for epoch in range(num_epochs):
        train_loss = train_fn(train_loader, model, optimizer, device, scheduler)
        val_loss = eval_fn(val_loader, model, device)

        # Step the scheduler (Cosine Annealing is typically stepped per epoch)
        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Early Stopping Logic
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    print(f"Training complete. Best Validation Loss: {best_loss}")
    return best_model_path


def generate_submission(model, test_loader, device, output_path=None):
    """
    Generates predictions for the test set and saves to CSV.

    Args:
        model: Trained model.
        test_loader: DataLoader for the test set.
        device: Device to run inference on.
        output_path: Path to save the submission CSV.
    """
    if output_path is None:
        output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    model.eval()
    patient_weeks = []
    fvc_preds = []
    conf_preds = []

    print("Generating submission...")

    with torch.no_grad():
        for batch in test_loader:
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            meta = batch["meta"].to(device)
            # batch['patient_week'] is a list of strings
            p_weeks = batch["patient_week"]

            pred_fvc, pred_sigma = model(img_ax, img_cor, tabular, meta)

            # Move to CPU and numpy
            pred_fvc = pred_fvc.cpu().numpy()
            pred_sigma = pred_sigma.cpu().numpy()

            patient_weeks.extend(p_weeks)
            fvc_preds.extend(pred_fvc)
            conf_preds.extend(pred_sigma)

    # Create DataFrame
    df_sub = pd.DataFrame(
        {"Patient_Week": patient_weeks, "FVC": fvc_preds, "Confidence": conf_preds}
    )

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
