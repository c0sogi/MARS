import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from library.utils import MetricMonitor
from library.config import Config


def train_one_epoch(model, dataloader, optimizer, device, loss_fn):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: Training dataloader.
        optimizer: Optimizer instance.
        device: Device to train on (cuda/cpu).
        loss_fn: Loss function (HierarchicalLoss).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    monitor = MetricMonitor()

    for batch in dataloader:
        images = batch["image"].to(device)
        targets = {
            "species_id": batch["species_id"].to(device),
            "genus_id": batch["genus_id"].to(device),
            "family_id": batch["family_id"].to(device),
        }

        optimizer.zero_grad()
        outputs = model(images)

        # loss_fn returns (total_loss, metrics_dict)
        loss, loss_dict = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()

        monitor.update("loss", loss.item())

    return monitor.metrics["loss"]["avg"]


def validate(model, dataloader, device, loss_fn):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: Validation dataloader.
        device: Device to evaluate on.
        loss_fn: Loss function.

    Returns:
        tuple: (average_loss, macro_f1_score)
    """
    model.eval()
    monitor = MetricMonitor()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = {
                "species_id": batch["species_id"].to(device),
                "genus_id": batch["genus_id"].to(device),
                "family_id": batch["family_id"].to(device),
            }

            outputs = model(images)
            loss, _ = loss_fn(outputs, targets)

            monitor.update("loss", loss.item())

            # Get species predictions for F1 score
            species_logits = outputs["species"]
            preds = torch.argmax(species_logits, dim=1).cpu().numpy()
            labels = targets["species_id"].cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(labels)

    # Calculate Macro F1
    # Using average='macro' calculates metrics for each label, and finds their unweighted mean.
    f1 = f1_score(all_targets, all_preds, average="macro")

    return monitor.metrics["loss"]["avg"], f1


def predict(model, dataloader, device, output_path=Config.SUBMISSION_PATH):
    """
    Generates predictions for the test set and saves to CSV.

    Args:
        model: The trained PyTorch model.
        dataloader: Test dataloader.
        device: Device to run inference on.
        output_path: Path to save the submission CSV.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            # image_id matches the 'Id' column in submission
            image_ids = batch["image_id"].numpy()

            outputs = model(images)
            species_logits = outputs["species"]
            preds = torch.argmax(species_logits, dim=1).cpu().numpy()

            for img_id, pred in zip(image_ids, preds):
                results.append({"Id": int(img_id), "Predicted": int(pred)})

    df = pd.DataFrame(results)
    # Ensure columns are in correct order
    df = df[["Id", "Predicted"]]
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def train_eval_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    loss_fn,
    epochs,
    patience,
    save_path,
    scheduler=None,
):
    """
    Runs the training and validation loop with early stopping.

    Args:
        model: The PyTorch model.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        optimizer: Optimizer.
        device: Device.
        loss_fn: Loss function.
        epochs: Maximum number of epochs.
        patience: Early stopping patience.
        save_path: Path to save the best model checkpoint.
        scheduler: Learning rate scheduler (optional).
    """
    best_f1 = -1.0
    patience_counter = 0

    print(f"Starting training for {epochs} epochs with patience {patience}...")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        val_loss, val_f1 = validate(model, val_loader, device, loss_fn)

        if scheduler:
            # Handle ReduceLROnPlateau which requires a metric
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_f1)
            else:
                scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val F1: {val_f1}"
        )

        # Save best model based on Macro F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path} (F1: {best_f1})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    print(f"Training finished. Best Val F1: {best_f1}")
