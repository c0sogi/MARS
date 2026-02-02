import os
import torch
import pandas as pd
from library import utils
from library import config
from library import dataset


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """
    Performs one epoch of training.

    Args:
        model: The neural network model.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on ('cpu' or 'cuda').
        scaler: GradScaler for mixed precision training.

    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.train()
    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(images)
            loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Calculate accuracy
        # outputs shape: [batch_size, num_classes]
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == targets).float().mean().item()

        # Update meters
        losses.update(loss.item(), images.size(0))
        accuracies.update(acc, images.size(0))

    print(f"Train Loss: {losses.avg}")
    print(f"Train Accuracy: {accuracies.avg}")
    return losses.avg, accuracies.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The neural network model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.eval()
    losses = utils.AverageMeter()
    accuracies = utils.AverageMeter()

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            preds = torch.argmax(outputs, dim=1)
            acc = (preds == targets).float().mean().item()

            losses.update(loss.item(), images.size(0))
            accuracies.update(acc, images.size(0))

    print(f"Val Loss: {losses.avg}")
    print(f"Val Accuracy: {accuracies.avg}")
    return losses.avg, accuracies.avg


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    patience=5,
    scheduler=None,
):
    """
    Full training loop with early stopping.

    Args:
        model: The neural network model.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on.
        num_epochs: Maximum number of epochs.
        patience: Number of epochs to wait for improvement before stopping.
        scheduler: Learning rate scheduler (optional).

    Returns:
        float: The best validation accuracy achieved.
    """
    best_acc = 0.0
    patience_counter = 0

    # Ensure working directory exists for checkpoints
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Initialize scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        # Early Stopping and Checkpointing based on Validation Accuracy
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            patience_counter = 0
            print(f"New best validation accuracy: {best_acc}")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{patience}")

        # Save checkpoint
        utils.save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc,
            },
            is_best,
            filename=os.path.join(config.WORKING_DIR, "checkpoint.pth"),
            best_filename=config.MODEL_SAVE_PATH,
        )

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return best_acc


def generate_submission(model, test_loader, device, output_path=config.SUBMISSION_FILE):
    """
    Generates predictions for the test set and saves to CSV.

    Args:
        model: The trained model.
        test_loader: DataLoader for test data (yields images and ids).
        device: Device to run on.
        output_path: Path to save the submission CSV.
    """
    model.eval()
    predictions = []
    image_ids = []

    print("Generating predictions for test set...")

    # Get mapping to convert indices back to category IDs
    _, idx_to_cat = dataset.get_class_mappings()

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Map predictions back to original category IDs
            preds_np = preds.cpu().numpy()
            # Use .item() and get() for safety against type mismatches or out-of-bound indices in demo
            original_preds = [idx_to_cat.get(p.item(), 0) for p in preds_np]

            predictions.extend(original_preds)
            image_ids.extend(ids)

    # Create submission DataFrame
    # Column names based on Task Description: Id, Category
    df = pd.DataFrame({"Id": image_ids, "Category": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
