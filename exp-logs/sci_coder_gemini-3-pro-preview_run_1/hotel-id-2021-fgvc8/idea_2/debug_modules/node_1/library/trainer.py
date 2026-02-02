import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.utils import Config, AverageMeter, save_checkpoint, get_device
from library.model import LightweightMetricModel
from library.dataset import get_dataloaders


def mapk(target, preds, k=5):
    """
    Computes MAP@K for a batch of predictions with a single ground truth.

    Args:
        target (torch.Tensor): Ground truth labels (Batch,).
        preds (torch.Tensor): Predicted top-k indices (Batch, K).
        k (int): K value.

    Returns:
        float: The mean average precision score.
    """
    # target: (B,) -> (B, 1)
    target = target.view(-1, 1)

    # Check where prediction matches target
    # matches: (B, K) boolean tensor
    matches = preds == target

    # Find the rank (0-indexed) of the match.
    # If no match in row, we get all False.
    # We want 1 / (rank + 1).
    score = torch.zeros(target.size(0), device=target.device)

    for i in range(k):
        # matches[:, i] is True if the i-th prediction is correct
        # We add 1/(i+1) to the score for those rows
        score += matches[:, i].float() * (1.0 / (i + 1))

    return score.mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc@1", ":.2f")

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        # ArcFace head returns logits with margin penalty applied
        outputs = model(images, labels)

        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        acc1 = (outputs.argmax(dim=1) == labels).float().mean().item() * 100
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))

    print(f"Epoch [{epoch}] Train Loss: {losses.avg:.4f} | Acc@1: {top1.avg:.2f}%")
    return losses.avg, top1.avg


def validate(model, loader, device, criterion):
    """
    Evaluates the model on the validation set using MAP@5.
    """
    model.eval()
    losses = AverageMeter("Loss", ":.4f")
    map5_meter = AverageMeter("MAP@5", ":.5f")

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass (inference mode, label=None)
            # Returns scaled cosine similarities
            outputs = model(images)

            # Calculate validation loss
            loss = criterion(outputs, labels)
            losses.update(loss.item(), images.size(0))

            # Calculate MAP@5
            # Get top 5 indices
            _, preds = outputs.topk(5, dim=1, largest=True, sorted=True)

            score = mapk(labels, preds, k=5)
            map5_meter.update(score, images.size(0))

    print(f"Validation Loss: {losses.avg:.4f} | MAP@5: {map5_meter.avg:.5f}")
    return map5_meter.avg


def predict_and_submit(model, loader, encoder_classes, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    print("Generating submission...")
    model.eval()

    image_ids = []
    predictions = []

    with torch.no_grad():
        for images, filenames in loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Get top 5 indices
            _, top_indices = outputs.topk(5, dim=1, largest=True, sorted=True)

            top_indices = top_indices.cpu().numpy()

            # Map indices to original hotel IDs
            # encoder_classes is a numpy array of original IDs
            batch_preds = encoder_classes[top_indices]

            # Format as space-delimited string
            for pred_row in batch_preds:
                # pred_row is array of 5 hotel IDs
                pred_str = " ".join(map(str, pred_row))
                predictions.append(pred_str)

            image_ids.extend(filenames)

    # Create DataFrame
    submission_df = pd.DataFrame({"image": image_ids, "hotel_id": predictions})

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(
    load_cached_data=True,
    epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    patience=5,
):
    """
    Main training orchestration function.
    """
    device = get_device()
    print(f"Using device: {device}")

    # 1. Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader, encoder_classes = get_dataloaders(
        load_cached_data=load_cached_data, batch_size=batch_size
    )

    # 2. Model
    print("Initializing Model...")
    model = LightweightMetricModel(
        num_classes=len(encoder_classes),
        embedding_dim=Config.EMBEDDING_DIM,
        backbone_name=Config.BACKBONE,
    )
    model = model.to(device)

    # 3. Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # 4. Training Loop
    best_map5 = 0.0
    epochs_no_improve = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_map5 = validate(model, val_loader, device, criterion)

        # Scheduler Step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time
        print(f"Epoch {epoch} completed in {elapsed:.0f}s. LR: {current_lr:.6f}")

        # Checkpoint & Early Stopping
        is_best = val_map5 > best_map5
        if is_best:
            best_map5 = val_map5
            epochs_no_improve = 0
            print(f"New best MAP@5: {best_map5:.5f}. Saving model...")
        else:
            epochs_no_improve += 1

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_map5": best_map5,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            Config.CHECKPOINT_PATH,
        )

        if epochs_no_improve >= patience:
            print(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

    print(f"Training complete. Best MAP@5: {best_map5:.5f}")

    # 5. Inference / Submission
    print("Loading best model for submission...")
    best_model_path = Config.BEST_MODEL_PATH
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print("Warning: Best model not found, using current model.")

    predict_and_submit(
        model, test_loader, encoder_classes, device, Config.SUBMISSION_PATH
    )
