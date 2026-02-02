import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import AverageMeter
from library.model import mixup_data, mixup_criterion


def train_one_epoch(model, loader, criterion, optimizer, device, epoch=None):
    """
    Trains the model for one epoch using Mixup regularization.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (on_img, off_img, target) in enumerate(loader):
        on_img = on_img.to(device)
        off_img = off_img.to(device)
        target = target.to(device).unsqueeze(1)

        # Apply Mixup
        mixed_on, mixed_off, target_a, target_b, lam = mixup_data(
            on_img, off_img, target, alpha=Config.MIXUP_ALPHA, device=device
        )

        optimizer.zero_grad()

        # Forward pass with mixed inputs
        logits = model(mixed_on, mixed_off)

        # Calculate loss using Mixup criterion
        loss = mixup_criterion(criterion, logits, target_a, target_b, lam)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), on_img.size(0))

    return losses.avg


def validate_one_epoch(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for on_img, off_img, target in loader:
            on_img = on_img.to(device)
            off_img = off_img.to(device)
            target = target.to(device).unsqueeze(1)

            # Forward pass
            logits = model(on_img, off_img)
            loss = criterion(logits, target)

            losses.update(loss.item(), on_img.size(0))

            # Apply sigmoid for probabilities
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(target.cpu().numpy())

    # Concatenate results
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
    else:
        all_preds = np.array([])
        all_targets = np.array([])

    # Calculate AUC
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return losses.avg, auc


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    epochs,
    patience,
    save_path,
):
    """
    Main training loop with Early Stopping and Scheduler stepping.
    """
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, device)

        # Step Scheduler
        if scheduler is not None:
            scheduler.step()

        # Log Metrics (Full Precision)
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Early Stopping and Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with AUC: {val_auc}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return best_auc
