import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from library.utils import save_checkpoint
from library.config import DEVICE


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        # Unpack batch (dataset returns image, label)
        inputs, labels = batch

        inputs = inputs.to(device)
        # BCEWithLogitsLoss expects targets to be shape (N, 1) matching output
        labels = labels.to(device).view(-1, 1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch

            inputs = inputs.to(device)
            labels = labels.to(device).view(-1, 1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities for metrics
            probs = torch.sigmoid(outputs)

            all_targets.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    total_loss = running_loss / dataset_size

    # Concatenate all batches
    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)

    # Calculate Metrics
    # Accuracy (threshold 0.5)
    acc = accuracy_score(all_targets, (all_preds > 0.5).astype(int))

    # ROC AUC
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle edge case if only one class is present in validation (unlikely)
        auc = 0.5

    return total_loss, acc, auc


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    patience,
    min_delta,
    model_filename,
):
    """
    Orchestrates the training process including early stopping and scheduling.
    """
    best_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)

        # Step the scheduler (Cosine Annealing usually steps per epoch)
        if scheduler is not None:
            scheduler.step()

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(
            f"Epoch {epoch+1}/{num_epochs} | Time: {int(epoch_mins)}m {int(epoch_secs)}s"
        )
        print(f"    Train Loss: {train_loss:.8f}")
        print(f"    Val Loss:   {val_loss:.8f}")
        print(f"    Val Acc:    {val_acc:.8f}")
        print(f"    Val AUC:    {val_auc:.8f}")

        # Early Stopping Check
        if val_loss < (best_loss - min_delta):
            best_loss = val_loss
            patience_counter = 0
            save_checkpoint(model.state_dict(), model_filename)
            # print(f"    Validation loss improved. Model saved to {model_filename}")
        else:
            patience_counter += 1
            # print(f"    No improvement. EarlyStopping counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return best_loss


def predict_tta(model, dataloader, device):
    """
    Performs inference using Test Time Augmentation (Original, H-Flip, V-Flip).
    Returns averaged probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            # Test dataset might return just images or (images, labels) or (images, ids)
            # We handle the case where it returns just images or a tuple
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch

            inputs = inputs.to(device)

            # 1. Original Prediction
            out_orig = model(inputs)
            prob_orig = torch.sigmoid(out_orig)

            # 2. Horizontal Flip Prediction (dim 3 is width)
            inputs_h = torch.flip(inputs, dims=[3])
            out_h = model(inputs_h)
            prob_h = torch.sigmoid(out_h)

            # 3. Vertical Flip Prediction (dim 2 is height)
            inputs_v = torch.flip(inputs, dims=[2])
            out_v = model(inputs_v)
            prob_v = torch.sigmoid(out_v)

            # Average predictions
            avg_prob = (prob_orig + prob_h + prob_v) / 3.0

            all_preds.append(avg_prob.cpu().numpy())

    # Concatenate all batches -> Shape (N, 1)
    return np.vstack(all_preds)
