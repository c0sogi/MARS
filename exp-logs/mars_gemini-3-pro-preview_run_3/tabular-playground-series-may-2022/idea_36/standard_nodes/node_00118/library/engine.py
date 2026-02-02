import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config
import os


def train_one_epoch(model, dataloader, optimizer, scheduler, device, criterion):
    """
    Trains the model for one epoch.
    Calculates the sum of BCE losses across all 5 streams.
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        # Unpack batch
        x_cat = batch["cat"].to(device)
        x_cont = batch["cont"].to(device)
        targets = batch["target"].to(device)  # Shape: (Batch, 1)

        optimizer.zero_grad()

        # Forward pass: Output shape (Batch, 5)
        outputs = model(x_cat, x_cont)

        # Calculate Loss: Sum of independent BCE losses for each stream
        loss = 0
        # Iterate over the 5 streams
        for i in range(Config.NUM_STREAMS):
            # Select the i-th stream output (Batch, 1 after unsqueeze or just match dimensions)
            stream_output = outputs[:, i].unsqueeze(1)  # Shape: (Batch, 1)
            loss += criterion(stream_output, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Step the scheduler (OneCycleLR steps per batch)
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * x_cat.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate(model, dataloader, device, criterion):
    """
    Evaluates the model on the validation set.
    Metric: ROC AUC (calculated on the mean prediction of the 5 streams).
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            x_cat = batch["cat"].to(device)
            x_cont = batch["cont"].to(device)
            targets = batch["target"].to(device)

            # Forward pass: (Batch, 5)
            outputs = model(x_cat, x_cont)

            # Calculate Validation Loss (Sum of streams, consistent with training)
            loss = 0
            for i in range(Config.NUM_STREAMS):
                stream_output = outputs[:, i].unsqueeze(1)
                loss += criterion(stream_output, targets)

            running_loss += loss.item() * x_cat.size(0)

            # Inference Strategy: Mean of probabilities across 5 streams
            # 1. Sigmoid to get probabilities per stream
            probs_per_stream = torch.sigmoid(outputs)
            # 2. Arithmetic mean across streams (dim 1)
            avg_probs = torch.mean(probs_per_stream, dim=1)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(avg_probs.cpu().numpy())

    # Concatenate all batches
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Calculate Metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc_score = 0.0

    return epoch_loss, auc_score


def train_model(model, train_loader, val_loader):
    """
    Main training loop with Early Stopping and Scheduler handling.
    """
    device = Config.DEVICE
    model.to(device)

    # Optimizer: AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.MAX_LR,  # Initial LR is handled by OneCycleLR, but we pass max here for safety or reference
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Loss Function: BCEWithLogitsLoss
    # We sum the scalar losses manually in the loop, so reduction='mean' applies to the batch dimension.
    criterion = nn.BCEWithLogitsLoss()

    # Scheduler: OneCycleLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")
    print(f"Batch Size: {Config.BATCH_SIZE}, Streams: {Config.NUM_STREAMS}")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, criterion
        )

        # Validate
        val_loss, val_auc = evaluate(model, val_loader, device, criterion)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Val AUC: {best_auc}")
    return best_auc


def predict(model, test_loader):
    """
    Generates predictions for the test set using the trained model.
    Loads the best model state from disk.
    """
    device = Config.DEVICE

    # Load best model weights
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
        print("Loaded best model for inference.")
    else:
        print("Warning: Best model not found. Using current model weights.")

    model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            x_cat = batch["cat"].to(device)
            x_cont = batch["cont"].to(device)

            # Forward pass
            outputs = model(x_cat, x_cont)

            # Inference Strategy: Mean of probabilities across 5 streams
            probs_per_stream = torch.sigmoid(outputs)
            avg_probs = torch.mean(probs_per_stream, dim=1)

            predictions.extend(avg_probs.cpu().numpy())

    return np.array(predictions)
