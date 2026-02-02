import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.utils import compute_roc_auc


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """
    Handles the training of one epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)

    for x_cat, x_cont, targets in dataloader:
        x_cat = x_cat.to(device)
        x_cont = x_cont.to(device)
        targets = targets.to(device)

        # Ensure targets are (Batch, 1)
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        optimizer.zero_grad()

        # Forward pass returns lists of outputs (one per stream)
        main_outs = model(x_cat, x_cont)

        # Sum BCE loss for all streams
        loss = 0
        for out in main_outs:
            loss += criterion(out, targets)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()

    avg_loss = running_loss / num_batches
    return avg_loss


def validate(model, dataloader, device):
    """
    Evaluates the model by averaging predictions from the 5 main heads and computing ROC AUC.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_cat, x_cont, targets in dataloader:
            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)

            main_outs = model(x_cat, x_cont)

            # Apply sigmoid to each main output to get probabilities
            probs = [torch.sigmoid(out) for out in main_outs]

            # Stack and average: (num_streams, batch, 1) -> (batch, 1)
            avg_prob = torch.stack(probs).mean(dim=0)

            all_preds.append(avg_prob.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    auc = compute_roc_auc(all_targets, all_preds)
    return auc


def train_model(model, train_loader, val_loader, config):
    """
    Manages the training loop with Early Stopping.
    """
    device = config.DEVICE
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # OneCycleLR scheduler configuration
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=config.EPOCHS,
    )

    best_auc = 0.0
    patience = 5  # Early stopping patience
    patience_counter = 0

    print("Starting training...")

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_auc = validate(model, val_loader, device)

        # Print full precision as requested
        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss} | Val AUC: {val_auc}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"Saved new best model with AUC: {best_auc}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break


def predict_submission(model, test_loader, test_ids, config):
    """
    Generates predictions for the test set and saves to CSV.
    """
    device = config.DEVICE

    # Load best model weights
    if os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: Best model not found, using current weights.")

    model.to(device)
    model.eval()

    all_preds = []

    with torch.no_grad():
        for x_cat, x_cont in test_loader:
            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)

            main_outs = model(x_cat, x_cont)

            # Average probabilities from all streams
            probs = [torch.sigmoid(out) for out in main_outs]
            avg_prob = torch.stack(probs).mean(dim=0)

            all_preds.append(avg_prob.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()

    # Create submission DataFrame
    submission = pd.DataFrame({"id": test_ids, "target": all_preds})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
