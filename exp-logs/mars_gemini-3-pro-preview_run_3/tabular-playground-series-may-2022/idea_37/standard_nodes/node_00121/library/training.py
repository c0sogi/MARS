import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.model import DAR_PE_Model


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """
    Handles the training for a single epoch.
    Computes loss as the sum of BCE losses from all 5 independent streams.
    """
    model.train()
    running_loss = 0.0

    for x_cont, x_cat, y in dataloader:
        x_cont = x_cont.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Forward pass returns a list of outputs (one per stream)
        outputs = model(x_cont, x_cat)

        # Calculate loss: Sum of BCE loss for each stream
        loss = 0
        for out in outputs:
            loss += criterion(out, y)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Predictions are the arithmetic mean of the probabilities from all 5 streams.
    """
    model.eval()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for x_cont, x_cat, y in dataloader:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device)

            outputs = model(x_cont, x_cat)

            # Apply sigmoid to each stream's output and average them
            probs = [torch.sigmoid(out) for out in outputs]
            avg_prob = torch.mean(torch.stack(probs), dim=0)

            all_targets.append(y.cpu().numpy())
            all_preds.append(avg_prob.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    auc = roc_auc_score(all_targets, all_preds)
    return auc


def train_model(loaders, dims):
    """
    Main training loop.
    Initializes model, optimizer, scheduler, and handles the training/validation cycle.
    Saves the best model based on Validation AUC.
    """
    device = Config.DEVICE
    print(f"Initializing DAR-PE Model on {device}...")

    model = DAR_PE_Model(n_cont=dims["n_cont"], vocab_sizes=dims["vocab_sizes"])
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    steps_per_epoch = len(loaders["train"])
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, loaders["train"], optimizer, scheduler, criterion, device
        )
        val_auc = evaluate(model, loaders["val"], device)

        # Print full precision as requested
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Loss: {train_loss} | Val AUC: {val_auc}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best Val AUC: {best_auc}")
    return best_model_path


def generate_submission(model_path, loaders, dims, test_ids):
    """
    Generates predictions for the test set using the best saved model.
    Predictions are averaged across the 5 streams.
    """
    print("Generating submission...")
    device = Config.DEVICE

    model = DAR_PE_Model(n_cont=dims["n_cont"], vocab_sizes=dims["vocab_sizes"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []

    with torch.no_grad():
        for x_cont, x_cat in loaders["test"]:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)

            outputs = model(x_cont, x_cat)

            # Average probabilities across streams
            probs = [torch.sigmoid(out) for out in outputs]
            avg_prob = torch.mean(torch.stack(probs), dim=0)

            all_preds.append(avg_prob.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    sub_df = pd.DataFrame({"id": test_ids, "target": all_preds})
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
