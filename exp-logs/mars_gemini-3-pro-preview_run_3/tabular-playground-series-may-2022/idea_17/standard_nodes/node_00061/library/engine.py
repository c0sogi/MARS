import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import random
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.model import TreeFunnelEnsemble


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for cat_x, cont_x, targets in dataloader:
        batch_size = cat_x.size(0)
        dataset_size += batch_size

        cat_x = cat_x.to(device)
        cont_x = cont_x.to(device)
        # Targets need to be (batch_size, 1) for BCEWithLogitsLoss
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()

        # Forward pass: returns list of outputs from each head
        outputs = model(cat_x, cont_x)

        # Calculate loss: sum of losses from each head
        loss = 0
        for out in outputs:
            loss += criterion(out, targets)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns the average loss and AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for cat_x, cont_x, targets in dataloader:
            batch_size = cat_x.size(0)
            dataset_size += batch_size

            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(cat_x, cont_x)

            # Loss calculation: Sum of losses
            loss = 0
            probs_sum = 0

            for out in outputs:
                loss += criterion(out, targets)
                # Apply sigmoid to logits for probability
                probs_sum += torch.sigmoid(out)

            # Average predictions across heads for evaluation metric
            avg_probs = probs_sum / len(outputs)

            running_loss += loss.item() * batch_size

            all_targets.append(targets.cpu().numpy())
            all_preds.append(avg_probs.cpu().numpy())

    final_loss = running_loss / dataset_size

    # Concatenate all batches
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Calculate AUC
    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc_score = 0.5  # Fallback if only one class present in batch (unlikely)

    return final_loss, auc_score


def train_model(train_loader, val_loader, vocab_sizes, cont_dim):
    """
    Main training loop with Early Stopping.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Initializing model on {device}...")
    model = TreeFunnelEnsemble(vocab_sizes, cont_dim)
    model.to(device)

    # Optimizer: AdamW with calibrated weight decay
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Loss: BCEWithLogitsLoss (numerically stable)
    criterion = nn.BCEWithLogitsLoss()

    # Scheduler: OneCycleLR
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=10000.0,
    )

    # Early Stopping setup
    best_auc = 0.0
    patience_counter = 0
    best_model_path = Config.MODEL_PATH

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.10f} | Val Loss: {val_loss:.10f} | Val AUC: {val_auc:.10f}"
        )

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! AUC: {best_auc:.10f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Validation AUC: {best_auc:.10f}")
    return best_auc


def generate_submission(test_loader, test_ids, vocab_sizes, cont_dim):
    """
    Generates predictions for the test set using the best saved model
    and saves them to the submission CSV.
    """
    device = torch.device(Config.DEVICE)

    # Load Model
    print(f"Loading best model from {Config.MODEL_PATH}...")
    model = TreeFunnelEnsemble(vocab_sizes, cont_dim)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            # Unpack batch (test loader returns cat, cont)
            cat_x, cont_x = batch

            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)

            outputs = model(cat_x, cont_x)

            # Average predictions across heads
            probs_sum = 0
            for out in outputs:
                probs_sum += torch.sigmoid(out)
            avg_probs = probs_sum / len(outputs)

            all_preds.append(avg_probs.cpu().numpy())

    # Concatenate and flatten
    predictions = np.concatenate(all_preds).flatten()

    # Create submission DataFrame
    df = pd.DataFrame({"id": test_ids, "target": predictions})

    # Ensure ID is integer
    df["id"] = df["id"].astype(int)

    # Save to CSV
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")
