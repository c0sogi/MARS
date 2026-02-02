import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.data_utils import get_dataloaders
from library.model import SDPEModel


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """
    Executes one training epoch.
    Calculates loss as the sum of Binary Cross-Entropy losses from all 5 streams.
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        x_cont = batch["x_cont"].to(device)
        x_cat = batch["x_cat"].to(device)
        target = batch["target"].to(device).unsqueeze(1)

        optimizer.zero_grad()

        # Forward pass returns list of logits from 5 streams
        outputs = model(x_cont, x_cat)

        # Loss is sum of BCEs
        loss = 0
        for logits in outputs:
            loss += criterion(logits, target)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Uses the arithmetic mean of the 5 stream probabilities as the final prediction.
    Returns ROC AUC score.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            x_cont = batch["x_cont"].to(device)
            x_cat = batch["x_cat"].to(device)
            target = batch["target"].to(device)

            outputs = model(x_cont, x_cat)

            # Ensemble Mean for Validation
            # Stack logits: [5, Batch, 1] -> Sigmoid -> Mean -> [Batch, 1]
            probs = torch.stack([torch.sigmoid(out) for out in outputs])
            avg_prob = torch.mean(probs, dim=0).squeeze(1)

            all_preds.extend(avg_prob.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    auc = roc_auc_score(all_targets, all_preds)
    return auc


def predict_and_submit(model, test_loader, device):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    # Load best weights
    if os.path.exists(Config.MODEL_PATH):
        print(f"Loading best model from {Config.MODEL_PATH}...")
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model file not found. Using current weights.")

    model.eval()
    all_ids = []
    all_preds = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            x_cont = batch["x_cont"].to(device)
            x_cat = batch["x_cat"].to(device)
            ids = batch["id"]

            outputs = model(x_cont, x_cat)

            # Ensemble Mean
            probs = torch.stack([torch.sigmoid(out) for out in outputs])
            avg_prob = torch.mean(probs, dim=0).squeeze(1)

            all_preds.extend(avg_prob.cpu().numpy())
            all_ids.extend(ids.numpy())

    # Create submission dataframe
    df_sub = pd.DataFrame({"id": all_ids, "target": all_preds})

    # Ensure ID is int
    df_sub["id"] = df_sub["id"].astype(int)

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training(load_cached_data=True):
    """
    Main function to manage training and validation loops.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    # 1. Data Loading
    train_loader, val_loader, test_loader, vocab_sizes = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # Infer num_cont from the first batch
    # We need to fetch one batch to see the shape of x_cont
    sample_batch = next(iter(train_loader))
    num_cont = sample_batch["x_cont"].shape[1]
    print(f"Inferred num_cont: {num_cont}")

    # 2. Model Initialization
    model = SDPEModel(vocab_sizes=vocab_sizes, num_cont=num_cont)
    model.to(device)

    # 3. Optimizer & Scheduler
    # Explicitly using standard Adam as per strategy
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=Config.NUM_EPOCHS,
    )

    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_auc = 0.0
    patience = 5
    patience_counter = 0

    print(f"Starting training on {device} for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )

        # Validate
        val_auc = evaluate(model, val_loader, device)

        # Print metrics (Full precision)
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Train Loss: {train_loss} - Val AUC: {val_auc}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved with AUC: {val_auc}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Validation AUC: {best_auc}")

    # 5. Prediction
    predict_and_submit(model, test_loader, device)
