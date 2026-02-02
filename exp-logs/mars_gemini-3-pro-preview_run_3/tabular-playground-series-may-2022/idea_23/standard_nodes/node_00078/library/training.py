import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import seed_everything, compute_auc
from library.data_processing import prepare_data
from library.model import AVPFEModel


def train_one_epoch(model, loader, optimizer, scheduler, device, criterion):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        x_cont = batch["x_cont"].to(device)
        x_cat = batch["x_cat"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass: returns (Batch, Num_Streams) logits
        logits = model(x_cont, x_cat)

        # Calculate loss: Sum of BCE for each stream
        loss = 0
        for i in range(logits.shape[1]):
            loss += criterion(logits[:, i], targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * x_cont.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Returns the average ensemble AUC and the raw predictions/targets.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            x_cont = batch["x_cont"].to(device)
            x_cat = batch["x_cat"].to(device)
            targets = batch["target"].to(device)

            # Forward pass
            logits = model(x_cont, x_cat)

            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Ensemble Strategy: Arithmetic Mean of the 5 streams
            # Shape: (Batch, 5) -> (Batch,)
            avg_preds = torch.mean(probs, dim=1)

            all_preds.append(avg_preds.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    auc = compute_auc(all_targets, all_preds)
    return auc


def run_training():
    """
    Main function to run the training pipeline.
    """
    # Reproducibility
    seed_everything(Config.SEED)

    # 1. Prepare Data
    print("Preparing data...")
    train_loader, val_loader, test_loader, vocab_sizes = prepare_data(
        load_cached_data=True
    )

    # Determine continuous feature count from a sample batch
    sample_batch = next(iter(train_loader))
    num_cont_features = sample_batch["x_cont"].shape[1]

    # 2. Initialize Model
    print(f"Initializing AV-PFE Model on {Config.DEVICE}...")
    model = AVPFEModel(vocab_sizes=vocab_sizes, num_cont_features=num_cont_features)
    model.to(Config.DEVICE)

    # 3. Setup Optimizer and Scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,  # This is max_lr for OneCycle
        weight_decay=Config.WEIGHT_DECAY,
    )

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_auc = 0.0
    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, Config.DEVICE, criterion
        )

        val_auc = validate(model, val_loader, Config.DEVICE)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val AUC: {val_auc:.10f}"
        )

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            print(f"New best AUC! Saving model to {Config.MODEL_PATH}")
            torch.save(model.state_dict(), Config.MODEL_PATH)

    print(f"Training complete. Best Validation AUC: {best_auc:.10f}")


def generate_submission():
    """
    Loads the best model and generates predictions for the test set.
    """
    print("Generating submission...")

    # Load Data (Test loader needed)
    _, _, test_loader, vocab_sizes = prepare_data(load_cached_data=True)

    # Determine continuous feature count
    sample_batch = next(iter(test_loader))
    num_cont_features = sample_batch["x_cont"].shape[1]

    # Initialize Model
    model = AVPFEModel(vocab_sizes=vocab_sizes, num_cont_features=num_cont_features)
    model.to(Config.DEVICE)

    # Load Weights
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.eval()

    all_ids = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            x_cont = batch["x_cont"].to(Config.DEVICE)
            x_cat = batch["x_cat"].to(Config.DEVICE)
            ids = batch["id"]

            logits = model(x_cont, x_cat)
            probs = torch.sigmoid(logits)

            # Ensemble Mean
            avg_preds = torch.mean(probs, dim=1)

            all_ids.extend(ids.numpy())
            all_preds.extend(avg_preds.cpu().numpy())

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"id": all_ids, "target": all_preds})

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
