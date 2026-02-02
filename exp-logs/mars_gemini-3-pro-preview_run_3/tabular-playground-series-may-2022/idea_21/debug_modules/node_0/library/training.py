import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

# Import from library files
from library.config import Config
from library.data_processing import set_seed, load_and_preprocess, get_dataloaders
from library.model import RSPFEModel


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """
    Trains the model for one epoch.
    Computes loss as the sum of BCE losses from all 5 streams.
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        cat_features = batch["cat_features"].to(device)
        cont_features = batch["cont_features"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass: returns (batch_size, 5) logits
        outputs = model(cat_features, cont_features)

        # Calculate loss: Sum of BCE loss for each stream
        loss = 0
        for i in range(5):
            loss += criterion(outputs[:, i], targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * targets.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Predictions are the arithmetic mean of the 5 stream probabilities.
    Metric: ROC AUC.
    """
    model.eval()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            cat_features = batch["cat_features"].to(device)
            cont_features = batch["cont_features"].to(device)
            targets = batch["target"].to(device)

            # Forward pass: (batch_size, 5) logits
            logits = model(cat_features, cont_features)

            # Convert to probabilities
            probs = torch.sigmoid(logits)

            # Ensemble prediction: Arithmetic mean across streams (dim=1)
            mean_preds = probs.mean(dim=1)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(mean_preds.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    auc_score = roc_auc_score(all_targets, all_preds)
    return auc_score, all_preds


def train_pipeline():
    """
    Main training pipeline:
    1. Load Data
    2. Init Model, Optimizer, Scheduler
    3. Train with Early Stopping
    4. Save Best Model
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Load Data
    df_train, df_val, df_test, vocab_sizes = load_and_preprocess(load_cached_data=True)
    train_loader, val_loader, test_loader = get_dataloaders(df_train, df_val, df_test)

    # 2. Initialize Model
    model = RSPFEModel(vocab_sizes=vocab_sizes)
    model.to(device)

    # 3. Optimizer & Scheduler
    # Explicitly using AdamW with weight decay 1e-5 as per strategy
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR
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

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_auc = 0.0
    patience = 5
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_auc, _ = evaluate(model, val_loader, device)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val AUC: {val_auc}"
        )

        # Early Stopping & Model Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Val AUC: {best_auc}")
    return test_loader, vocab_sizes


def inference_and_submit(test_loader, vocab_sizes):
    """
    Loads the best model, generates predictions for the test set,
    and creates the submission file.
    """
    device = torch.device(Config.DEVICE)

    # Initialize model structure
    model = RSPFEModel(vocab_sizes=vocab_sizes)
    model.to(device)

    # Load best weights
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    print("Generating predictions on test set...")
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            cat_features = batch["cat_features"].to(device)
            cont_features = batch["cont_features"].to(device)

            # Forward pass
            logits = model(cat_features, cont_features)
            probs = torch.sigmoid(logits)

            # Mean of 5 streams
            mean_preds = probs.mean(dim=1)
            all_preds.extend(mean_preds.cpu().numpy())

    # Create Submission DataFrame
    # Load sample submission to get IDs
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Ensure lengths match
    if len(all_preds) != len(sample_sub):
        print(
            f"Warning: Prediction length {len(all_preds)} does not match sample submission length {len(sample_sub)}."
        )
        # In case of debug mode, we might have fewer predictions.
        # We will just take the IDs from the test set processed.
        # But for full run, they must match.
        if Config.DEBUG:
            # Just use the first N ids
            sample_sub = sample_sub.iloc[: len(all_preds)].copy()
        else:
            raise ValueError("Prediction count mismatch.")

    submission = pd.DataFrame({"id": sample_sub["id"], "target": all_preds})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run():
    """
    Orchestrates the training and submission process.
    """
    # Train
    test_loader, vocab_sizes = train_pipeline()

    # Inference
    inference_and_submit(test_loader, vocab_sizes)
