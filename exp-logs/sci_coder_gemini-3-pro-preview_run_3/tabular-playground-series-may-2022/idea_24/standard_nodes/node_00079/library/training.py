import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.data_processing import ManufacturingDataset, preprocess_features
from library.model import ParallelFunnelEnsemble, set_seed


def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    """
    Trains the model for one epoch.
    Computes the loss as the sum of Binary Cross-Entropy losses from the 5 streams.
    """
    model.train()
    running_loss = 0.0

    for x_cont, x_cat, y in loader:
        x_cont, x_cat, y = x_cont.to(device), x_cat.to(device), y.to(device)

        optimizer.zero_grad()

        # Forward pass: (batch, num_streams)
        logits = model(x_cont, x_cat)

        # Compute sum of BCE losses for each stream
        loss = 0
        for i in range(logits.shape[1]):
            loss += criterion(logits[:, i], y)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Calculates the ROC AUC score by averaging the probabilities of the 5 streams.
    """
    model.eval()
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for x_cont, x_cat, y in loader:
            x_cont, x_cat, y = x_cont.to(device), x_cat.to(device), y.to(device)

            logits = model(x_cont, x_cat)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Ensemble prediction: Mean of probabilities
            mean_probs = torch.mean(probs, dim=1)

            val_preds.append(mean_probs.cpu().numpy())
            val_targets.append(y.cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)

    auc = roc_auc_score(val_targets, val_preds)
    return auc


def inference(model, loader, device):
    """
    Generates predictions for the test set using the ensemble mean.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for x_cont, x_cat, _ in loader:
            x_cont, x_cat = x_cont.to(device), x_cat.to(device)

            logits = model(x_cont, x_cat)
            probs = torch.sigmoid(logits)

            # Ensemble prediction: Mean of probabilities
            mean_probs = torch.mean(probs, dim=1)
            all_preds.append(mean_probs.cpu().numpy())

    return np.concatenate(all_preds)


def train_model():
    """
    Main training loop with Early Stopping.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load and Process Data
    # Use cached data if available, otherwise process and cache
    train_df, val_df, _, vocab_sizes, cat_cols, cont_cols = preprocess_features(
        load_cached_data=True, config=Config
    )

    # 2. Create Datasets and Loaders
    train_dataset = ManufacturingDataset(train_df, cat_cols, cont_cols, mode="train")
    val_dataset = ManufacturingDataset(val_df, cat_cols, cont_cols, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Initialize Model
    model = ParallelFunnelEnsemble(
        vocab_sizes=vocab_sizes,
        cont_dim=len(cont_cols),
        embed_dim=Config.EMBED_DIM,
        stream_configs=Config.MODEL_STREAMS,
    ).to(device)

    # 4. Setup Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.MAX_LR, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        avg_train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_auc = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val AUC: {val_auc:.10f}"
        )

        # Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print("New best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Val AUC: {best_auc:.10f}")


def predict_and_submit():
    """
    Generates predictions for the test set and saves to submission file.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load processed data (Test set)
    _, _, test_df, vocab_sizes, cat_cols, cont_cols = preprocess_features(
        load_cached_data=True, config=Config
    )

    test_dataset = ManufacturingDataset(test_df, cat_cols, cont_cols, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize Model
    model = ParallelFunnelEnsemble(
        vocab_sizes=vocab_sizes,
        cont_dim=len(cont_cols),
        embed_dim=Config.EMBED_DIM,
        stream_configs=Config.MODEL_STREAMS,
    ).to(device)

    # Load Best Weights
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    print("Generating predictions...")
    final_preds = inference(model, test_loader, device)

    # Create Submission File
    submission = pd.DataFrame({"id": test_df["id"], "target": final_preds})

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
