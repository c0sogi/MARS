import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.model import FunnelMLP
from library.data_utils import get_dataloaders


def train_epoch(model, dataloader, optimizer, scheduler, device, criterion):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for x_cont, x_cat, y in dataloader:
        x_cont = x_cont.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device).unsqueeze(1)  # Match output shape (batch, 1)

        optimizer.zero_grad()

        logits = model(x_cont, x_cat)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * x_cont.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate(model, dataloader, device, criterion):
    """
    Evaluates the model on the validation set.
    Returns average loss and AUC.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for x_cont, x_cat, y in dataloader:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device).unsqueeze(1)

            logits = model(x_cont, x_cat)
            loss = criterion(logits, y)

            running_loss += loss.item() * x_cont.size(0)

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits)

            all_targets.append(y.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)

    auc = roc_auc_score(all_targets, all_preds)

    return epoch_loss, auc


def train_model(load_cached_data=True):
    """
    Main training loop with Early Stopping and Model Checkpointing.
    """
    # 1. Data Setup
    print("Initializing DataLoaders...")
    train_loader, val_loader, _, num_continuous, vocab_sizes = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=load_cached_data,
    )

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Model Setup
    model = FunnelMLP(
        num_continuous=num_continuous,
        categorical_vocab_sizes=vocab_sizes,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_layers=Config.HIDDEN_LAYERS,
        dropout_rate=Config.DROPOUT_RATE,
        output_dim=Config.OUTPUT_DIM,
    ).to(device)

    # 3. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.MAX_LR,  # Initial LR is handled by OneCycleLR, but this sets the max implicitly if not overridden
        weight_decay=Config.WEIGHT_DECAY,
    )

    # OneCycleLR requires steps_per_epoch and epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        steps_per_epoch=len(train_loader),
        epochs=Config.EPOCHS,
        pct_start=0.3,  # Standard default
        div_factor=25,
        final_div_factor=1e4,
    )

    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_val_auc = 0.0
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, criterion
        )
        val_loss, val_auc = evaluate(model, val_loader, device, criterion)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc}"
        )  # Full precision print

        # Early Stopping & Checkpointing
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with AUC: {best_val_auc}")
        else:
            patience_counter += 1
            print(
                f"EarlyStopping counter: {patience_counter} out of {Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_val_auc}")


def generate_submission(load_cached_data=True):
    """
    Generates submission file using the best trained model.
    """
    print("Generating submission...")

    # Reload dataloaders to ensure we have the test set and metadata
    # We ignore train/val loaders here
    _, _, test_loader, num_continuous, vocab_sizes = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=load_cached_data,
    )

    device = torch.device(Config.DEVICE)

    # Initialize model structure
    model = FunnelMLP(
        num_continuous=num_continuous,
        categorical_vocab_sizes=vocab_sizes,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_layers=Config.HIDDEN_LAYERS,
        dropout_rate=Config.DROPOUT_RATE,
        output_dim=Config.OUTPUT_DIM,
    ).to(device)

    # Load best weights
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_SAVE_PATH}. Train model first."
        )

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    all_preds = []

    # Inference
    with torch.no_grad():
        for x_cont, x_cat in test_loader:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)

            logits = model(x_cont, x_cat)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())

    all_preds = np.vstack(all_preds).flatten()

    # Prepare Submission DataFrame
    # Load test IDs from metadata file
    test_df = pd.read_csv(Config.TEST_PATH)
    submission_df = pd.DataFrame({"id": test_df["id"], "target": all_preds})

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
