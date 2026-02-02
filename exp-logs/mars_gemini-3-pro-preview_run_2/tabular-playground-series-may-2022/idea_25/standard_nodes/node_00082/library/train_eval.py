import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything, custom_weight_init
from library.data_loader import get_dataloaders
from library.model import BalancedProcessCompressHybrid


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for x_num, x_cat, y in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(x_num, x_cat).squeeze()
        loss = criterion(logits, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and AUC score.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_num, x_cat, y in loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device)

            logits = model(x_num, x_cat).squeeze()
            loss = criterion(logits, y)
            running_loss += loss.item()

            preds = torch.sigmoid(logits)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss = running_loss / len(loader)

    # Concatenate predictions and targets
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate AUC
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle case where only one class is present in batch (unlikely with proper shuffling/size)
        auc = 0.5

    return avg_loss, auc


def run_training(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=False):
    """
    Main function to run the training pipeline, validation, and submission generation.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    # If debug is enabled, we might want to limit the data, but get_dataloaders
    # doesn't support slicing easily without reloading. We will limit epochs instead.
    if debug:
        print("Debug mode enabled: reducing epochs to 2.")
        epochs = 2

    train_loader, val_loader, test_loader, vocab_size = get_dataloaders(
        batch_size=batch_size, num_workers=Config.NUM_WORKERS, load_cached_data=True
    )

    # Determine input dimensions from a sample batch
    sample_x_num, sample_x_cat, _ = next(iter(train_loader))
    num_continuous = sample_x_num.shape[1]
    cat_seq_len = sample_x_cat.shape[1]

    # 3. Model Initialization
    model = BalancedProcessCompressHybrid(
        num_continuous=num_continuous,
        cat_seq_len=cat_seq_len,
        vocab_size=vocab_size,
        embed_dim=Config.EMBED_DIM,
        transformer_layers=Config.TRANSFORMER_LAYERS,
        transformer_heads=Config.TRANSFORMER_HEADS,
        backbone_stages=Config.BACKBONE_STAGES,
        dropout_transformer=Config.DROPOUT_TRANSFORMER,
        dropout_backbone=Config.DROPOUT_BACKBONE,
    ).to(device)

    # Apply specific weight initialization
    model.apply(custom_weight_init)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Aggressive Step Learning Rate Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Validation
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.10f} | Val Loss: {val_loss:.10f} | Val AUC: {val_auc:.10f} | LR: {current_lr:.2e}"
        )

        # Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    print(f"Training complete. Best Validation AUC: {best_auc:.10f}")

    # 6. Inference and Submission
    print("Generating submission...")

    # Load best model
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model path not found, using current model state.")

    model.eval()
    test_preds = []

    with torch.no_grad():
        for x_num, x_cat in test_loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)

            logits = model(x_num, x_cat).squeeze()
            preds = torch.sigmoid(logits)
            test_preds.append(preds.cpu().numpy())

    test_preds = np.concatenate(test_preds)

    # Retrieve Test IDs
    # Since get_dataloaders doesn't return IDs, we load them from the cache
    # The cache file is created by library.data_loader.process_data
    cache_path = os.path.join(Config.CACHE_DIR, "processed_data.npz")
    if os.path.exists(cache_path):
        data_cache = np.load(cache_path, allow_pickle=True)
        test_ids = data_cache["test_ids"]
    else:
        # Fallback if cache is missing (should not happen if get_dataloaders worked)
        print("Cache file missing. Reloading test metadata for IDs.")
        test_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "test_metadata.csv"))
        test_ids = test_meta["id"].values

    # Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub = pd.DataFrame({"id": test_ids, "target": test_preds})
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
