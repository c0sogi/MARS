import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything, get_device
from library.data_processing import process_data, ManufacturingDataset
from library.model import MORPE


def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    """
    Trains the model for one epoch.
    Calculates the loss as the sum of independent losses from each stream.
    """
    model.train()
    total_loss = 0.0

    for cat, cont, target in loader:
        cat = cat.to(device)
        cont = cont.to(device)
        target = target.to(device).unsqueeze(1)

        optimizer.zero_grad()

        # Forward pass returns a list of outputs from each stream
        outputs = model(cat, cont)

        # Loss is the sum of BCE losses for each stream
        loss = sum(criterion(out, target) for out in outputs)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Predictions are the arithmetic mean of probabilities from all streams.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for cat, cont, target in loader:
            cat = cat.to(device)
            cont = cont.to(device)

            # Forward pass
            outputs = model(cat, cont)

            # Average probabilities across all 5 streams
            # stack: (5, Batch, 1) -> mean(dim=0): (Batch, 1)
            probs = torch.stack([torch.sigmoid(out) for out in outputs]).mean(dim=0)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(target.numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    auc = roc_auc_score(all_targets, all_preds)
    return auc


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for cat, cont in loader:
            cat = cat.to(device)
            cont = cont.to(device)

            outputs = model(cat, cont)
            probs = torch.stack([torch.sigmoid(out) for out in outputs]).mean(dim=0)
            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds).flatten()


def run_training(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
    patience=5,
    sample_size=None,
):
    """
    Main pipeline function to train the MORPE model and generate submission.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoaders.
        load_cached_data (bool): Whether to load processed data from cache.
        patience (int): Early stopping patience.
        sample_size (int, optional): If provided, subsets the data for debugging.
    """
    seed_everything(Config.SEED)
    device = get_device()

    # 1. Data Processing
    # process_data handles caching internally
    df_train, df_val, df_test, meta_dict = process_data(
        load_cached_data=load_cached_data
    )

    # Optional debugging subset
    if sample_size is not None:
        print(f"Subsetting data to {sample_size} samples for debugging...")
        df_train = df_train.iloc[:sample_size]
        df_val = df_val.iloc[: min(sample_size, len(df_val))]

    cat_cols = meta_dict["cat_cols"]
    cont_cols = meta_dict["cont_cols"]
    vocab_sizes_dict = meta_dict["vocab_sizes"]
    vocab_sizes = [vocab_sizes_dict[c] for c in cat_cols]

    # 2. Dataset and Loaders
    train_ds = ManufacturingDataset(df_train, cat_cols, cont_cols, "target")
    val_ds = ManufacturingDataset(df_val, cat_cols, cont_cols, "target")
    test_ds = ManufacturingDataset(df_test, cat_cols, cont_cols, None)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = MORPE(
        vocab_sizes_list=vocab_sizes,
        num_cont=len(cont_cols),
        embed_dim=Config.EMBED_DIM,
        stream_configs=Config.STREAMS,
    ).to(device)

    # 4. Optimizer Configuration (Heterogeneous Weight Decay)
    # Map specific weight decay values to each stream's parameters
    param_groups = []
    for i, stream in enumerate(model.streams):
        wd = Config.STREAMS[i]["wd"]
        param_groups.append({"params": stream.parameters(), "weight_decay": wd})

    optimizer = optim.AdamW(param_groups, lr=Config.MAX_LR)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
    )

    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    print(f"Starting training on {device}...")
    best_auc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_auc = validate(model, val_loader, device)

        # Print full precision as requested
        print(f"Epoch {epoch+1:02d} | Loss: {train_loss:.6f} | Val AUC: {val_auc}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best Validation AUC: {best_auc}")

    # 6. Submission
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("Generating predictions on test set...")
    test_preds = predict(model, test_loader, device)

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission = pd.DataFrame({"id": df_test["id"], "target": test_preds})
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
