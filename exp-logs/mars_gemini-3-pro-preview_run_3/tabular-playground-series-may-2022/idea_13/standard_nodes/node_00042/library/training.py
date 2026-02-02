import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config, set_seed
from library.model import ManufacturingMLP


def train_epoch(model, loader, optimizer, criterion, scheduler, device):
    """
    Training loop for a single epoch.
    """
    model.train()
    running_loss = 0.0

    for x_cont, x_cat, y in loader:
        x_cont = x_cont.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(x_cont, x_cat).squeeze()
        loss = criterion(logits, y)

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Validation loop. Returns ROC AUC.
    """
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for x_cont, x_cat, y in loader:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)

            logits = model(x_cont, x_cat).squeeze()
            probs = torch.sigmoid(logits)

            preds.extend(probs.cpu().numpy())
            targets.extend(y.numpy())

    return roc_auc_score(targets, preds)


def train_model(train_loader, val_loader, meta):
    """
    Main training function.
    Initializes model, optimizer, scheduler, and runs the training loop with early stopping.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    # Initialize Model
    # meta['vocab_sizes'] is a list of integers representing the vocabulary size for each categorical feature
    model = ManufacturingMLP(
        num_cont=len(meta["cont_cols"]),
        vocab_sizes=meta["vocab_sizes"],
        embed_dim=Config.EMBEDDING_DIM,
        hidden_layers=Config.HIDDEN_LAYERS,
        dropout_rate=Config.DROPOUT_RATE,
    ).to(device)

    # Loss, Optimizer, Scheduler
    criterion = nn.BCEWithLogitsLoss()

    # AdamW with specific weight decay as requested
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=Config.EPOCHS,
        pct_start=0.1,
    )

    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, device
        )
        val_auc = validate(model, val_loader, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val AUC: {val_auc}"
        )

        # Checkpointing and Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Best Validation AUC: {best_auc}")
    return best_auc


def generate_submission(test_loader, meta):
    """
    Generates predictions for the test set using the best saved model.
    """
    print("Generating submission...")
    device = Config.DEVICE

    # Re-initialize model structure
    model = ManufacturingMLP(
        num_cont=len(meta["cont_cols"]),
        vocab_sizes=meta["vocab_sizes"],
        embed_dim=Config.EMBEDDING_DIM,
        hidden_layers=Config.HIDDEN_LAYERS,
        dropout_rate=Config.DROPOUT_RATE,
    ).to(device)

    # Load best weights
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    ids_list = []
    preds_list = []

    with torch.no_grad():
        for x_cont, x_cat, ids in test_loader:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)

            logits = model(x_cont, x_cat).squeeze()
            probs = torch.sigmoid(logits)

            ids_list.extend(ids)
            preds_list.extend(probs.cpu().numpy())

    submission = pd.DataFrame({"id": ids_list, "target": preds_list})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
