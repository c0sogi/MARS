import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Import pre-defined components and constants from the library
from library.config import (
    PIFEModel,
    TabularDataset,
    process_data,
    set_seed,
    CACHE_DIR,
    SUBMISSION_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_STREAMS,
    PATIENCE,
    EPOCHS,
)


def train_model(epochs=EPOCHS, max_samples=None, load_cached_data=True):
    """
    Trains the PIFE model with the specified hyperparameters.

    Args:
        epochs (int): Maximum number of training epochs.
        max_samples (int, optional): If set, limits the training data size for debugging.
        load_cached_data (bool): Whether to attempt loading processed data from cache.

    Returns:
        model (nn.Module): The trained model with best weights loaded.
        device (torch.device): The device used for training.
        data (dict): The processed data dictionary.
    """
    # Ensure reproducibility
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and process data
    data = process_data(load_cached_data=load_cached_data)

    X_train_cat = data["X_train_cat"]
    X_train_cont = data["X_train_cont"]
    y_train = data["y_train"]

    # Handle debugging subset if requested
    if max_samples is not None:
        print(f"Debug Mode: Limiting training data to {max_samples} samples.")
        X_train_cat = X_train_cat[:max_samples]
        X_train_cont = X_train_cont[:max_samples]
        y_train = y_train[:max_samples]

    # Create Datasets and Loaders
    train_dataset = TabularDataset(X_train_cat, X_train_cont, y_train)
    val_dataset = TabularDataset(data["X_val_cat"], data["X_val_cont"], data["y_val"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize Model
    # Determine input dimensions from data
    num_cont = X_train_cont.shape[1]
    vocab_sizes = data["vocab_sizes"]

    model = PIFEModel(vocab_sizes, num_cont).to(device)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=100.0,
    )

    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    best_val_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(CACHE_DIR, "best_model.pth")

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x_cat, x_cont, y in train_loader:
            x_cat, x_cont, y = x_cat.to(device), x_cont.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x_cat, x_cont)  # Shape: [batch, NUM_STREAMS]

            # Calculate loss: Sum of BCE for each independent stream
            loss = 0
            for i in range(NUM_STREAMS):
                loss += criterion(outputs[:, i : i + 1], y)

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for x_cat, x_cont, y in val_loader:
                x_cat, x_cont = x_cat.to(device), x_cont.to(device)
                outputs = model(x_cat, x_cont)

                # Ensemble prediction: Average probability across streams
                probs = torch.sigmoid(outputs).mean(dim=1)
                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(y.numpy())

        val_auc = roc_auc_score(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.6f} | Val AUC: {val_auc:.10f}"
        )

        # Checkpointing and Early Stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Validation AUC: {best_val_auc:.10f}")

    # Load best weights before returning
    model.load_state_dict(torch.load(best_model_path))
    return model, device, data


def generate_submission(model, device, data):
    """
    Generates predictions for the test set and saves the submission file.
    """
    test_dataset = TabularDataset(data["X_test_cat"], data["X_test_cont"])
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    model.eval()
    all_preds = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for x_cat, x_cont in test_loader:
            x_cat, x_cont = x_cat.to(device), x_cont.to(device)
            outputs = model(x_cat, x_cont)

            # Ensemble prediction: Average probability across streams
            probs = torch.sigmoid(outputs).mean(dim=1)
            all_preds.extend(probs.cpu().numpy())

    # Create submission dataframe
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission = pd.DataFrame({"id": data["test_ids"], "target": all_preds})

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
