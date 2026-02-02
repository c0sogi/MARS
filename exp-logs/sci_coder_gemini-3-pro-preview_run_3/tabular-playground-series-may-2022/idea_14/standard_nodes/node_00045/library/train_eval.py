import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os

from library.config import Config, set_seed
from library.data import preprocess_data
from library.model import ManufacturingMLP
from library.utils import compute_auc


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for x_cat, x_cont, y in dataloader:
        x_cat = x_cat.to(device)
        x_cont = x_cont.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x_cat, x_cont)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and AUC score.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_cat, x_cont, y in dataloader:
            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)
            y = y.to(device)

            logits = model(x_cat, x_cont)
            loss = criterion(logits, y)
            running_loss += loss.item()

            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    auc_score = compute_auc(all_targets, all_preds)

    return avg_loss, auc_score


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for x_cat, x_cont in test_loader:
            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)

            logits = model(x_cat, x_cont)
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy().flatten())

    # Load test IDs
    df_test = pd.read_csv(Config.TEST_PATH)
    submission = pd.DataFrame({"id": df_test["id"], "target": all_preds})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def train_model(load_cached_data=True, epochs=Config.EPOCHS, patience=Config.PATIENCE):
    """
    Main function to train the model, perform early stopping, and generate submission.
    """
    set_seed()

    # 1. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader, vocab_sizes, num_cont = preprocess_data(
        load_cached_data=load_cached_data, batch_size=Config.BATCH_SIZE
    )

    # 2. Model Initialization
    device = Config.DEVICE
    model = ManufacturingMLP(vocab_sizes, num_cont, Config).to(device)

    # 3. Optimization
    # Use Adam instead of AdamW (Cite solution_lesson_node_00023)
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR requires steps_per_epoch
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )

    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.10f} | Val Loss: {val_loss:.10f} | Val AUC: {val_auc:.10f}"
        )

        # Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"--> New Best Model Saved (AUC: {best_auc:.10f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # 5. Submission
    print("Loading best model for submission...")
    best_model = ManufacturingMLP(vocab_sizes, num_cont, Config).to(device)
    best_model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    generate_submission(best_model, test_loader, device, Config.SUBMISSION_PATH)
