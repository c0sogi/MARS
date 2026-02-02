import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from library.config import (
    PIFEModel,
    set_seed,
    CACHE_DIR,
    SUBMISSION_PATH,
    NUM_STREAMS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    PATIENCE,
)
from library.data_utils import process_data, get_dataloaders


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    """
    Trains the model for one epoch using the multi-stream loss objective.
    Loss = Sum(BCE(stream_i, target)) for all streams.
    """
    model.train()
    running_loss = 0.0

    for x_cat, x_cont, y in loader:
        x_cat, x_cont, y = x_cat.to(device), x_cont.to(device), y.to(device)

        optimizer.zero_grad()

        # Forward pass: [batch_size, NUM_STREAMS]
        outputs = model(x_cat, x_cont)

        # Calculate loss: Sum of BCE for each independent stream
        loss = 0
        for i in range(NUM_STREAMS):
            loss += criterion(outputs[:, i : i + 1], y)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Prediction = Mean(Sigmoid(stream_i)) for all streams.
    """
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for x_cat, x_cont, y in loader:
            x_cat, x_cont = x_cat.to(device), x_cont.to(device)
            outputs = model(x_cat, x_cont)

            # Ensemble prediction: Average probability across streams
            probs = torch.sigmoid(outputs).mean(dim=1)

            preds.extend(probs.cpu().numpy())
            targets.extend(y.numpy())

    auc = roc_auc_score(targets, preds)
    return auc


def predict_test(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for x_cat, x_cont in loader:
            x_cat, x_cont = x_cat.to(device), x_cont.to(device)
            outputs = model(x_cat, x_cont)

            # Ensemble prediction: Average probability across streams
            probs = torch.sigmoid(outputs).mean(dim=1)
            preds.extend(probs.cpu().numpy())

    return np.array(preds)


def train_and_evaluate(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    patience=PATIENCE,
    load_cached_data=True,
    max_samples=None,
):
    """
    Main function to orchestrate training, evaluation, and submission generation.
    """
    # 1. Setup
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    # process_data handles caching internally as per library.data_utils
    data = process_data(load_cached_data=load_cached_data)

    # Debugging: Limit samples if requested
    if max_samples is not None:
        print(f"Debug Mode: Limiting training data to {max_samples} samples.")
        for k in ["X_train_cat", "X_train_cont", "y_train"]:
            data[k] = data[k][:max_samples]

    train_loader, val_loader, test_loader = get_dataloaders(data, batch_size)

    # 3. Model Initialization
    num_cont = data["X_train_cont"].shape[1]
    vocab_sizes = data["vocab_sizes"]
    model = PIFEModel(vocab_sizes, num_cont).to(device)

    # 4. Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # OneCycleLR Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=100.0,
    )

    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(CACHE_DIR, "best_model.pth")
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_auc = validate(model, val_loader, device)

        # Print full precision
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss} | Val AUC: {val_auc}")

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Best Validation AUC: {best_auc}")

    # 6. Submission Generation
    # Load best model weights
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))

    print("Generating predictions on test set...")
    test_preds = predict_test(model, test_loader, device)

    # Save submission
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission = pd.DataFrame({"id": data["test_ids"], "target": test_preds})

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
