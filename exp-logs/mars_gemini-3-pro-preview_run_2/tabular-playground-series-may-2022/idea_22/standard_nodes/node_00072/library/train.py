import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, compute_auc
from library.data import get_dataloaders
from library.model import SustainedDepthHybridNet


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for batch in loader:
        # Move data to device
        cont_data = batch["cont"].to(device)
        seq_data = batch["seq"].to(device)
        targets = batch["target"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(cont_data, seq_data)

        # Compute loss
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Update statistics
        running_loss += loss.item() * targets.size(0)
        count += targets.size(0)

    return running_loss / count


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and AUC.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            cont_data = batch["cont"].to(device)
            seq_data = batch["seq"].to(device)
            targets = batch["target"].to(device)

            logits = model(cont_data, seq_data)
            loss = criterion(logits, targets)

            running_loss += loss.item() * targets.size(0)
            count += targets.size(0)

            # Store for AUC calculation (apply sigmoid for probability)
            probs = torch.sigmoid(logits)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    avg_loss = running_loss / count

    # Concatenate all batches
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)

    auc_score = compute_auc(y_true, y_pred)

    return avg_loss, auc_score


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            cont_data = batch["cont"].to(device)
            seq_data = batch["seq"].to(device)

            logits = model(cont_data, seq_data)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds)


def run_training(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    device=Config.DEVICE,
    save_path=Config.MODEL_SAVE_PATH,
    submission_path=Config.SUBMISSION_PATH,
    patience=12,  # Allow enough patience for LR scheduler step (every 10 epochs)
):
    """
    Main training loop with Early Stopping and Checkpointing.
    """
    # 1. Setup
    seed_everything(Config.RANDOM_STATE)
    print(f"Starting training on device: {device}")

    # 2. Data
    print("Loading data...")
    loaders = get_dataloaders(batch_size=batch_size)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]
    test_ids = loaders["test_ids"]

    # 3. Model
    model = SustainedDepthHybridNet().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # BCEWithLogitsLoss includes Sigmoid, numerically stable
    criterion = nn.BCEWithLogitsLoss()

    # Scheduler: Decay LR by 0.1 every 10 epochs
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Logging (Full Precision)
        print(
            f"Epoch {epoch} | LR: {current_lr:.1e} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            # print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {epoch} epochs. Best AUC: {best_auc}"
            )
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")

    # 6. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(save_path, map_location=device))

    print("Generating predictions on test set...")
    predictions = predict(model, test_loader, device)

    # 7. Submission
    print(f"Saving submission to {submission_path}...")
    submission_df = pd.DataFrame({"id": test_ids, "target": predictions})

    # Ensure directory exists
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    print("Submission saved successfully.")
