import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.data import get_dataloaders
from library.model import get_model
from library.utils import seed_everything, compute_auc


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # Unpack batch: x_seq, x_cont, y
        x_seq, x_cont, y = batch

        x_seq = x_seq.to(device)
        x_cont = x_cont.to(device)
        y = y.to(device).unsqueeze(1)  # Reshape to (B, 1) for BCEWithLogitsLoss

        optimizer.zero_grad()

        # Forward pass
        logits = model(x_seq, x_cont)

        # Compute loss
        loss = criterion(logits, y)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and AUC.
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            x_seq, x_cont, y = batch

            x_seq = x_seq.to(device)
            x_cont = x_cont.to(device)
            y = y.to(device).unsqueeze(1)

            logits = model(x_seq, x_cont)
            loss = criterion(logits, y)

            running_loss += loss.item()
            num_batches += 1

            # Store for AUC calculation
            probs = torch.sigmoid(logits)
            all_targets.append(y.cpu())
            all_preds.append(probs.cpu())

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    # Concatenate all batches
    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()

    auc_score = compute_auc(all_targets, all_preds)

    return avg_loss, auc_score


def run_training(
    config=None, load_cached_data=True, max_train_samples=None, max_val_samples=None
):
    """
    Main function to run the training pipeline, validation, and submission generation.
    """
    if config is None:
        config = Config()

    # 1. Setup
    seed_everything(config.SEED)
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    print(f"Starting training with device: {config.DEVICE}")

    # 2. Data Loading
    train_dl, val_dl, test_dl, data_dict = get_dataloaders(
        config,
        load_cached_data=load_cached_data,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )

    # 3. Model Initialization
    model = get_model(config, vocab_size=data_dict["vocab_size"])

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config.SCHEDULER_STEP_SIZE, gamma=config.SCHEDULER_GAMMA
    )

    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    patience = 10
    patience_counter = 0

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_dl, optimizer, criterion, config.DEVICE
        )
        val_loss, val_auc = validate(model, val_dl, criterion, config.DEVICE)

        # Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print metrics (full precision)
        print(
            f"Epoch {epoch}/{config.EPOCHS} | LR: {current_lr} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"New Best AUC! Model saved to {best_model_path}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    # 6. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
    model.eval()

    test_preds = []

    with torch.no_grad():
        for batch in test_dl:
            x_seq, x_cont = batch
            x_seq = x_seq.to(config.DEVICE)
            x_cont = x_cont.to(config.DEVICE)

            logits = model(x_seq, x_cont)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_preds.extend(probs.flatten())

    # 7. Submission
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

    submission_df = pd.DataFrame({"id": data_dict["test_ids"], "target": test_preds})

    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
