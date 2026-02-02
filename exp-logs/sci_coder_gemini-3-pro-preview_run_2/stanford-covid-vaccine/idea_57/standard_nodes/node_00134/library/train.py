import os
import torch
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import (
    seed_everything,
    MCRMSELoss,
    calculate_global_mcrmse,
    format_submission,
)
from library.data import get_loaders
from library.model import GCDARN


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (features, partner_indices, targets, _) in enumerate(loader):
        features = features.to(device)
        partner_indices = partner_indices.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass returns outputs from both refinement steps
        # out1: Pass 1 (Zero Feedback)
        # out2: Pass 2 (Feedback from detached out1)
        out1, out2 = model(features, partner_indices)

        # Calculate loss
        loss1 = criterion(out1, targets)
        loss2 = criterion(out2, targets)

        # Combined loss: Focus on final output, auxiliary loss on first pass
        total_loss = loss2 + 0.5 * loss1

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using Global MCRMSE.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, partner_indices, targets, _ in loader:
            features = features.to(device)
            partner_indices = partner_indices.to(device)

            # Forward pass
            # We only care about the final refined output (out2) for validation
            _, out2 = model(features, partner_indices)

            all_preds.append(out2.cpu().numpy())
            all_targets.append(targets.numpy())

    # Calculate Global MCRMSE
    # Concatenation happens inside the utility function
    score = calculate_global_mcrmse(all_preds, all_targets)

    return score


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for features, partner_indices, _, ids in loader:
            features = features.to(device)
            partner_indices = partner_indices.to(device)

            _, out2 = model(features, partner_indices)

            all_preds.append(out2.cpu().numpy())
            all_ids.extend(ids)

    return np.concatenate(all_preds, axis=0), all_ids


def run_training(load_cached_data=True, num_epochs=Config.EPOCHS):
    """
    Main training pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    print(f"Device: {device}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")

    # 2. Data Loaders
    train_loader, val_loader, test_loader = get_loaders(
        load_cached_data=load_cached_data
    )

    # 3. Model, Optimizer, Loss
    model = GCDARN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = MCRMSELoss().to(device)

    # 4. Training Loop
    best_score = float("inf")
    patience = 7
    patience_counter = 0

    print("\nStarting training...")

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_score = validate(model, val_loader, device)

        # Update scheduler
        scheduler.step(val_score)

        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score}"
        )

        # Save best model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  >>> New Best Model Saved (Score: {best_score})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # 5. Inference
    print("\nLoading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    print("Generating test predictions...")
    test_preds, test_ids = predict(model, test_loader, device)

    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    format_submission(test_ids, test_preds, save_path=Config.SUBMISSION_PATH)

    print("Done.")
