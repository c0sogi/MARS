import os
import time
import torch
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

from library.config import Config
from library.utils import set_seed, mcrmse_loss, GlobalMCRMSE, format_submission
from library.data import get_dataloaders
from library.model import PFDRN


def train_epoch(model, loader, optimizer, device):
    """
    Executes one training epoch.
    Computes loss as: L_total = L_pass2 + 0.5 * L_pass1
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        partner_indices = batch["partner_indices"].to(device)
        pairing_mask = batch["pairing_mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass returns tuple (preds_1, preds_2) in training mode
        preds_1, preds_2 = model(inputs, partner_indices, pairing_mask)

        # Calculate loss for both passes
        loss_1 = mcrmse_loss(preds_1, targets)
        loss_2 = mcrmse_loss(preds_2, targets)

        # Weighted sum
        loss = Config.PASS2_WEIGHT * loss_2 + Config.PASS1_WEIGHT * loss_1

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using Global MCRMSE.
    """
    model.eval()
    metric = GlobalMCRMSE()

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            pairing_mask = batch["pairing_mask"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass returns only preds_2 in eval mode
            preds = model(inputs, partner_indices, pairing_mask)

            metric.update(preds, targets)

    return metric.compute()


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            partner_indices = batch["partner_indices"].to(device)
            pairing_mask = batch["pairing_mask"].to(device)
            ids = batch["id"]

            # Forward pass returns only preds_2 in eval mode
            preds = model(inputs, partner_indices, pairing_mask)

            all_preds.append(preds.cpu().numpy())
            all_ids.extend(ids)

    return np.concatenate(all_preds, axis=0), all_ids


def run_training():
    """
    Main function to run the training loop, validation, and submission generation.
    """
    # 1. Setup
    set_seed(Config.SEED)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    device = torch.device(Config.DEVICE)

    print(f"Device: {device}")
    print(f"Artifacts will be saved to: {Config.WORKING_DIR}")

    # 2. Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # 3. Model
    print("Initializing PF-DRN Model...")
    model = PFDRN().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # 5. Training Loop
    best_score = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    patience_counter = 0

    print("\nStarting Training...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step(val_score)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_score} | "  # Full precision as requested
            f"Time: {elapsed:.2f}s"
        )

        # Checkpointing & Early Stopping
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New Best Model Saved! Score: {best_score}")
        else:
            patience_counter += 1
            print(f"  ... Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("\nEarly stopping triggered.")
            break

    # 6. Inference & Submission
    print("\nLoading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("Generating predictions on Test Set...")
    test_preds, test_ids = predict(model, test_loader, device)

    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    format_submission(test_ids, test_preds, save_path=Config.SUBMISSION_PATH)

    print("Done.")
