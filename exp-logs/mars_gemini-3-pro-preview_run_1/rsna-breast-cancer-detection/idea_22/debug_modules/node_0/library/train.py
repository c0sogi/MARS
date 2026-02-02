import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, probabilistic_f1
from library.data import get_dataloaders
from library.model import SiameseEfficientNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        target_img = batch["target"].to(device)
        contra_img = batch["contra"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)

        optimizer.zero_grad()

        logits = model(target_img, contra_img)
        loss = criterion(logits, labels)

        loss.backward()

        # Strategy: Disable Gradient Clipping to allow large updates for minority class
        # if Config.MAX_GRAD_NORM is not None:
        #    torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item() * target_img.size(0)

        probs = torch.sigmoid(logits).detach().cpu()
        all_preds.extend(probs.numpy())
        all_targets.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    pf1 = probabilistic_f1(np.array(all_targets), np.array(all_preds))

    return epoch_loss, pf1


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            target_img = batch["target"].to(device)
            contra_img = batch["contra"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            logits = model(target_img, contra_img)
            loss = criterion(logits, labels)

            running_loss += loss.item() * target_img.size(0)

            probs = torch.sigmoid(logits).cpu()
            all_preds.extend(probs.numpy())
            all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    pf1 = probabilistic_f1(np.array(all_targets), np.array(all_preds))

    return epoch_loss, pf1


def run_training(num_epochs=Config.NUM_EPOCHS):
    """
    Main training loop with Early Stopping and Checkpointing.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Data
    train_loader, val_loader, _ = get_dataloaders()

    # Model
    model = SiameseEfficientNet().to(device)

    # Loss (Weighted BCE)
    # Aggressive positive weighting to approximate inverse class frequency
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=Config.ETA_MIN
    )

    # Tracking
    best_pf1 = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {device} for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        train_loss, train_pf1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_pf1 = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.6f} | Train pF1: {train_pf1}")
        print(f"  Val Loss:   {val_loss:.6f} | Val pF1:   {val_pf1}")

        # Checkpointing & Early Stopping
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  [Saved Best Model]")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Val pF1: {best_pf1}")
    return best_model_path


def generate_submission_file():
    """
    Generates predictions for the test set and saves submission.csv.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Data
    _, _, test_loader = get_dataloaders()

    # Load Model
    model = SiameseEfficientNet().to(device)
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print("Warning: No trained model found. Using random initialization.")

    model.eval()

    results = []  # List of dicts

    with torch.no_grad():
        for batch in test_loader:
            target_img = batch["target"].to(device)
            contra_img = batch["contra"].to(device)
            pred_ids = batch["prediction_id"]

            logits = model(target_img, contra_img)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            for pid, prob in zip(pred_ids, probs):
                results.append({"prediction_id": pid, "cancer": prob})

    # Aggregate predictions
    df_res = pd.DataFrame(results)

    # Group by prediction_id and take max (handling multiple views per breast)
    df_sub = df_res.groupby("prediction_id", as_index=False)["cancer"].max()

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
