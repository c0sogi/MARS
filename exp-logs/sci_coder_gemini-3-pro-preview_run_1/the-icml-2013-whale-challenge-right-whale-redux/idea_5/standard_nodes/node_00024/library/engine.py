import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import set_seed, mixup_data, mixup_criterion
from library.dataset import get_dataloaders
from library.model import TimePreservingEfficientNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch using Mixup augmentation.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device).float().view(-1, 1)  # Ensure shape [Batch, 1]

        # Apply Mixup
        data, target_a, target_b, lam = mixup_data(
            data, target, Config.MIXUP_ALPHA, device
        )

        optimizer.zero_grad()
        output = model(data)

        # Compute Loss
        loss = mixup_criterion(criterion, output, target_a, target_b, lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns the average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device).float().view(-1, 1)

            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item() * data.size(0)

            # Apply sigmoid for probabilities
            preds = torch.sigmoid(output)

            all_targets.append(target.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    total_loss = running_loss / len(loader.dataset)

    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)

    # Calculate AUC
    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle case where only one class is present in batch (unlikely in full val set)
        auc_score = 0.5

    return total_loss, auc_score


def predict(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    results = []

    # Load test metadata to get clip names matching the loader order
    # The loader is created from test.csv sequentially without shuffling
    test_df = pd.read_csv(Config.TEST_CSV)
    if Config.DEBUG:
        test_df = test_df.iloc[:100]

    clips = test_df["clip"].values
    probabilities = []

    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            output = model(data)
            preds = torch.sigmoid(output)
            probabilities.extend(preds.cpu().numpy().flatten().tolist())

    # Create submission DataFrame
    submission = pd.DataFrame({"clip": clips, "probability": probabilities})

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training():
    """
    Main execution function for training and evaluation.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    Config.setup()
    Config.print_config()

    # 2. Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(debug=Config.DEBUG)

    # 3. Model
    print("Initializing model...")
    model = TimePreservingEfficientNet()
    model = model.to(device)

    # 4. Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        verbose=True,
        min_lr=Config.MIN_LR,
    )

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0
    early_stopping_patience = 7  # Stop if no improvement for 7 epochs
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step(val_auc)

        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.10f} | "
            f"LR: {current_lr:.2e}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved! AUC: {best_auc:.10f}")
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # 6. Final Prediction
    print("Loading best model for prediction...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("Generating submission...")
    predict(model, test_loader, device, Config.SUBMISSION_PATH)

    return best_auc
