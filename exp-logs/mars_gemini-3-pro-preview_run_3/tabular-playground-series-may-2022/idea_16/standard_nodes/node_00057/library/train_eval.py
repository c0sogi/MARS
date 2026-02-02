import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.dataset import get_dataloaders
from library.model import FunnelMLP


def set_seed(seed=Config.SEED):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        cat_x = batch["cat_features"].to(device)
        cont_x = batch["cont_features"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)

        optimizer.zero_grad()

        logits = model(cat_x, cont_x)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * targets.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set and returns the ROC AUC score.
    """
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            cat_x = batch["cat_features"].to(device)
            cont_x = batch["cont_features"].to(device)
            targets = batch["target"].to(device)

            logits = model(cat_x, cont_x)
            probs = torch.sigmoid(logits).squeeze(1)

            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    auc = roc_auc_score(all_targets, all_probs)
    return auc


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            cat_x = batch["cat_features"].to(device)
            cont_x = batch["cont_features"].to(device)

            logits = model(cat_x, cont_x)
            probs = torch.sigmoid(logits).squeeze(1)

            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs)


def train_model(train_loader, val_loader, metadata, epochs=Config.EPOCHS, patience=5):
    """
    Main training loop with Early Stopping and Model Checkpointing.
    """
    device = Config.DEVICE
    print(f"Training on device: {device}")

    # Initialize Model
    vocab_sizes = metadata["vocab_sizes"]
    num_cont_features = len(metadata["cont_cols"])

    model = FunnelMLP(vocab_sizes=vocab_sizes, num_cont_features=num_cont_features).to(
        device
    )

    # Optimizer & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.BCEWithLogitsLoss()

    # Scheduler: OneCycleLR
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,  # Standard default
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # Early Stopping Tracking
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_auc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val AUC: {val_auc}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            # print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Val AUC: {best_auc}")

    # Load best model for return
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    return model


def generate_submission(predictions, sample_sub_path, output_path):
    """
    Saves predictions to a CSV file in the required format.
    """
    sub_df = pd.read_csv(sample_sub_path)
    sub_df["target"] = predictions
    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(load_cached_data=True, max_samples=None):
    """
    Orchestrates the full pipeline: Data Loading -> Training -> Prediction -> Submission.
    """
    set_seed()
    Config.create_directories()

    # 1. Load Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader, metadata = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        load_cached_data=load_cached_data,
        max_samples=max_samples,
    )

    # 2. Train Model
    print("Starting Training...")
    model = train_model(train_loader, val_loader, metadata, epochs=Config.EPOCHS)

    # 3. Predict on Test Set
    print("Generating Predictions...")
    test_probs = predict(model, test_loader, Config.DEVICE)

    # 4. Create Submission
    generate_submission(
        test_probs, Config.SAMPLE_SUBMISSION_PATH, Config.SUBMISSION_PATH
    )
