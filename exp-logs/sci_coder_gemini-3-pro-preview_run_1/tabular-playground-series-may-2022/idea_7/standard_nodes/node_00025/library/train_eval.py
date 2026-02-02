import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.data_utils import preprocess_pipeline
from library.dataset import create_dataloaders
from library.model import ManufacturingTransformer


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        x_num = batch["numerical"].to(device)
        x_seq = batch["sequence"].to(device)
        targets = batch["target"].to(device)
        batch_size = x_num.size(0)

        optimizer.zero_grad()

        logits = model(x_num, x_seq)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and AUC.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_logits = []

    with torch.no_grad():
        for batch in loader:
            x_num = batch["numerical"].to(device)
            x_seq = batch["sequence"].to(device)
            targets = batch["target"].to(device)
            batch_size = x_num.size(0)

            logits = model(x_num, x_seq)
            loss = criterion(logits, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate and compute AUC
    y_true = np.vstack(all_targets)
    y_scores = np.vstack(all_logits)
    # y_scores are logits, but AUC is rank-based so logits are fine.
    # However, for consistency with probability interpretation elsewhere, we can use sigmoid,
    # but raw logits work identically for ROC AUC.

    auc_score = roc_auc_score(y_true, y_scores)

    return epoch_loss, auc_score


def predict(model, loader, device):
    """
    Generates predictions (probabilities) for the test set.
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            x_num = batch["numerical"].to(device)
            x_seq = batch["sequence"].to(device)

            logits = model(x_num, x_seq)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())

    return np.vstack(all_probs).flatten()


def run_training(load_cached_data=True):
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    # preprocess_pipeline handles caching internally based on the flag
    data_dict, vocab_size = preprocess_pipeline(load_cached_data=load_cached_data)

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dict,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # 3. Model Initialization
    # Determine number of numerical features from the data
    num_numerical_features = data_dict["X_num_train"].shape[1]

    model = ManufacturingTransformer(
        num_numerical_features=num_numerical_features,
        vocab_size=vocab_size,
        seq_len=Config.SEQ_LEN,
    ).to(device)

    # 4. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.BCEWithLogitsLoss()

    # OneCycleLR Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Early Stopping and Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Prediction and Submission
    print("Loading best model for prediction...")
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device))

    predictions = predict(model, test_loader, device)

    ids = data_dict["ids_test"]

    submission = pd.DataFrame({"id": ids, "target": predictions})

    # Ensure output directory exists (Config.setup() does this, but good to be safe)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
