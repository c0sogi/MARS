import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint
from library.data_processing import get_data_loaders
from library.model import MultiGranularityNet


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        # Unpack batch based on dataset definition (cont, cat, target)
        x_cont, x_cat, y = batch

        x_cont = x_cont.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)

        batch_size = x_cont.size(0)

        # Forward pass
        optimizer.zero_grad()
        logits = model(x_cont, x_cat)
        loss = criterion(logits, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set using ROC AUC.
    """
    model.eval()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            x_cont, x_cat, y = batch

            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device)

            logits = model(x_cont, x_cat)
            probs = torch.sigmoid(logits)

            all_targets.append(y.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    auc_score = roc_auc_score(all_targets, all_preds)
    return auc_score


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            # Test loader returns (cont, cat) only
            x_cont, x_cat = batch

            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)

            logits = model(x_cont, x_cat)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds)


def run_training():
    """
    Main execution function:
    1. Sets up data loaders.
    2. Initializes model, optimizer, scheduler.
    3. Runs training loop with early stopping.
    4. Evaluates and saves best model.
    5. Generates submission file.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Data Loading
    print("Initializing Data Loaders...")
    train_loader, val_loader, test_loader, vocab_sizes = get_data_loaders(
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
        debug=Config.DEBUG,
        debug_samples=Config.DEBUG_SAMPLES,
    )

    # Determine continuous feature count from a sample batch
    sample_cont, _, _ = next(iter(train_loader))
    num_cont_features = sample_cont.shape[1]
    print(f"Continuous features detected: {num_cont_features}")

    # 2. Model Initialization
    print("Initializing Model...")
    model = MultiGranularityNet(
        vocab_sizes=vocab_sizes, num_cont_features=num_cont_features
    )
    model.to(device)

    # Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.BCEWithLogitsLoss()

    # Scheduler: OneCycleLR
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,  # 30% of training for warmup
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # 3. Training Loop
    print("Starting Training...")
    best_auc = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        val_auc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val AUC: {val_auc}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_auc": best_auc,
                },
                filename=Config.MODEL_PATH,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Training complete. Best Validation AUC: {best_auc}")

    # 4. Inference
    print("Loading best model for inference...")
    # Re-initialize model to ensure clean state or load weights into existing
    # We load the best weights saved during training
    load_checkpoint(model, filename=Config.MODEL_PATH, device=Config.DEVICE)

    print("Generating predictions on test set...")
    predictions = predict(model, test_loader, device)

    # 5. Submission
    print("Creating submission file...")
    # Load test IDs from metadata file to ensure alignment
    test_df = pd.read_csv(Config.TEST_CSV)

    # In debug mode, the loader might have fewer samples than the full CSV if not handled carefully,
    # but get_data_loaders handles subsampling for debug consistently.
    if Config.DEBUG:
        test_df = test_df.iloc[: Config.DEBUG_SAMPLES]

    submission = pd.DataFrame(
        {
            Config.ID_COL: test_df[Config.ID_COL],
            Config.TARGET_COL: predictions.flatten(),
        }
    )

    Config.create_dirs()
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
