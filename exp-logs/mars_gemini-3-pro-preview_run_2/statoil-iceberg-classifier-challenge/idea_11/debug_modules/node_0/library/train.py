import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import os
import time
from library import config, utils, model, data_loader

# ==========================================
# 1. TRAINING & VALIDATION LOOPS
# ==========================================


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # Match shape (Batch, 1)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, angles)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss += loss.item() * images.size(0)

        # Accuracy (Sigmoid > 0.5 is equivalent to Logits > 0)
        predicted = (outputs > 0).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predicted = (outputs > 0).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


# ==========================================
# 2. FOLD ORCHESTRATION
# ==========================================


def run_fold(fold_idx):
    """
    Trains a single model instance (Ensemble Member).
    Uses a unique seed for initialization diversity.
    """
    # Set seed for this fold to ensure diversity in initialization
    fold_seed = config.SEED + fold_idx
    utils.seed_everything(fold_seed)

    print(f"\n{'='*20}")
    print(f"Starting Training: Fold/Run {fold_idx}")
    print(f"{'='*20}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Get Loaders (Note: These use the fixed metadata split)
    # We reload them to ensure clean state if needed, though caching handles overhead
    train_loader, val_loader, _ = data_loader.get_loaders(load_cached_data=True)

    # Initialize Model
    net = model.WEBN().to(device)

    # Criterion: Binary Cross Entropy with Logits
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer: Adam
    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    # Scheduler: Reduce LR on Plateau
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config.PATIENCE // 2,
        verbose=True,
        min_lr=config.MIN_LR,
    )

    # Early Stopping
    early_stopping = utils.EarlyStopping(patience=config.PATIENCE, verbose=True)

    # Training Loop
    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            net, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(net, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Scheduler Step
        scheduler.step(val_loss)

        # Early Stopping Step
        early_stopping(val_loss, net)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Save Best Model
    save_path = config.get_model_path(fold_idx)
    torch.save(early_stopping.best_model_state, save_path)
    print(f"Best model for Fold {fold_idx} saved to {save_path}")

    return early_stopping.best_score


# ==========================================
# 3. SUBMISSION GENERATION
# ==========================================


def generate_submission():
    """
    Generates predictions using an ensemble of all trained models.
    """
    print(f"\n{'='*20}")
    print("Generating Submission (Ensemble)")
    print(f"{'='*20}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get Test Loader
    _, _, test_loader = data_loader.get_loaders(load_cached_data=True)

    # Initialize array for accumulated probabilities
    # We need to know the size of the test set.
    # data_loader doesn't expose len directly, but we can infer from metadata or loader.
    # We'll collect predictions in a list first.

    # Load Test Metadata to get IDs (order is preserved in test_loader)
    df_test = pd.read_csv(config.TEST_META_CSV)
    test_ids = df_test["id"].values

    # Placeholder for ensemble predictions
    ensemble_preds = np.zeros((len(test_ids), 1))

    # Iterate over all folds
    for fold_idx in range(config.N_FOLDS):
        model_path = config.get_model_path(fold_idx)
        if not os.path.exists(model_path):
            print(
                f"Warning: Model for fold {fold_idx} not found at {model_path}. Skipping."
            )
            continue

        print(f"Loading model from {model_path}...")
        net = model.WEBN().to(device)
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.eval()

        fold_preds = []

        with torch.no_grad():
            for images, angles in test_loader:
                images = images.to(device)
                angles = angles.to(device)

                # Forward
                logits = net(images, angles)
                probs = torch.sigmoid(logits)

                fold_preds.append(probs.cpu().numpy())

        # Concatenate batches
        fold_preds = np.vstack(fold_preds)

        # Accumulate
        ensemble_preds += fold_preds

    # Average predictions
    avg_preds = ensemble_preds / config.N_FOLDS

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds.flatten()})

    # Save
    print(f"Saving submission to {config.SUBMISSION_PATH}...")
    df_sub.to_csv(config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")


# ==========================================
# 4. MAIN EXECUTION
# ==========================================


def main():
    # 1. Train Ensemble Members
    # We train N_FOLDS models. Since the data split is fixed by metadata,
    # these act as a Deep Ensemble (same data, different seeds).
    for fold_idx in range(config.N_FOLDS):
        run_fold(fold_idx)

    # 2. Generate Submission
    generate_submission()


# Execute main
main()
