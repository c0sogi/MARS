import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.dataset import get_dataloaders
from library.model import ResNet18Model, set_seed


def fit(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    max_lr=Config.MAX_LR,
    weight_decay=Config.WEIGHT_DECAY,
    debug=Config.DEBUG,
    patience=5,
):
    """
    Trains the ResNet18 model with Early Stopping, AMP, and OneCycleLR.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Load data
    loaders = get_dataloaders(batch_size=batch_size, debug=debug)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # Initialize model
    model = ResNet18Model(num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED)
    model.to(device)

    # Setup optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
    )

    scaler = GradScaler()

    best_auc = -float("inf")
    trigger_times = 0

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss_sum = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss_sum += loss.item() * images.size(0)

        avg_train_loss = train_loss_sum / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss_sum = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * images.size(0)

                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_val_loss = val_loss_sum / len(val_loader.dataset)

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Calculate AUC
        aucs = []
        for i in range(Config.NUM_CLASSES):
            # Only calculate AUC if there is more than one class present in the validation set
            if len(np.unique(all_labels[:, i])) > 1:
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                aucs.append(auc)

        avg_auc = np.mean(aucs) if aucs else 0.0

        # Print full precision metrics
        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss} | Val Loss: {avg_val_loss} | Val AUC: {avg_auc}"
        )

        # --- Early Stopping & Saving ---
        if avg_auc > best_auc:
            best_auc = avg_auc
            trigger_times = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping triggered after {patience} epochs.")
                break

    return best_auc


def predict(batch_size=Config.BATCH_SIZE, debug=Config.DEBUG):
    """
    Runs inference on the test set and creates the submission file.
    """
    device = Config.DEVICE
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Load test data
    loaders = get_dataloaders(batch_size=batch_size, debug=debug)
    test_loader = loaders["test"]

    # Initialize model and load weights
    model = ResNet18Model(num_classes=Config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_probs = []
    all_uids = []

    print("Starting inference...")

    with torch.no_grad():
        for images, uids in test_loader:
            images = images.to(device, non_blocking=True)

            with autocast():
                logits = model(images)
                probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_uids.extend(uids)

    all_probs = np.concatenate(all_probs)

    # Prepare submission dataframe
    submission_data = {"StudyInstanceUID": all_uids}
    for i, col_name in enumerate(Config.TARGET_COLS):
        submission_data[col_name] = all_probs[:, i]

    df_sub = pd.DataFrame(submission_data)

    # Save submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
