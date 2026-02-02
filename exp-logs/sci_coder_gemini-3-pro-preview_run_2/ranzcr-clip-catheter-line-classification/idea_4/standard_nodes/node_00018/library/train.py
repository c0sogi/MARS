import os
import time
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from library.config import Config
from library.utils import seed_everything, get_score, ModelEma
from library.dataset import CatheterDataset, get_transforms
from library.model import CatheterModel


def train_one_epoch(
    model, optimizer, scheduler, dataloader, device, epoch, scaler, ema_model=None
):
    """
    Trains the model for one epoch.
    """
    model.train()

    dataset_size = 0
    running_loss = 0.0

    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)

        batch_size = images.size(0)

        optimizer.zero_grad()

        # Forward pass with AMP (Cite debug_lesson_1)
        with torch.cuda.amp.autocast(enabled=True):
            logits = model(images)
            loss = criterion(logits, labels)

        # Backward pass with Scaler
        scaler.scale(loss).backward()

        # Gradient clipping (unscale first)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        # Update EMA
        if ema_model is not None:
            ema_model.update(model)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, device):
    """
    Validates the model on the validation set.
    """
    model.eval()

    running_loss = 0.0
    dataset_size = 0

    preds = []
    targets = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            batch_size = images.size(0)

            with torch.cuda.amp.autocast(enabled=True):
                logits = model(images)
                loss = criterion(logits, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits)

            preds.append(probs.cpu().numpy())
            targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    # Calculate AUC score
    score = get_score(targets, preds)

    return epoch_loss, score


def predict_and_submit(model, device, output_path):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print(f"Loading test metadata from {Config.test_metadata_path}...")
    df_test = pd.read_csv(Config.test_metadata_path)

    test_dataset = CatheterDataset(
        df_test, transform=get_transforms("valid"), is_test=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    model.eval()
    preds = []

    print("Running inference on test set...")
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device, dtype=torch.float)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds.append(probs.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    # Create submission DataFrame
    # We need StudyInstanceUID and the target columns
    submission_df = pd.DataFrame(preds, columns=Config.target_cols)
    submission_df.insert(0, "StudyInstanceUID", df_test["StudyInstanceUID"])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Saving submission to {output_path}...")
    submission_df.to_csv(output_path, index=False)
    print("Submission saved successfully.")


def run_training():
    """
    Main training loop.
    """
    seed_everything(Config.seed)

    # --- Data Loading ---
    print("Loading metadata...")
    df_train = pd.read_csv(Config.train_metadata_path)
    df_val = pd.read_csv(Config.val_metadata_path)

    if Config.debug:
        print("Debug mode enabled: Subsampling data.")
        df_train = df_train.sample(
            n=min(len(df_train), 500), random_state=Config.seed
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(len(df_val), 100), random_state=Config.seed
        ).reset_index(drop=True)

    # Create Datasets
    train_dataset = CatheterDataset(df_train, transform=get_transforms("train"))
    val_dataset = CatheterDataset(df_val, transform=get_transforms("valid"))

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # --- Model Initialization ---
    print(f"Initializing model: {Config.model_name}")
    device = torch.device(Config.device)
    model = CatheterModel(
        model_name=Config.model_name,
        pretrained=Config.pretrained,
        num_classes=Config.num_classes,
        drop_rate=Config.drop_rate,
        drop_path_rate=Config.drop_path_rate,
    )
    model.to(device)

    # EMA
    ema_model = None
    if Config.use_ema:
        print(f"Initializing EMA with decay {Config.ema_decay}")
        ema_model = ModelEma(model, decay=Config.ema_decay, device=device)

    # Optimizer
    optimizer = AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    # Gradient Scaler for AMP (Cite debug_lesson_1)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Scheduler
    total_steps = len(train_loader) * Config.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.lr,
        total_steps=total_steps,
        pct_start=Config.pct_start,
        div_factor=Config.div_factor,
        final_div_factor=Config.final_div_factor,
    )

    # --- Training Loop ---
    best_score = -np.inf
    best_model_path = os.path.join(Config.working_dir, "best_model.pth")

    print(f"Starting training for {Config.epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, Config.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, epoch, scaler, ema_model
        )

        # Validate (use EMA model if available)
        val_model = ema_model.module if ema_model else model
        val_loss, val_score = validate(val_model, val_loader, device)

        elapsed = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{Config.epochs} - "
            f"Time: {elapsed:.0f}s - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
            f"Val AUC: {val_score:.16f}"
        )

        # Save Best Model
        if val_score > best_score:
            print(
                f"Validation Score Improved ({best_score:.6f} ---> {val_score:.6f}). Saving model..."
            )
            best_score = val_score
            torch.save(val_model.state_dict(), best_model_path)

    total_time = time.time() - start_time
    print(
        f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s. Best AUC: {best_score:.16f}"
    )

    # --- Inference & Submission ---
    print("\nStarting inference/submission generation...")

    # Load best model weights
    if os.path.exists(best_model_path):
        state_dict = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Loaded best model weights.")
    else:
        print("Warning: Best model not found. Using current model weights.")

    submission_path = "./submission/submission.csv"
    predict_and_submit(model, device, submission_path)

    # Clear memory
    del model, ema_model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    return best_score
