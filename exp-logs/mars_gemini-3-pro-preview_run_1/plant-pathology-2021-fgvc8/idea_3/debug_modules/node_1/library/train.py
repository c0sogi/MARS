import os
import time
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from library.config import Config
from library.utils import seed_everything, get_logger, EarlyStopping
from library.dataset import AppleDataset, get_transforms
from library.model import (
    AppleDiseaseSwinModel,
    LabelSmoothingBCEWithLogitsLoss,
    train_one_epoch,
    validate,
)


def run_training(
    debug=Config.debug,
    epochs=Config.epochs,
    batch_size=Config.batch_size,
    learning_rate=Config.learning_rate,
    weight_decay=Config.weight_decay,
    min_lr=Config.min_lr,
    label_smoothing=Config.label_smoothing,
    num_workers=Config.num_workers,
):
    """
    Executes the training pipeline.
    """
    # Setup
    seed_everything(Config.seed)
    os.makedirs(Config.working_dir, exist_ok=True)
    logger = get_logger(os.path.join(Config.working_dir, "train_log.txt"))
    device = Config.device

    logger.info(f"Starting training with model: {Config.model_name}")
    logger.info(f"Epochs: {epochs}, Batch Size: {batch_size}, Debug: {debug}")

    # 1. Load Metadata
    df_train = pd.read_csv(Config.train_metadata_path)
    df_val = pd.read_csv(Config.val_metadata_path)

    if debug:
        logger.info(f"Debug mode: using {Config.debug_sample_size} samples.")
        df_train = df_train.sample(
            n=Config.debug_sample_size, random_state=Config.seed
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=Config.debug_sample_size, random_state=Config.seed
        ).reset_index(drop=True)

    # 2. Datasets & Loaders
    train_dataset = AppleDataset(
        df_train, mode="train", transform=get_transforms("train", Config.img_size)
    )
    val_dataset = AppleDataset(
        df_val, mode="val", transform=get_transforms("val", Config.img_size)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 3. Model
    model = AppleDiseaseSwinModel(pretrained=True)
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

    # 5. Loss & Scaler
    criterion = LabelSmoothingBCEWithLogitsLoss(smoothing=label_smoothing)
    scaler = GradScaler()

    # 6. Early Stopping
    early_stopping = EarlyStopping(
        patience=5, mode="max", save_path=Config.model_save_path, verbose=True
    )

    # 7. Training Loop
    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch
        )
        val_loss, val_score = validate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start_time
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
            f"Val F1: {val_score:.16f} - "
            f"Time: {elapsed:.0f}s"
        )

        early_stopping(val_score, model, optimizer, scheduler, epoch)

        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break

    logger.info("Training complete.")


def run_inference(batch_size=Config.batch_size, num_workers=Config.num_workers):
    """
    Generates predictions for the test set and saves the submission file.
    """
    device = Config.device

    # Load Metadata
    df_test = pd.read_csv(Config.test_metadata_path)

    # Dataset & Loader
    test_dataset = AppleDataset(
        df_test, mode="test", transform=get_transforms("test", Config.img_size)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Load Model
    model = AppleDiseaseSwinModel(pretrained=False)
    checkpoint_path = Config.model_save_path
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Cannot generate submission.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_probs = []

    # Inference Loop
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            # Mixed precision inference
            with autocast():
                logits = model(images)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)

    # Process predictions
    pred_labels = []
    class_labels = Config.class_labels

    for probs in all_probs:
        # Get indices where prob > threshold
        indices = np.where(probs > Config.threshold)[0]

        # Handle case where no class exceeds threshold (pick max)
        if len(indices) == 0:
            indices = [np.argmax(probs)]

        labels = [class_labels[i] for i in indices]
        pred_labels.append(" ".join(labels))

    # Create submission DataFrame
    df_sub = pd.DataFrame({"image": df_test["image"], "labels": pred_labels})

    # Save
    os.makedirs(Config.submission_dir, exist_ok=True)
    df_sub.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
