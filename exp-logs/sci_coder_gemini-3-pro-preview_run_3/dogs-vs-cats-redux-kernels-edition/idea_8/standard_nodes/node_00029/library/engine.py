import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import AverageMeter, accuracy
from library.dataset import DogCatDataset, get_transforms
from library.models import create_model


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()

    losses = AverageMeter()
    accs = AverageMeter()

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        # Output is logits because we use BCEWithLogitsLoss
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))
        acc = accuracy(outputs, labels)
        accs.update(acc, images.size(0))

    print(f"Epoch {epoch} Training Results:")
    print(f"Avg Loss: {losses.avg}")
    print(f"Avg Acc: {accs.avg}")

    return losses.avg, accs.avg


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

            # Update metrics
            losses.update(loss.item(), images.size(0))
            acc = accuracy(outputs, labels)
            accs.update(acc, images.size(0))

    print(f"Validation Results:")
    print(f"Avg Loss: {losses.avg}")
    print(f"Avg Acc: {accs.avg}")

    return losses.avg, accs.avg


def train_model(model_key, device=Config.DEVICE):
    """
    Orchestrates the training process for a specific model architecture.
    Handles data loading with model-specific resolution, optimizer, scheduler,
    and checkpoint saving.
    """
    print(f"\nStarting training for model: {model_key}")

    # 1. Setup Configuration
    if model_key not in Config.MODEL_SPECS:
        raise ValueError(f"Unknown model key: {model_key}")

    spec = Config.MODEL_SPECS[model_key]
    img_size = spec["img_size"]

    # 2. Setup Data
    train_dataset = DogCatDataset(
        split="train",
        img_size=img_size,
        transform=get_transforms(img_size, is_train=True),
    )
    val_dataset = DogCatDataset(
        split="val",
        img_size=img_size,
        transform=get_transforms(img_size, is_train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Setup Model
    model = create_model(model_key, pretrained=True)
    model = model.to(device)

    # 4. Setup Optimizer and Scheduler
    # Using AdamW as per strategy
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Using CosineAnnealingLR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)

    # 5. Setup Loss
    # BCEWithLogitsLoss without label smoothing
    criterion = nn.BCEWithLogitsLoss()

    # 6. Training Loop
    best_loss = float("inf")
    save_path = os.path.join(Config.WORKING_DIR, f"{model_key}_best.pth")

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{Config.NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()

        # Save Best Model
        if val_loss < best_loss:
            print(
                f"Validation loss improved from {best_loss} to {val_loss}. Saving model to {save_path}"
            )
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
        else:
            print(f"Validation loss did not improve from {best_loss}.")

    return best_loss


def predict_with_tta(model, dataloader, device):
    """
    Generates predictions for a single model using Test Time Augmentation (TTA).
    Averages predictions from original and horizontally flipped images.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, ids in dataloader:
            images = images.to(device)

            # Original images
            logits_orig = model(images).squeeze(1)
            probs_orig = torch.sigmoid(logits_orig)

            # Flipped images (TTA)
            images_flipped = torch.flip(images, dims=[3])  # Flip along width
            logits_flip = model(images_flipped).squeeze(1)
            probs_flip = torch.sigmoid(logits_flip)

            # Average probabilities
            probs_avg = (probs_orig + probs_flip) / 2.0

            all_preds.extend(probs_avg.cpu().numpy())
            all_ids.extend(ids.numpy())

    return all_ids, all_preds


def generate_submission(device=Config.DEVICE):
    """
    Loads all trained models, performs ensemble inference with TTA,
    and generates the final submission file.
    """
    print("\nStarting Ensemble Inference...")

    ensemble_preds = None
    ids = None

    # Iterate over all models in the ensemble
    for model_key, spec in Config.MODEL_SPECS.items():
        print(f"Processing {model_key}...")

        img_size = spec["img_size"]
        checkpoint_path = os.path.join(Config.WORKING_DIR, f"{model_key}_best.pth")

        if not os.path.exists(checkpoint_path):
            print(
                f"Warning: Checkpoint for {model_key} not found at {checkpoint_path}. Skipping."
            )
            continue

        # Setup Data
        test_dataset = DogCatDataset(
            split="test",
            img_size=img_size,
            transform=get_transforms(img_size, is_train=False),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Load Model
        model = create_model(model_key, pretrained=False)  # Weights loaded manually
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model = model.to(device)

        # Predict
        current_ids, current_preds = predict_with_tta(model, test_loader, device)

        current_preds = np.array(current_preds)

        if ensemble_preds is None:
            ensemble_preds = current_preds
            ids = current_ids
        else:
            # Ensure IDs align (they should given deterministic loader, but good to be safe conceptually)
            if ids != current_ids:
                raise ValueError("Mismatch in test IDs between models.")
            ensemble_preds += current_preds

    if ensemble_preds is None:
        raise RuntimeError("No models were successfully loaded for inference.")

    # Average predictions
    num_models = len(Config.MODEL_SPECS)
    final_preds = ensemble_preds / num_models

    # Create Submission DataFrame
    df = pd.DataFrame({"id": ids, "label": final_preds})

    # Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}")
    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")


def run_full_training_pipeline():
    """
    Helper function to run the complete training and submission pipeline.
    """
    # Train each model
    for model_key in Config.MODEL_SPECS.keys():
        train_model(model_key)

    # Generate submission
    generate_submission()
