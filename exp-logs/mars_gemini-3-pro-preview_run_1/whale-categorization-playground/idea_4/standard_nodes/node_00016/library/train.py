import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import (
    AverageMeter,
    save_checkpoint,
    load_checkpoint,
    map_at_5,
    seed_everything,
)
from library.dataset import get_dataloaders
from library.model import WhaleConvNeXt


def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch):
    """
    Handles the training of one epoch.
    """
    model.train()

    losses = AverageMeter()

    # Iterate over data
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass (ArcFace requires labels during training)
        # Returns logits with angular margin penalty
        logits = model(images, labels)

        # Calculate loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))

        # Log periodically (optional, keeping it minimal as requested)
        if i % 100 == 0:
            # Just a simple print to show aliveness, mostly relying on epoch summary
            pass

    return losses.avg


def validate(val_loader, model, criterion, device, classes):
    """
    Evaluates the model on the validation set using TTA (Horizontal Flip).
    Computes Loss and MAP@5.
    """
    model.eval()

    losses = AverageMeter()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # ---------------------------------------------------------
            # Test-Time Augmentation (TTA): Horizontal Flip
            # ---------------------------------------------------------
            # 1. Forward pass with original images
            # Passing labels=None returns raw scaled cosine similarities (logits)
            logits_orig = model(images, labels=None)

            # 2. Forward pass with flipped images
            images_flipped = torch.flip(
                images, dims=[3]
            )  # Flip width dimension (B, C, H, W)
            logits_flip = model(images_flipped, labels=None)

            # 3. Average logits
            logits = (logits_orig + logits_flip) / 2.0

            # ---------------------------------------------------------
            # Metrics
            # ---------------------------------------------------------
            # For validation loss, we can use the averaged logits against the targets.
            # Note: CrossEntropy expects raw logits. Since these are scaled cosines,
            # they function as logits for the Softmax inside CrossEntropy.
            loss = criterion(logits, labels)
            losses.update(loss.item(), images.size(0))

            # Get Top 5 predictions
            # We don't need Softmax for ranking
            _, top_indices = torch.topk(logits, k=5, dim=1)

            # Convert indices to class names
            # top_indices is (Batch, 5)
            top_indices = top_indices.cpu().numpy()
            labels = labels.cpu().numpy()

            batch_preds = []
            for idx_list in top_indices:
                pred_names = [classes[i] for i in idx_list]
                batch_preds.append(pred_names)

            # Convert targets to class names
            batch_targets = [classes[i] for i in labels]

            all_preds.extend(batch_preds)
            all_targets.extend(batch_targets)

    # Calculate MAP@5
    map5_score = map_at_5(all_preds, all_targets)

    return losses.avg, map5_score


def create_submission(test_loader, model, device, classes):
    """
    Generates predictions for the test set using TTA and saves to submission file.
    """
    model.eval()

    image_ids = []
    predictions = []

    print("Generating submission predictions with TTA...")

    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)

            # TTA: Horizontal Flip
            logits_orig = model(images, labels=None)
            images_flipped = torch.flip(images, dims=[3])
            logits_flip = model(images_flipped, labels=None)

            logits = (logits_orig + logits_flip) / 2.0

            # Get Top 5
            _, top_indices = torch.topk(logits, k=5, dim=1)
            top_indices = top_indices.cpu().numpy()

            for i, filename in enumerate(filenames):
                image_ids.append(filename)

                # Map indices to class names
                pred_labels = [classes[idx] for idx in top_indices[i]]
                predictions.append(" ".join(pred_labels))

    # Create DataFrame
    df_sub = pd.DataFrame({"Image": image_ids, "Id": predictions})

    # Save
    os.makedirs("./submission", exist_ok=True)
    submission_path = "./submission/submission.csv"
    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


def run_training():
    """
    Main execution loop for training and validation.
    """
    seed_everything(Config.SEED)

    # 1. Data
    print("Loading Data...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        load_cached_data=True
    )
    print(f"Classes: {len(classes)}")

    # 2. Model
    print(f"Initializing Model: {Config.BACKBONE} + ArcFace")
    model = WhaleConvNeXt()
    model = model.to(Config.DEVICE)

    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR)

    # 4. Training Loop
    best_map5 = 0.0

    print("Starting Training...")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, Config.DEVICE, epoch
        )

        # Validate
        val_loss, val_map5 = validate(
            val_loader, model, criterion, Config.DEVICE, classes
        )

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        # Print metrics (Full precision)
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Time: {elapsed:.2f}s | LR: {current_lr:.8f}"
        )
        print(f"  Train Loss: {train_loss}")
        print(f"  Val Loss:   {val_loss}")
        print(f"  Val MAP@5:  {val_map5}")

        # Checkpoint
        is_best = val_map5 > best_map5
        if is_best:
            best_map5 = val_map5
            print(f"  Found new best model! (MAP@5: {best_map5})")

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_map5": best_map5,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
        )

    print(f"Training Complete. Best MAP@5: {best_map5}")

    # 5. Inference
    print("Loading best model for inference...")
    # Re-initialize model to ensure clean state or load weights into existing
    best_checkpoint = load_checkpoint(
        model, filename="model_best.pth.tar", device=Config.DEVICE
    )
    if best_checkpoint:
        print(
            f"Loaded checkpoint from epoch {best_checkpoint['epoch']} with MAP@5 {best_checkpoint.get('best_map5', 'N/A')}"
        )
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    create_submission(test_loader, model, Config.DEVICE, classes)


if __name__ == "__main__":
    run_training()
