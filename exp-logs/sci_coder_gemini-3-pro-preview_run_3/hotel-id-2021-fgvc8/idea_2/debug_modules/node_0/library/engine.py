import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import AverageMeter, mapk
from library.model import HotelEfficientNet


def train_fn(dataloader, model, criterion, optimizer, device, scheduler=None):
    """
    Performs one epoch of training.
    """
    model.train()
    loss_meter = AverageMeter()

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass:
        # Pass labels to the model to calculate ArcFace logits (with margin)
        outputs = model(images, labels)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    # Step the scheduler if it's epoch-based
    if scheduler:
        scheduler.step()

    return loss_meter.avg


def eval_fn(dataloader, model, device):
    """
    Performs evaluation on the validation set and computes MAP@5.
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            # Forward pass:
            # Pass labels=None to get scaled cosine similarities (no margin)
            outputs = model(images, labels=None)

            # Get top 5 predictions (indices)
            # outputs shape: (B, Num_Classes)
            _, indices = torch.topk(outputs, k=5, dim=1)

            # Collect predictions and targets for MAP calculation
            all_preds.extend(indices.cpu().numpy().tolist())

            # Targets need to be a list of lists for mapk function: [[label], [label], ...]
            all_targets.extend(labels.view(-1, 1).numpy().tolist())

    # Calculate MAP@5
    score = mapk(all_targets, all_preds, k=5)

    return score


def train_loop(train_loader, val_loader, num_classes):
    """
    Orchestrates the training process:
    - Initializes model, optimizer, scheduler.
    - Runs training and validation loops.
    - Implements Early Stopping.
    - Saves the best model.
    """
    device = Config.DEVICE

    # Initialize Model
    model = HotelEfficientNet(num_classes=num_classes)
    model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # Tracking
    best_score = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_fn(
            train_loader, model, criterion, optimizer, device, scheduler
        )

        # Validate
        val_score = eval_fn(val_loader, model, device)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - "
            f"LR: {current_lr:.6f} - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val MAP@5: {val_score:.10f}"
        )

        # Check for improvement
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"New best score! Model saved to {best_model_path}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    return best_score


def predict_and_submit(test_loader, num_classes, unique_ids):
    """
    Generates predictions for the test set using the best saved model.
    Saves the result to submission.csv.
    """
    device = Config.DEVICE
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        print(
            f"Error: Best model not found at {best_model_path}. Cannot generate submission."
        )
        return

    # Load Model
    print("Loading best model for inference...")
    model = HotelEfficientNet(num_classes=num_classes)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    image_ids = []
    predictions = []

    print("Generating predictions...")

    with torch.no_grad():
        for images, paths in test_loader:
            images = images.to(device)

            # Inference: Get cosine similarities
            if Config.USE_TTA:
                # Simple TTA: Average of original and horizontal flip
                out1 = model(images, labels=None)
                out2 = model(torch.flip(images, dims=[3]), labels=None)
                outputs = (out1 + out2) / 2.0
            else:
                outputs = model(images, labels=None)

            # Get top K indices
            _, indices = torch.topk(outputs, k=Config.TOP_K, dim=1)
            indices = indices.cpu().numpy()

            # Map indices back to Hotel IDs
            # unique_ids is a numpy array where index matches the class index
            batch_preds = []
            for row in indices:
                pred_ids = unique_ids[row]  # Retrieve hotel_ids
                pred_str = " ".join(map(str, pred_ids))
                batch_preds.append(pred_str)

            # Extract filenames from paths
            # paths are like "test_images/filename.jpg", we want "filename.jpg"
            batch_image_ids = [os.path.basename(p) for p in paths]

            image_ids.extend(batch_image_ids)
            predictions.extend(batch_preds)

    # Create Submission DataFrame
    submission_df = pd.DataFrame({"image": image_ids, "hotel_id": predictions})

    # Save to file
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
