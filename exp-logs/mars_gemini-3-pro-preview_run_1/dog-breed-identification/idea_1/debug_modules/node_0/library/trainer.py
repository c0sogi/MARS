import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time

from library.config import (
    DEVICE,
    MODEL_SAVE_PATH,
    EPOCHS_HEAD,
    LEARNING_RATE_HEAD,
    EPOCHS_FINETUNE,
    LEARNING_RATE_FINETUNE,
    PATIENCE,
    SUBMISSION_PATH,
)
from library.model import get_model, freeze_backbone, unfreeze_layer4_and_head
from library.utils import save_submission


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns the average loss (Log Loss).
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    avg_loss = running_loss / dataset_size
    return avg_loss


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    Returns a tuple of (probabilities, ids).
    """
    model.eval()
    all_probs = []
    all_ids = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)

            outputs = model(images)
            probs = softmax(outputs)

            all_probs.append(probs.cpu().numpy())
            all_ids.extend(ids)

    return np.concatenate(all_probs, axis=0), all_ids


def run_training(train_loader, val_loader, test_loader, classes):
    """
    Orchestrates the two-phase training process, evaluation, and submission.
    """
    print(f"Initializing model for {len(classes)} classes on {DEVICE}...")
    model = get_model(num_classes=len(classes), pretrained=True, device=DEVICE)

    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    patience_counter = 0

    # ==========================================
    # Phase 1: Head Adaptation
    # ==========================================
    print("\n=== Phase 1: Head Adaptation (Frozen Backbone) ===")
    freeze_backbone(model)

    # Optimizer for head only (though passing all parameters with requires_grad=True works too)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE_HEAD
    )

    for epoch in range(EPOCHS_HEAD):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch {epoch+1}/{EPOCHS_HEAD} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        # Checkpointing
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

    # ==========================================
    # Phase 2: Fine-Tuning
    # ==========================================
    print("\n=== Phase 2: Fine-Tuning (Layer 4 + Head) ===")

    # Load best weights from Phase 1 before starting Phase 2?
    # Usually better to continue from current state if we didn't overfit,
    # but let's load best to be safe against last-epoch jitters.
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

    unfreeze_layer4_and_head(model)

    # Re-initialize optimizer with lower learning rate for fine-tuning
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE_FINETUNE
    )

    # Reset patience for this phase?
    # Often good to keep tracking global best, but we reset counter to give fine-tuning a chance.
    patience_counter = 0

    for epoch in range(EPOCHS_FINETUNE):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch {epoch+1}/{EPOCHS_FINETUNE} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
            print(f"New best model saved with Val Loss: {val_loss}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    # ==========================================
    # Inference
    # ==========================================
    print("\n=== Generating Predictions ===")

    # Load absolute best model
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading best model from {MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    else:
        print("Warning: No model file found. Using current model state.")

    probs, ids = predict(model, test_loader, DEVICE)

    print(f"Predictions shape: {probs.shape}")

    # Save submission
    save_submission(probs, ids, classes, submission_path=SUBMISSION_PATH)
    print("Training and inference pipeline completed.")
