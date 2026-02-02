import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, calculate_macro_f1, save_checkpoint, load_checkpoint
from library.dataset import get_dataloaders, get_label_map
from library.model import ArcFaceResNet


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass with labels to apply ArcFace margin
        logits = model(images, labels)
        loss = criterion(logits, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        count += images.size(0)

    epoch_loss = running_loss / count if count > 0 else 0.0
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Validates the model on the validation set.
    Returns average loss and Macro F1 score.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # 1. Compute Loss (using ArcFace margin to track objective convergence)
            logits_margin = model(images, labels)
            loss = criterion(logits_margin, labels)
            running_loss += loss.item() * images.size(0)
            count += images.size(0)

            # 2. Compute Predictions (Inference mode: no labels, pure cosine similarity)
            # model(images) returns s * cosine_similarity
            logits_inference = model(images)
            preds = torch.argmax(logits_inference, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / count if count > 0 else 0.0

    # Concatenate all batches
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        macro_f1 = calculate_macro_f1(all_labels, all_preds)
    else:
        macro_f1 = 0.0

    return avg_loss, macro_f1


def generate_submission(model_path, debug_sample_size=None, load_cached_data=True):
    """
    Generates the submission CSV using the trained model.
    """
    device = Config.DEVICE

    # Initialize model structure
    model = ArcFaceResNet()
    model = model.to(device)

    # Load weights
    print(f"Loading model from {model_path} for inference...")
    load_checkpoint(model_path, model, device=device)
    model.eval()

    # Get test dataloader
    dataloaders = get_dataloaders(
        load_cached_data=load_cached_data, debug_sample_size=debug_sample_size
    )
    test_loader = dataloaders["test"]

    # Load label map and create inverse mapping
    label_map = get_label_map()
    inv_label_map = {v: k for k, v in label_map.items()}

    results = []

    print("Running inference on test set...")
    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)

            # Inference: get cosine similarities
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            preds_np = preds.cpu().numpy()
            ids_np = (
                image_ids.numpy() if isinstance(image_ids, torch.Tensor) else image_ids
            )

            for img_id, pred in zip(ids_np, preds_np):
                original_label = inv_label_map[pred]
                results.append({"Id": img_id, "Predicted": original_label})

    # Create DataFrame
    submission_df = pd.DataFrame(results)

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training(
    debug_sample_size=None, num_epochs=Config.NUM_EPOCHS, load_cached_data=True
):
    """
    Main training pipeline.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    # DataLoaders
    print("Initializing DataLoaders...")
    dataloaders = get_dataloaders(
        load_cached_data=load_cached_data, debug_sample_size=debug_sample_size
    )
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # Model
    print("Initializing ArcFaceResNet...")
    model = ArcFaceResNet()
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Training Loop Variables
    best_f1 = -1.0
    patience = 5
    patience_counter = 0

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_f1 = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Macro F1: {val_f1}")

        # Checkpoint
        is_best = val_f1 > best_f1
        if is_best:
            best_f1 = val_f1
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_f1": best_f1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=True,
                best_filename=Config.MODEL_SAVE_PATH,
            )
            print("New best model saved.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        # Early Stopping
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation F1: {best_f1}")

    # Generate Submission
    generate_submission(
        Config.MODEL_SAVE_PATH,
        debug_sample_size=debug_sample_size,
        load_cached_data=load_cached_data,
    )
