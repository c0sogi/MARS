import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from library.config import Config
from library.utils import seed_everything
from library.model import HybridResNet


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): Validation dataloader.
        criterion (nn.Module): Loss function.
        device (str): Device to run evaluation on.

    Returns:
        tuple: (average_loss, macro_f1_score)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total_loss = running_loss / len(dataloader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return total_loss, macro_f1


def train_stage1_warmup(
    model,
    train_loader,
    val_loader,
    criterion,
    device,
    epochs=Config.STAGE1_EPOCHS,
    lr=Config.STAGE1_LR,
):
    """
    Stage 1: Linear Warmup using Adam.
    Freezes the backbone and trains only the head.
    """
    print(f"\n=== Starting Stage 1: Linear Warmup (Adam, LR={lr}) ===")

    # Freeze backbone, ensure head is trainable
    model.freeze_backbone()
    model.to(device)

    # Adam optimizer for the head
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss, val_f1 = validate(model, val_loader, criterion, device)

        print(
            f"Stage 1 Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f} | Val F1: {val_f1:.6f}"
        )

    return model


def train_stage2_finetune(
    model,
    train_loader,
    val_loader,
    criterion,
    device,
    epochs=Config.STAGE2_EPOCHS,
    lr=Config.STAGE2_LR,
    patience=Config.PATIENCE,
):
    """
    Stage 2: Fine-Tuning using AdamW and Cosine Annealing.
    Unfreezes Layer 4 and trains with a lower learning rate.
    Implements Early Stopping and Checkpointing.
    """
    print(f"\n=== Starting Stage 2: Fine-Tuning (AdamW, LR={lr}) ===")

    # Unfreeze Layer 4
    model.unfreeze_layer4()
    model.to(device)

    # AdamW optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=Config.STAGE2_WEIGHT_DECAY,
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss, val_f1 = validate(model, val_loader, criterion, device)

        print(
            f"Stage 2 Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f} | Val F1: {val_f1:.6f}"
        )

        # Checkpointing based on Macro F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            # Save immediately to disk
            torch.save(best_model_wts, Config.MODEL_CHECKPOINT_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs with no improvement."
            )
            break

    print(f"Best Val F1: {best_f1:.6f}")

    # Load best weights
    model.load_state_dict(best_model_wts)
    return model


def generate_submission(model, test_loader, device, output_path=Config.SUBMISSION_PATH):
    """
    Generates predictions for the test set and saves to CSV.
    Format: Id,Predicted
    """
    print(f"\n=== Generating Submission ===")
    model.eval()
    model.to(device)

    predictions = []
    ids = []

    # Extract IDs from the dataset dataframe directly to ensure alignment
    # Assuming the test_loader is sequential (shuffle=False)
    test_df = test_loader.dataset.df
    ids = test_df["Id"].values

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())

    # Create submission DataFrame
    # Note: Using 'Predicted' as the column name based on the task description "Submission Format"
    submission_df = pd.DataFrame({"Id": ids, "Predicted": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Number of predictions: {len(submission_df)}")
    print(submission_df.head())


def run_training_pipeline():
    """
    Main entry point to run the training pipeline.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Device: {device}")

    # 1. Load Data
    print("Loading datasets...")
    train_dataset = load_dataset(
        "train", transform=None
    )  # Transform handled inside dataset if passed, but library/dataset.py handles it if we pass None?
    # Checking library/dataset.py: load_dataset takes transform.
    # We need to get transforms from utils
    from library.utils import get_transforms, compute_class_weights

    train_transform = get_transforms("train")
    val_transform = get_transforms("val")
    test_transform = get_transforms("test")

    train_dataset = load_dataset("train", transform=train_transform)
    val_dataset = load_dataset("val", transform=val_transform)
    test_dataset = load_dataset("test", transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # 2. Compute Class Weights
    print("Computing class weights...")
    # We need the dataframe from the dataset
    class_weights = compute_class_weights(train_dataset.df)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 3. Initialize Model
    print("Initializing model...")
    model = HybridResNet()

    # 4. Stage 1: Warmup
    model = train_stage1_warmup(model, train_loader, val_loader, criterion, device)

    # 5. Stage 2: Fine-Tuning
    model = train_stage2_finetune(model, train_loader, val_loader, criterion, device)

    # 6. Generate Submission
    generate_submission(model, test_loader, device)
