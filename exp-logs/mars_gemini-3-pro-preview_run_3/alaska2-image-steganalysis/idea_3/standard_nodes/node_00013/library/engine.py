import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.utils import seed_everything, weighted_auc_score
from library.dataset import get_dataloaders, StegoDataset
from library.model import MonoResidualEfficientNet


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    scheduler=None,
    label_smoothing=0.05,
):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        batch_size = inputs.size(0)

        # Apply Label Smoothing
        # Target is binary (0 or 1).
        # Smoothed target: y_ls = y * (1 - alpha) + 0.5 * alpha
        if label_smoothing > 0:
            targets = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
        else:
            targets = labels

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        # Outputs shape is (Batch, 1), squeeze to match labels (Batch)
        outputs = outputs.squeeze(1)

        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    # Step scheduler if it is epoch-based (CosineAnnealingLR is usually stepped per epoch)
    if scheduler:
        scheduler.step()

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and Weighted AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            batch_size = inputs.size(0)

            outputs = model(inputs)
            outputs = outputs.squeeze(1)

            # Validation loss calculated against true labels (no smoothing)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for scoring
            probs = torch.sigmoid(outputs)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Calculate Weighted AUC
    score = weighted_auc_score(all_labels, all_preds)

    return epoch_loss, score


def predict_tta(model, dataloader, device):
    """
    Generates predictions using Test Time Augmentation (D4).
    Averages predictions for Original, Rot90, Rot180, Rot270.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            # inputs shape: (B, 1, H, W)

            # Create rotated versions
            # dims [2, 3] correspond to H, W
            x0 = inputs
            x1 = torch.rot90(inputs, 1, [2, 3])
            x2 = torch.rot90(inputs, 2, [2, 3])
            x3 = torch.rot90(inputs, 3, [2, 3])

            # Forward passes
            # Squeeze output to (B,)
            out0 = torch.sigmoid(model(x0).squeeze(1))
            out1 = torch.sigmoid(model(x1).squeeze(1))
            out2 = torch.sigmoid(model(x2).squeeze(1))
            out3 = torch.sigmoid(model(x3).squeeze(1))

            # Average scores
            avg_preds = (out0 + out1 + out2 + out3) / 4.0

            all_preds.extend(avg_preds.cpu().numpy())

    return all_preds


def train_model(
    device_name="cuda",
    epochs=10,
    batch_size=32,
    learning_rate=1e-3,
    patience=3,
    num_workers=4,
    seed=42,
):
    """
    Main training loop with Early Stopping.
    Saves the best model to ./working/idea_3/best_model.pth.
    """
    seed_everything(seed)

    # Setup directories
    working_dir = "./working/idea_3"
    os.makedirs(working_dir, exist_ok=True)
    best_model_path = os.path.join(working_dir, "best_model.pth")

    # Device
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)
    print(f"Using device: {device}")

    # Data Loaders
    print("Initializing DataLoaders...")
    train_loader, val_loader = get_dataloaders(
        input_dir="./input",
        metadata_dir="./metadata",
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    # Model Setup
    print("Initializing Model...")
    model = MonoResidualEfficientNet(
        model_name="efficientnet_b2", pretrained=True, num_classes=1
    )
    model.to(device)

    # Optimizer & Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_score = 0.0
    patience_counter = 0

    print("Starting Training...")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scheduler,
            label_smoothing=0.05,
        )

        # Validate
        val_loss, val_score = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Print metrics
        print(f"Epoch {epoch+1}/{epochs} | Time: {elapsed:.2f}s")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f} | Val Weighted AUC: {val_score:.10f}")

        # Early Stopping & Checkpointing
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Weighted AUC: {best_score:.10f}")
    return best_model_path


def generate_submission(model_path, batch_size=32, device_name="cuda"):
    """
    Generates submission file for the test set.
    """
    seed_everything(42)

    # Device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    # Load Test Metadata
    test_csv_path = "./metadata/test.csv"
    if not os.path.exists(test_csv_path):
        print("Test metadata not found. Skipping submission.")
        return

    test_df = pd.read_csv(test_csv_path)
    print(f"Found {len(test_df)} test images.")

    # Create Test Dataset and Loader
    # We use the StegoDataset class. It expects a 'label' column, which is -1 in test.csv.
    # No transforms are passed (TTA handles rotations).
    test_dataset = StegoDataset(df=test_df, input_dir="./input", transform=None)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Load Model
    print(f"Loading model from {model_path}...")
    model = MonoResidualEfficientNet(
        model_name="efficientnet_b2", pretrained=False, num_classes=1
    )

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        return

    model.to(device)

    # Predict
    print("Generating predictions with TTA...")
    predictions = predict_tta(model, test_loader, device)

    # Create Submission
    submission_df = pd.DataFrame({"Id": test_df["image_id"], "Label": predictions})

    # Save
    output_dir = "./submission"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "submission.csv")

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved successfully to {output_path}")
