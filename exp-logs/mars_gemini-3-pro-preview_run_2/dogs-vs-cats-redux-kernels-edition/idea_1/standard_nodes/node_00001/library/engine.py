import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.utils import get_device, set_seed
from library.dataset import create_dataloaders
from library.model import FineTunedResNet18


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        # Labels need to be [Batch, 1] to match model output
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    avg_loss = running_loss / total_samples
    return avg_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    avg_loss = running_loss / total_samples
    return avg_loss


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_ids = []
    all_probs = []

    with torch.no_grad():
        for inputs, ids in dataloader:
            inputs = inputs.to(device)

            # Forward pass to get logits
            logits = model(inputs)
            # Apply Sigmoid to get probabilities [0, 1]
            probs = torch.sigmoid(logits)

            # Flatten and convert to numpy
            probs = probs.view(-1).cpu().numpy()
            ids = ids.cpu().numpy()

            all_ids.extend(ids)
            all_probs.extend(probs)

    return all_ids, all_probs


def run(
    num_epochs=5,
    batch_size=32,
    lr=1e-4,
    patience=3,
    max_samples=None,
    output_dir="./submission",
):
    """
    Main driver function to train the model and generate submission.
    """
    # Ensure reproducibility
    set_seed(42)
    device = get_device()

    # Create DataLoaders
    dataloaders = create_dataloaders(batch_size=batch_size, max_samples=max_samples)

    # Initialize Model
    model = FineTunedResNet18().to(device)

    # Optimizer and Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop with Early Stopping
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, dataloaders["train"], optimizer, criterion, device
        )
        val_loss = evaluate(model, dataloaders["val"], criterion, device)

        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Generate Predictions
    print("Generating predictions on test set...")
    ids, probs = predict(model, dataloaders["test"], device)

    # Save Submission
    os.makedirs(output_dir, exist_ok=True)
    submission_path = os.path.join(output_dir, "submission.csv")

    submission_df = pd.DataFrame({"id": ids, "label": probs})

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    return model
