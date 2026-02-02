import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.model import WhaleResNet
from library.dataset import get_dataloaders


def train_model(num_epochs=Config.NUM_EPOCHS, patience=Config.PATIENCE):
    """
    Trains the WhaleResNet model with Early Stopping and saves the best model.

    Args:
        num_epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs to wait for improvement before stopping.

    Returns:
        model (nn.Module): The model loaded with the best weights.
    """
    # Ensure reproducibility
    Config.set_seed()

    # Create working directory for model checkpoints
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders()

    # Initialize Model, Loss, and Optimizer
    device = torch.device(Config.DEVICE)
    model = WhaleResNet(pretrained=Config.PRETRAINED)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Training State Tracking
    best_val_auc = -1.0
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(1, num_epochs + 1):
        # ==========================
        # Training Phase
        # ==========================
        model.train()
        running_loss = 0.0
        train_steps = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_steps += 1

        avg_train_loss = running_loss / train_steps if train_steps > 0 else 0.0

        # ==========================
        # Validation Phase
        # ==========================
        model.eval()
        val_running_loss = 0.0
        val_steps = 0
        all_targets = []
        all_logits = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                val_steps += 1

                all_logits.append(outputs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        avg_val_loss = val_running_loss / val_steps if val_steps > 0 else 0.0

        # Concatenate and calculate metrics
        all_logits = np.concatenate(all_logits)
        all_targets = np.concatenate(all_targets)

        # Apply sigmoid to get probabilities for AUC
        all_probs = 1.0 / (1.0 + np.exp(-all_logits))

        try:
            val_auc = roc_auc_score(all_targets, all_probs)
        except ValueError:
            # Handle edge case if only one class is present in batch (unlikely with stratified split)
            val_auc = 0.5

        # Print metrics with full precision
        print(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {avg_train_loss} - "
            f"Val Loss: {avg_val_loss} - "
            f"Val AUC: {val_auc}"
        )

        # ==========================
        # Checkpointing & Early Stopping
        # ==========================
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"New best model saved with AUC: {best_val_auc}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    # Load best weights
    if os.path.exists(best_model_path):
        print(f"Loading best model weights from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: No best model file found. Returning current model.")

    return model


def generate_submission(model):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model (nn.Module): Trained model.
    """
    print("Generating submission...")

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Get Test Loader
    _, _, test_loader = get_dataloaders()

    device = torch.device(Config.DEVICE)
    model = model.to(device)
    model.eval()

    clips = []
    probabilities = []

    with torch.no_grad():
        for images, clip_names in test_loader:
            images = images.to(device)

            # Forward pass
            logits = model(images)

            # Apply sigmoid to get probabilities [0, 1]
            probs = torch.sigmoid(logits).cpu().numpy()

            # Flatten probs (batch_size, 1) -> (batch_size,)
            probs = probs.flatten()

            clips.extend(clip_names)
            probabilities.extend(probs)

    # Create DataFrame
    submission_df = pd.DataFrame({"clip": clips, "probability": probabilities})

    # Save to CSV
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")
