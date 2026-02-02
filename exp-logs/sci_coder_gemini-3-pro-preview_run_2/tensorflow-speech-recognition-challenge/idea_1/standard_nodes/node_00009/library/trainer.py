import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from library.config import Config
from library.dataset import get_dataloaders, get_test_loader
from library.model import SimpleConvNet


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch using standard CrossEntropy.
    Cite solution_lesson_node_00007: Mixing augmentations require extended training budgets.
    Reverting to standard training for faster convergence.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train(debug=Config.DEBUG, epochs=Config.NUM_EPOCHS):
    """
    Main training loop with early stopping and model checkpointing.
    """
    set_seed()
    device = torch.device(Config.DEVICE)
    print(f"Training on device: {device}")

    # 1. Load Data
    train_loader, val_loader = get_dataloaders(debug=debug)

    # 2. Initialize Model
    model = SimpleConvNet(num_classes=Config.NUM_CLASSES)
    model = model.to(device)

    # 3. Setup Optimizer and Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    # Cite solution_lesson_node_00007: Prefer standard domain-specific augmentations and StepLR.
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

    # 4. Training Loop
    best_acc = 0.0
    patience_counter = 0

    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.MODEL_CHECKPOINT_PATH), exist_ok=True)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        # Print full precision metrics
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Train Acc: {train_acc}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Acc: {val_acc}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            print(f"Saved best model with Val Acc: {val_acc}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training finished. Best Validation Accuracy: {best_acc}")


def predict():
    """
    Generates predictions for the test set using the best saved model.
    """
    device = torch.device(Config.DEVICE)
    model = SimpleConvNet(num_classes=Config.NUM_CLASSES)

    if not os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {Config.MODEL_CHECKPOINT_PATH}")
        return

    # Load best model weights
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    test_loader = get_test_loader()

    all_preds = []

    # Inference
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())

    # Load test metadata to ensure correct filename mapping
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    fnames = df_test["fname"].tolist()

    # Map indices to labels
    predicted_labels = [Config.IDX2LABEL[idx] for idx in all_preds]

    # Create submission dataframe
    submission_df = pd.DataFrame({"fname": fnames, "label": predicted_labels})

    # Save submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
