import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from library import config


def train_one_epoch(model, dataloader, optimizer, scaler, device, criterion):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        # Use Automatic Mixed Precision
        with torch.amp.autocast(device_type="cuda", enabled=True):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    return epoch_loss


def evaluate(model, dataloader, device, criterion):
    """
    Evaluates the model on the validation set.
    Returns average loss and Mean F1-Score.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            with torch.amp.autocast(device_type="cuda", enabled=True):
                outputs = model(images)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            total_samples += batch_size

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            # Threshold at 0.5
            preds = (probs > 0.5).float()

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    epoch_loss = running_loss / total_samples

    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # Calculate Mean F1-Score (Macro average for multi-label)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return epoch_loss, f1


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.
    Handles the logic: if no label > 0.5, pick the max probability.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for images, _, image_ids in dataloader:
            images = images.to(device)

            with torch.amp.autocast(device_type="cuda", enabled=True):
                outputs = model(images)
                probs = torch.sigmoid(outputs)

            probs = probs.cpu().numpy()

            for i, img_id in enumerate(image_ids):
                img_probs = probs[i]

                # Get indices where probability > 0.5
                predicted_indices = np.where(img_probs > 0.5)[0]

                # If no labels exceed threshold, pick the class with highest probability
                if len(predicted_indices) == 0:
                    predicted_indices = [np.argmax(img_probs)]

                # Map indices to class names
                predicted_labels = [config.CLASSES[idx] for idx in predicted_indices]

                # Join with space
                label_str = " ".join(predicted_labels)

                results.append({"image": img_id, "labels": label_str})

    return results


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    patience=3,
):
    """
    Runs the full training loop with Early Stopping.
    """
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()

    best_f1 = -1.0
    best_model_state = None
    patience_counter = 0

    print("Starting training...")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, criterion
        )
        val_loss, val_f1 = evaluate(model, val_loader, device, criterion)

        if scheduler:
            scheduler.step()

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val F1: {val_f1}"
        )

        # Early Stopping Logic based on F1 Score
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_f1


def generate_submission(model, test_loader, device, output_path=config.SUBMISSION_PATH):
    """
    Generates predictions and saves them to a CSV file.
    """
    print("Generating submission...")
    predictions = predict(model, test_loader, device)

    df = pd.DataFrame(predictions)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
