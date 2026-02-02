import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import compute_score


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    """
    Trains the model for one epoch using Gradient Accumulation and Gradient Clipping.
    """
    model.train()

    # CrossEntropyLoss with Label Smoothing
    # We map the continuous scores to class indices, so we use CrossEntropy
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    dataset_size = len(dataloader.dataset)
    running_loss = 0.0

    optimizer.zero_grad()

    for step, data in enumerate(dataloader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        structural_features = data["structural_features"].to(device)
        labels = data["label"].to(device)

        # Convert float labels (0.0, 0.25, 0.5, 0.75, 1.0) to indices (0, 1, 2, 3, 4)
        # We multiply by 4 and round to get the integer class index
        target_indices = (labels * 4).round().long()

        batch_size = input_ids.size(0)

        # Forward pass
        outputs = model(input_ids, attention_mask, structural_features)

        # Compute loss
        loss = criterion(outputs, target_indices)

        # Normalize loss for gradient accumulation
        loss = loss / Config.GRAD_ACCUM_STEPS

        # Backward pass
        loss.backward()

        # Update weights if accumulation steps are reached or at the end of the epoch
        if (step + 1) % Config.GRAD_ACCUM_STEPS == 0 or (step + 1) == len(dataloader):
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Track loss (multiply back by accum steps to get true average)
        running_loss += (loss.item() * Config.GRAD_ACCUM_STEPS) * batch_size

    epoch_loss = running_loss / dataset_size
    print(f"Train Loss: {epoch_loss}")
    return epoch_loss


def validate(model, dataloader, device):
    """
    Validates the model on the validation set.
    Computes Loss and Pearson Correlation Coefficient.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    dataset_size = len(dataloader.dataset)
    running_loss = 0.0

    all_preds = []
    all_labels = []

    # Vector mapping class indices to score values for expected value calculation
    score_map = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device)

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            structural_features = data["structural_features"].to(device)
            labels = data["label"].to(device)

            target_indices = (labels * 4).round().long()

            outputs = model(input_ids, attention_mask, structural_features)

            loss = criterion(outputs, target_indices)
            running_loss += loss.item() * input_ids.size(0)

            # Convert logits to probabilities
            probs = torch.softmax(outputs, dim=1)

            # Calculate Expected Value: sum(prob_i * score_i)
            # This converts the distribution back to a scalar score for Pearson correlation
            expected_scores = torch.sum(probs * score_map, dim=1)

            all_preds.extend(expected_scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size
    epoch_score = compute_score(all_labels, all_preds)

    print(f"Val Loss: {epoch_loss}")
    print(f"Val Score: {epoch_score}")

    return epoch_loss, epoch_score


def train_loop(
    model, train_loader, val_loader, optimizer, scheduler, device, epochs, save_path
):
    """
    Orchestrates the training process across epochs, including Early Stopping.
    """
    best_score = -1.0
    patience = 3  # Stop if no improvement for 3 epochs
    counter = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_one_epoch(model, optimizer, scheduler, train_loader, device, epoch)
        val_loss, val_score = validate(model, val_loader, device)

        # Check for improvement
        if val_score > best_score:
            print(f"Score Improved ({best_score} -> {val_score}). Saving model...")
            best_score = val_score
            torch.save(model.state_dict(), save_path)
            counter = 0
        else:
            counter += 1
            print(f"No improvement. EarlyStopping counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training finished. Best Score: {best_score}")
