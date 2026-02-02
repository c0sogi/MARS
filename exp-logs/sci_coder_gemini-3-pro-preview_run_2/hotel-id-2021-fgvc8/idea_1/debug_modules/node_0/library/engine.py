import os
import torch
import torch.nn as nn
import pandas as pd
from library.config import Config


def calculate_map5(logits, targets):
    """
    Calculates Mean Average Precision @ 5 for a batch.

    Args:
        logits (torch.Tensor): Predicted logits of shape (Batch, Num_Classes).
        targets (torch.Tensor): Ground truth labels of shape (Batch,).

    Returns:
        float: Sum of AP@5 scores for the batch.
    """
    k = 5
    # Get top k indices: (B, k)
    _, topk_indices = logits.topk(k, dim=1)

    # Expand targets to (B, k) to compare against all top k predictions
    targets_expanded = targets.view(-1, 1).expand_as(topk_indices)

    # Check matches: (B, k) boolean mask
    hits = topk_indices == targets_expanded

    # Ranks: 1, 2, 3, 4, 5
    ranks = (
        torch.arange(1, k + 1, device=logits.device).float().view(1, -1).expand_as(hits)
    )

    # Score = 1/rank where hit is True, else 0. Sum over k for each sample.
    # Since there is only 1 ground truth, there is at most 1 hit per sample.
    scores = torch.sum(hits.float() / ranks, dim=1)

    return scores.sum().item()


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    total_map = 0.0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_map += calculate_map5(logits, labels)
        total_samples += batch_size

    return total_loss / total_samples, total_map / total_samples


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0
    total_map = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_map += calculate_map5(logits, labels)
            total_samples += batch_size

    return total_loss / total_samples, total_map / total_samples


def train_model(
    model, train_loader, val_loader, optimizer, scheduler, device, num_epochs, patience
):
    """
    Main training loop with Early Stopping.
    """
    # Use Label Smoothing as defined in Config
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    best_val_map = -1.0
    patience_counter = 0

    # Ensure the directory for saving the model exists
    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)

    for epoch in range(num_epochs):
        train_loss, train_map = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_map = validate(model, val_loader, criterion, device)

        # Step the scheduler if provided
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch + 1}: Train Loss={train_loss}, Train MAP@5={train_map}, Val Loss={val_loss}, Val MAP@5={val_map}"
        )

        # Early Stopping Logic based on MAP@5
        if val_map > best_val_map:
            best_val_map = val_map
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print("Validation MAP@5 improved. Model saved.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{patience}.")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load the best model weights before returning
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    return model


def generate_submission(model, test_loader, classes, device):
    """
    Generates predictions for the test set and saves to submission.csv.

    Args:
        model: Trained model.
        test_loader: DataLoader for test data.
        classes: Array-like mapping from class index to original hotel_id string.
        device: Torch device.
    """
    model.eval()
    results = []

    print("Generating predictions for test set...")

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = images.to(device)

            logits = model(images)

            # Get top 5 indices for each image
            _, topk_indices = logits.topk(5, dim=1)

            # Convert to numpy for easier processing
            topk_indices = topk_indices.cpu().numpy()

            for i, img_id in enumerate(image_ids):
                indices = topk_indices[i]
                # Map integer indices back to original hotel_id strings
                hotel_ids = [str(classes[idx]) for idx in indices]

                # Format as space-delimited string
                prediction_str = " ".join(hotel_ids)
                results.append({"image": img_id, "hotel_id": prediction_str})

    # Create DataFrame and save
    df = pd.DataFrame(results)

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
