import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import library.config as config
import library.utils as utils


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)  # Ordinal vectors (B, K-1)

        batch_size = images.size(0)
        dataset_size += batch_size

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()

        if config.MAX_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item() * batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set and computes QWK score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            targets = batch["target"].numpy()  # Integer ground truth labels

            batch_size = images.size(0)
            dataset_size += batch_size

            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item() * batch_size

            # Decode ordinal predictions
            # Sigmoid converts logits to probabilities P(y > k)
            probs = torch.sigmoid(logits)
            # Summing probabilities gives the expected ordinal score (continuous 0 to K-1)
            scores = probs.sum(dim=1)
            # Round to nearest integer to get class label
            preds = scores.round().cpu().numpy().astype(int)

            all_preds.extend(preds)
            all_targets.extend(targets)

    epoch_loss = running_loss / dataset_size
    qwk = utils.quadratic_weighted_kappa(all_targets, all_preds)

    return epoch_loss, qwk


def train_model(
    model, train_loader, val_loader, optimizer, criterion, device, num_epochs, patience
):
    """
    Main training loop with early stopping and checkpointing.
    """
    best_score = -float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_score = evaluate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val QWK: {val_score}")

        # Checkpointing & Early Stopping
        if val_score > best_score:
            print(f"Score improved from {best_score} to {val_score}. Saving model...")
            best_score = val_score
            utils.save_checkpoint(model, optimizer, epoch, val_score)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Score did not improve. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


def predict_and_submit(model, test_loader, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    # Load best model weights
    checkpoint = utils.load_checkpoint(model)
    if checkpoint:
        print(
            f"Loaded checkpoint from epoch {checkpoint['epoch']} with score {checkpoint['score']}"
        )
    else:
        print("No checkpoint found. Using current model weights.")

    model.eval()
    all_preds = []
    all_ids = []

    print("Starting inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            ids = batch["id_code"]

            logits = model(images)
            probs = torch.sigmoid(logits)
            scores = probs.sum(dim=1)
            preds = scores.round().cpu().numpy().astype(int)

            all_preds.extend(preds)
            all_ids.extend(ids)

    # Create submission directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame({"id_code": all_ids, "diagnosis": all_preds})
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
