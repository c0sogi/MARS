import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from library.config import Config
from library.utils import compute_score


def get_expected_scores(logits):
    """
    Converts 5-class logits to expected scalar scores [0, 1].
    Calculates the expected value of the probability distribution.
    """
    # Convert logits to probabilities
    # logits shape: (N, 5)
    # Use as_tensor to handle both Tensor and Numpy inputs efficiently
    probs = torch.softmax(torch.as_tensor(logits), dim=1)

    # Class values corresponding to scores: 0.0, 0.25, 0.5, 0.75, 1.0
    class_values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=probs.device)

    # Compute expected value: sum(p_i * v_i)
    # shape: (N,)
    expected_scores = torch.sum(probs * class_values, dim=1)
    return expected_scores.detach().cpu().numpy()


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch, scaler):
    """
    Trains the model for one epoch using Mixed Precision.
    """
    model.train()

    # Use CrossEntropyLoss with label smoothing as defined in Config
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.label_smoothing)

    running_loss = 0.0
    dataset_size = 0

    for step, data in enumerate(dataloader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["labels"].to(device)

        batch_size = input_ids.size(0)

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with autocast(enabled=Config.use_fp16):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        # Scaled Backward Pass
        scaler.scale(loss).backward()

        # Unscale gradients for clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        # Optimizer Step
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on a given dataloader.
    Returns average loss, raw logits, and true labels (if available).
    """
    model.eval()

    criterion = nn.CrossEntropyLoss(label_smoothing=Config.label_smoothing)

    running_loss = 0.0
    dataset_size = 0

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)

            batch_size = input_ids.size(0)

            # Use autocast for inference to match training precision and save memory
            with autocast(enabled=Config.use_fp16):
                logits = model(input_ids, attention_mask)

            if "labels" in data:
                labels = data["labels"].to(device)
                loss = criterion(logits, labels)
                running_loss += loss.item() * batch_size
                all_labels.append(labels.cpu().numpy())

            all_logits.append(logits.float().cpu().numpy())
            dataset_size += batch_size

    all_logits = np.concatenate(all_logits, axis=0)

    avg_loss = 0.0
    if dataset_size > 0 and running_loss > 0:
        avg_loss = running_loss / dataset_size

    if len(all_labels) > 0:
        all_labels = np.concatenate(all_labels, axis=0)
    else:
        all_labels = None

    return avg_loss, all_logits, all_labels


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    patience=3,
    fold=0,
):
    """
    Orchestrates the training loop with validation and early stopping.
    """
    scaler = GradScaler(enabled=Config.use_fp16)

    best_score = -1.0
    early_stopping_counter = 0

    # Path to save best model for this fold
    save_path = os.path.join(Config.working_dir, f"model_fold_{fold}.bin")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, epoch, scaler
        )

        val_loss, val_logits, val_labels = evaluate(model, val_loader, device)

        # Calculate Pearson Score for validation
        # 1. Convert logits to expected scalar scores
        val_preds_score = get_expected_scores(val_logits)

        # 2. Convert label indices (0-4) back to scores (0.0-1.0)
        if val_labels is not None:
            val_true_score = val_labels * 0.25
            score = compute_score(val_true_score, val_preds_score)
        else:
            score = 0.0

        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Pearson: {score}"
        )

        # Early Stopping Logic
        if score > best_score:
            best_score = score
            early_stopping_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best Validation Pearson Score: {best_score}")

    # Load best model weights before returning
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model


def generate_submission(model, test_loader, device):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    # Get raw logits from the model
    _, test_logits, _ = evaluate(model, test_loader, device)

    # Convert logits to continuous scores
    test_scores = get_expected_scores(test_logits)

    # Load test metadata to get IDs
    test_path = os.path.join(Config.metadata_dir, "test.csv")
    test_df = pd.read_csv(test_path)

    # Handle potential length mismatch if debug mode was used (subsampling)
    if len(test_df) != len(test_scores):
        if Config.debug:
            test_df = test_df.iloc[: len(test_scores)]
        else:
            print(
                f"Warning: Test DF length {len(test_df)} != Predictions {len(test_scores)}"
            )

    # Assign scores
    test_df["score"] = test_scores
    submission_df = test_df[["id", "score"]]

    # Save to file
    out_path = os.path.join(Config.submission_dir, "submission.csv")
    submission_df.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")
