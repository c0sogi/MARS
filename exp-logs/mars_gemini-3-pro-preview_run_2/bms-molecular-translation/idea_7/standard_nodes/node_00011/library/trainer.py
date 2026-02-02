import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import AverageMeter, compute_levenshtein
from library.tokenizer import InChITokenizer


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch using teacher forcing.

    Args:
        model: The PyTorch model (ViT2InChI).
        train_loader: DataLoader for training data.
        optimizer: Optimizer instance.
        criterion: Loss function (CrossEntropyLoss).
        device: Device to run training on.
        epoch: Current epoch number.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    for step, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        # Teacher Forcing Inputs and Targets
        # Input to decoder: [SOS, t1, t2, ..., tn] (exclude EOS)
        # Ground Truth for loss: [t1, t2, ..., tn, EOS] (exclude SOS)
        input_ids = targets[:, :-1]
        gt_ids = targets[:, 1:]

        # Forward pass
        # logits shape: (Batch, Seq_Len, Vocab)
        logits = model(images, input_ids)

        # Reshape for CrossEntropyLoss: (N*L, Vocab) vs (N*L)
        loss = criterion(logits.reshape(-1, logits.size(-1)), gt_ids.reshape(-1))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        # Record loss
        losses.update(loss.item(), images.size(0))

    print(f"Epoch {epoch} Training Loss: {losses.avg}")
    return losses.avg


def validate(model, val_loader, criterion, device, tokenizer):
    """
    Evaluates the model on the validation set.
    Computes validation loss and Levenshtein distance.

    Args:
        model: The PyTorch model.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run evaluation on.
        tokenizer: InChITokenizer for decoding sequences.

    Returns:
        tuple: (average_loss, average_levenshtein_distance)
    """
    model.eval()
    losses = AverageMeter()
    scores = AverageMeter()

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            # 1. Compute Validation Loss (Teacher Forcing)
            input_ids = targets[:, :-1]
            gt_ids = targets[:, 1:]

            logits = model(images, input_ids)
            loss = criterion(logits.reshape(-1, logits.size(-1)), gt_ids.reshape(-1))
            losses.update(loss.item(), images.size(0))

            # 2. Compute Levenshtein Distance (Greedy Decoding Inference)
            # Generate sequences
            pred_indices = model.predict(
                images, max_len=Config.MAX_TEXT_LEN, device=device
            )

            # Decode sequences to strings
            pred_texts = tokenizer.batch_decode(pred_indices)
            target_texts = tokenizer.batch_decode(targets)

            # Calculate metric
            score = compute_levenshtein(pred_texts, target_texts)
            scores.update(score, images.size(0))

    print(f"Validation Loss: {losses.avg}")
    print(f"Validation Levenshtein Distance: {scores.avg}")

    return losses.avg, scores.avg


def predict_and_submit(
    model, test_loader, device, tokenizer, save_path=Config.SUBMISSION_PATH
):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model: The trained PyTorch model.
        test_loader: DataLoader for test data.
        device: Device to run inference on.
        tokenizer: InChITokenizer for decoding.
        save_path: Path to save the submission CSV.
    """
    print("Generating predictions for test set...")
    model.eval()

    image_ids = []
    inchi_preds = []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)

            # Inference
            pred_indices = model.predict(
                images, max_len=Config.MAX_TEXT_LEN, device=device
            )

            # Decode
            pred_texts = tokenizer.batch_decode(pred_indices)

            image_ids.extend(ids)
            inchi_preds.extend(pred_texts)

    # Create submission DataFrame
    submission_df = pd.DataFrame({"image_id": image_ids, "InChI": inchi_preds})

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
    print(f"Submission shape: {submission_df.shape}")
