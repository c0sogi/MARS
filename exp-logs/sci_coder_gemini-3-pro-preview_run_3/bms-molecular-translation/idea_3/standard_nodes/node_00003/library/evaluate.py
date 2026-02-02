import time
import torch
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast
from library.config import Config
from library.utils import AverageMeter, compute_levenshtein
from library.tokenizer import Tokenizer


def validate(model, dataloader, tokenizer, criterion, device, sample_limit=1000):
    """
    Validates the model on the validation set.
    Computes Loss on the full set and Levenshtein distance on a subset.

    Args:
        model: The PyTorch model to evaluate.
        dataloader: DataLoader for the validation set.
        tokenizer: Tokenizer instance for decoding sequences.
        criterion: Loss function (e.g., CrossEntropyLoss).
        device: Device to run evaluation on (cpu or cuda).
        sample_limit (int): Maximum number of samples to use for Levenshtein calculation
                            (generation is slow). Set to None for full validation.

    Returns:
        tuple: (average_loss, average_levenshtein_distance)
    """
    model.eval()
    losses = AverageMeter()
    predictions = []
    targets = []

    # Ensure sample_limit is respected to avoid long validation times
    if sample_limit is None:
        sample_limit = len(dataloader.dataset)

    with torch.no_grad():
        for step, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            # Prepare inputs and targets for teacher forcing loss calculation
            # Input: [SOS, A, B, ...], Target: [A, B, ..., EOS]
            text_input_ids = labels[:, :-1]
            target_ids = labels[:, 1:]

            # Mixed precision context for loss calculation
            with autocast(enabled=True):
                logits = model(images, text_input_ids)
                # Reshape for CrossEntropyLoss: (N*T, Vocab) vs (N*T)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1)
                )

            losses.update(loss.item(), batch_size)

            # Generate predictions for Levenshtein calculation
            # Only if we haven't reached the sample limit
            if len(targets) < sample_limit:
                # Generate predictions using the model's autoregressive generation
                batch_preds = model.generate(
                    images, tokenizer, max_len=Config.MAX_TEXT_LEN
                )

                # Decode ground truth labels
                # labels contains indices including special tokens; sequence_to_text handles removal
                batch_targets = [
                    tokenizer.sequence_to_text(seq, remove_special_tokens=True)
                    for seq in labels.cpu().numpy()
                ]

                predictions.extend(batch_preds)
                targets.extend(batch_targets)

    # Compute Levenshtein distance on the collected subset
    # Truncate to sample_limit if we slightly overshot due to batch size
    predictions = predictions[:sample_limit]
    targets = targets[:sample_limit]

    lev_score = compute_levenshtein(predictions, targets)

    # Print full precision metrics as requested
    print(f"Validation Loss: {losses.avg}")
    print(f"Validation Levenshtein (Subset {len(targets)}): {lev_score}")

    return losses.avg, lev_score


def predict(model, dataloader, tokenizer, device):
    """
    Generates predictions for the test set and saves them to a submission file.

    Args:
        model: The trained PyTorch model.
        dataloader: DataLoader for the test set.
        tokenizer: Tokenizer instance for decoding sequences.
        device: Device to run inference on.

    Returns:
        pd.DataFrame: The submission dataframe.
    """
    print("Starting prediction on test set...")
    model.eval()
    all_preds = []

    # Load test metadata to get image IDs
    # We use the metadata file generated in the preprocessing steps to ensure alignment
    test_df = pd.read_csv(Config.TEST_METADATA)
    test_ids = test_df["image_id"].values

    start_time = time.time()

    with torch.no_grad():
        for step, images in enumerate(dataloader):
            images = images.to(device)

            # Generate predictions
            # The model.generate method handles the autoregressive loop (Image -> SOS -> ... -> EOS)
            batch_preds = model.generate(images, tokenizer, max_len=Config.MAX_TEXT_LEN)
            all_preds.extend(batch_preds)

            if step % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Predicted batch {step}/{len(dataloader)} - {elapsed}s elapsed")

    # Ensure alignment between predictions and IDs
    if len(all_preds) != len(test_ids):
        print(
            f"Warning: Prediction count {len(all_preds)} differs from ID count {len(test_ids)}"
        )
        # Adjust length to match IDs (assuming dataloader order matches metadata order)
        min_len = min(len(all_preds), len(test_ids))
        all_preds = all_preds[:min_len]
        test_ids = test_ids[:min_len]

    # Create submission dataframe
    submission = pd.DataFrame({"image_id": test_ids, "InChI": all_preds})

    # Save submission to the configured path
    submission.to_csv(Config.PREDICTIONS_CSV, index=False)
    print(f"Submission saved to {Config.PREDICTIONS_CSV}")

    return submission
