import torch
import numpy as np
from library.config import Config
from library.utils import AverageMeter, jaccard


def train_fn(data_loader, model, optimizer, device, scheduler, criterion):
    """
    Executes the training loop for one epoch.

    Args:
        data_loader: PyTorch DataLoader for the training set.
        model: The TweetModel instance.
        optimizer: The optimizer (e.g., AdamW).
        device: The torch device (CPU or GPU).
        scheduler: The learning rate scheduler.
        criterion: The loss function (HybridLoss).

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    for data in data_loader:
        # Move inputs to the computing device
        input_ids = data["ids"].to(device, dtype=torch.long)
        attention_mask = data["mask"].to(device, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
        targets_start = data["targets_start"].to(device, dtype=torch.long)
        targets_end = data["targets_end"].to(device, dtype=torch.long)

        # Zero gradients before the forward pass
        optimizer.zero_grad()

        # Forward pass
        start_logits, end_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Calculate loss
        loss = criterion(start_logits, end_logits, targets_start, targets_end)

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        # Optimizer and scheduler steps
        optimizer.step()
        scheduler.step()

        # Update loss tracking
        losses.update(loss.item(), input_ids.size(0))

    return losses.avg


def eval_fn(data_loader, model, device):
    """
    Evaluates the model on the validation set and computes the Jaccard score.

    Args:
        data_loader: PyTorch DataLoader for the validation set.
        model: The TweetModel instance.
        device: The torch device.

    Returns:
        float: The average Jaccard score.
    """
    model.eval()
    jaccards = AverageMeter()

    with torch.no_grad():
        for data in data_loader:
            # Move inputs to device
            input_ids = data["ids"].to(device, dtype=torch.long)
            attention_mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)

            # Metadata required for decoding and metric calculation
            # These remain on CPU as they are strings or used for post-processing
            orig_tweets = data["orig_tweet"]
            sentiments = data["sentiment"]
            orig_selected = data["orig_selected"]
            offsets = data["offsets"].numpy()

            # Forward pass
            start_logits, end_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            # Move logits to CPU as numpy arrays for efficient processing
            start_logits = start_logits.cpu().numpy()
            end_logits = end_logits.cpu().numpy()

            # Process each sample in the batch
            for i in range(len(input_ids)):
                tweet = orig_tweets[i]
                sentiment = sentiments[i]
                target_text = orig_selected[i]
                offset = offsets[i]

                s_logits = start_logits[i]
                e_logits = end_logits[i]

                # Apply Neutral Heuristic
                if Config.neutral_heuristic and sentiment == "neutral":
                    pred_text = tweet
                else:
                    # Span Selection Logic:
                    # Find indices (start, end) that maximize (start_logit + end_logit)
                    # subject to the constraint: start_index <= end_index

                    # Create a matrix of sums: scores[j][k] = s_logits[j] + e_logits[k]
                    scores = s_logits[:, np.newaxis] + e_logits[np.newaxis, :]

                    # Mask out the lower triangle (where start > end) by setting to -inf
                    upper_tri_mask = np.triu(np.ones_like(scores), k=0)
                    scores = np.where(upper_tri_mask == 1, scores, -np.inf)

                    # Find the indices of the maximum score
                    max_idx = np.argmax(scores)
                    idx_start, idx_end = np.unravel_index(max_idx, scores.shape)

                    # Map token indices back to character indices using offsets
                    # offset[k] is a tuple (start_char, end_char)
                    char_start = offset[idx_start][0]
                    char_end = offset[idx_end][1]

                    # Extract the predicted substring
                    pred_text = tweet[char_start:char_end]

                # Calculate Jaccard score for this sample
                score = jaccard(pred_text, target_text)
                jaccards.update(score, 1)

    return jaccards.avg
