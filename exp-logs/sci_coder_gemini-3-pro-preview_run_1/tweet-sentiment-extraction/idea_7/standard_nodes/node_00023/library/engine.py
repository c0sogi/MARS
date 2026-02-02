import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.utils import AverageMeter, jaccard, normalize_text


def loss_fn(start_logits, end_logits, start_targets, end_targets):
    """
    Computes the KL Divergence loss for start and end logits against soft targets.
    The final loss is the average of the start and end losses.

    Args:
        start_logits: Logits for start index (Batch, Seq_Len)
        end_logits: Logits for end index (Batch, Seq_Len)
        start_targets: Gaussian smoothed start targets (Batch, Seq_Len)
        end_targets: Gaussian smoothed end targets (Batch, Seq_Len)

    Returns:
        torch.Tensor: Scalar loss value
    """
    loss_fct = nn.KLDivLoss(reduction="batchmean")
    # KLDivLoss expects log-probabilities as input
    start_loss = loss_fct(F.log_softmax(start_logits, dim=1), start_targets)
    end_loss = loss_fct(F.log_softmax(end_logits, dim=1), end_targets)
    return 0.5 * start_loss + 0.5 * end_loss


def train_fn(data_loader, model, optimizer, device, scheduler):
    """
    Executes one training epoch.

    Args:
        data_loader: PyTorch DataLoader for training data
        model: The neural network model
        optimizer: The optimizer
        device: The device to run training on
        scheduler: Learning rate scheduler

    Returns:
        float: Average loss for the epoch
    """
    model.train()
    losses = AverageMeter()

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_targets = batch["start_targets"].to(device)
        end_targets = batch["end_targets"].to(device)

        optimizer.zero_grad()
        start_logits, end_logits = model(input_ids, attention_mask)

        loss = loss_fn(start_logits, end_logits, start_targets, end_targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.update(loss.item(), input_ids.size(0))

    return losses.avg


def eval_fn(data_loader, model, device, df):
    """
    Evaluates the model on the validation set using Joint Logit Decoding.
    Computes the Jaccard score.

    Args:
        data_loader: PyTorch DataLoader for validation data
        model: The neural network model
        device: The device to run evaluation on
        df: The validation dataframe containing 'text' and 'selected_text'

    Returns:
        float: Mean Jaccard score
    """
    model.eval()

    start_preds = []
    end_preds = []

    # Inference loop: Collect logits
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            start_logits, end_logits = model(input_ids, attention_mask)

            start_preds.append(start_logits.cpu().numpy())
            end_preds.append(end_logits.cpu().numpy())

    start_preds = np.concatenate(start_preds)
    end_preds = np.concatenate(end_preds)

    predictions = []
    # Access offsets directly from the dataset (assuming sequential access matching df)
    dataset_offsets = data_loader.dataset.offsets

    # Decoding loop
    for i in range(len(df)):
        # Normalize text to align with tokenizer offsets (Normalize-First strategy)
        row = df.iloc[i]
        text = normalize_text(row["text"])
        offsets = dataset_offsets[i]

        s_logits = start_preds[i]
        e_logits = end_preds[i]

        # Joint Logit Decoding: Maximize (Start_Logit + End_Logit)
        # Create a matrix of sums: (Seq, Seq)
        sum_logits = s_logits[:, None] + e_logits[None, :]

        # Enforce start <= end constraint by masking the lower triangle
        # np.triu returns the upper triangle (including diagonal)
        mask = np.triu(np.ones_like(sum_logits))
        sum_logits = np.where(mask == 1, sum_logits, -np.inf)

        # Find the indices (start, end) that maximize the sum
        best_idx = np.argmax(sum_logits)
        best_start, best_end = np.unravel_index(best_idx, sum_logits.shape)

        # Extract substring using offsets
        if best_start < len(offsets) and best_end < len(offsets):
            start_char = offsets[best_start][0]
            end_char = offsets[best_end][1]
            pred_text = text[start_char:end_char]
        else:
            # Fallback to full text if indices are invalid
            pred_text = text

        predictions.append(pred_text)

    # Calculate Jaccard Score
    scores = [jaccard(p, t) for p, t in zip(predictions, df["selected_text"])]
    return np.mean(scores)
