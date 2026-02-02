import torch
import torch.nn as nn
from library.config import Config
from library.utils import compute_exact_match


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        device (torch.device): Device to run training on.
        epoch (int): Current epoch number (for logging).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Loss functions
    bce_loss_fn = nn.BCELoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    for batch in dataloader:
        if not batch:  # Skip empty batches
            continue

        # Move data to device
        q_input_ids = batch["q_input_ids"].to(device)
        c_input_ids = batch["c_input_ids"].to(device)

        rank_labels = batch["rank_labels"].to(device)
        span_start_labels = batch["span_start_labels"].to(device)
        span_end_labels = batch["span_end_labels"].to(device)
        yn_labels = batch["yn_labels"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(q_input_ids, c_input_ids)

        rank_score = outputs["rank_score"]
        span_start_logits = outputs["span_start_logits"]
        span_end_logits = outputs["span_end_logits"]
        yn_logits = outputs["yn_logits"]

        # Calculate losses
        # 1. Ranking Loss
        loss_rank = bce_loss_fn(rank_score, rank_labels)

        # 2. Span Loss
        # We predict indices within the candidate sequence
        loss_span_start = ce_loss_fn(span_start_logits, span_start_labels)
        loss_span_end = ce_loss_fn(span_end_logits, span_end_labels)
        loss_span = (loss_span_start + loss_span_end) / 2.0

        # 3. Yes/No Loss
        loss_class = ce_loss_fn(yn_logits, yn_labels)

        # Weighted Sum
        loss = (
            Config.WEIGHT_RANKING_LOSS * loss_rank
            + Config.WEIGHT_SPAN_LOSS * loss_span
            + Config.WEIGHT_CLASS_LOSS * loss_class
        )

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f"Epoch {epoch} Training Loss: {avg_loss:.6f}")
    return avg_loss


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): Validation data loader.
        device (torch.device): Device to run evaluation on.

    Returns:
        tuple: (average_val_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # Metrics accumulators
    rank_correct = 0
    rank_total = 0

    span_exact_match = 0
    span_total = 0

    yn_correct = 0
    yn_total = 0

    # Loss functions
    bce_loss_fn = nn.BCELoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            if not batch:
                continue

            # Move data to device
            q_input_ids = batch["q_input_ids"].to(device)
            c_input_ids = batch["c_input_ids"].to(device)

            rank_labels = batch["rank_labels"].to(device)
            span_start_labels = batch["span_start_labels"].to(device)
            span_end_labels = batch["span_end_labels"].to(device)
            yn_labels = batch["yn_labels"].to(device)

            # Forward pass
            outputs = model(q_input_ids, c_input_ids)

            rank_score = outputs["rank_score"]
            span_start_logits = outputs["span_start_logits"]
            span_end_logits = outputs["span_end_logits"]
            yn_logits = outputs["yn_logits"]

            # Calculate losses
            loss_rank = bce_loss_fn(rank_score, rank_labels)

            loss_span_start = ce_loss_fn(span_start_logits, span_start_labels)
            loss_span_end = ce_loss_fn(span_end_logits, span_end_labels)
            loss_span = (loss_span_start + loss_span_end) / 2.0

            loss_class = ce_loss_fn(yn_logits, yn_labels)

            loss = (
                Config.WEIGHT_RANKING_LOSS * loss_rank
                + Config.WEIGHT_SPAN_LOSS * loss_span
                + Config.WEIGHT_CLASS_LOSS * loss_class
            )

            total_loss += loss.item()
            num_batches += 1

            # --- Calculate Metrics ---

            # 1. Ranking Accuracy (Threshold 0.5)
            preds_rank = (rank_score >= 0.5).float()
            rank_correct += (preds_rank == rank_labels).sum().item()
            rank_total += rank_labels.size(0)

            # 2. Span Exact Match
            # Only consider positive samples for span evaluation to avoid skewing with negatives
            # where the answer is (0,0)
            positive_mask = rank_labels == 1.0
            if positive_mask.sum() > 0:
                pred_start = torch.argmax(span_start_logits, dim=1)
                pred_end = torch.argmax(span_end_logits, dim=1)

                # Filter for positives
                p_start = pred_start[positive_mask]
                p_end = pred_end[positive_mask]
                t_start = span_start_labels[positive_mask]
                t_end = span_end_labels[positive_mask]

                # Exact match: start and end must both match
                match = (p_start == t_start) & (p_end == t_end)
                span_exact_match += match.sum().item()
                span_total += positive_mask.sum().item()

            # 3. Yes/No Accuracy
            # Only consider positive samples (long answer exists)
            if positive_mask.sum() > 0:
                pred_yn = torch.argmax(yn_logits, dim=1)

                p_yn = pred_yn[positive_mask]
                t_yn = yn_labels[positive_mask]

                yn_correct += (p_yn == t_yn).sum().item()
                yn_total += positive_mask.sum().item()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    metrics = {
        "rank_acc": rank_correct / rank_total if rank_total > 0 else 0.0,
        "span_em": span_exact_match / span_total if span_total > 0 else 0.0,
        "yn_acc": yn_correct / yn_total if yn_total > 0 else 0.0,
    }

    print(f"Validation Loss: {avg_loss:.10f}")
    print(f"Ranking Accuracy: {metrics['rank_acc']:.10f}")
    print(f"Span Exact Match (Positives): {metrics['span_em']:.10f}")
    print(f"Yes/No Accuracy (Positives): {metrics['yn_acc']:.10f}")

    return avg_loss, metrics
