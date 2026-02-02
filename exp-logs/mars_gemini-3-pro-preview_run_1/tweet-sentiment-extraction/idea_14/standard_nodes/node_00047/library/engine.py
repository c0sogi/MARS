import torch
import numpy as np
import pandas as pd
from library.utils import jaccard
from library.config import Config


class AverageMeter:
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_fn(
    dataloader,
    model,
    optimizer,
    device,
    scheduler,
    criterion_task,
    criterion_rdrop,
    config,
):
    """
    Performs one epoch of training with R-Drop consistency regularization.

    Args:
        dataloader: PyTorch DataLoader for training data.
        model: The neural network model.
        optimizer: The optimizer.
        device: Calculation device (CPU/GPU).
        scheduler: Learning rate scheduler.
        criterion_task: Loss function for the main task (SoftTargetKLLoss).
        criterion_rdrop: Loss function for consistency regularization (RDropLoss).
        config: Configuration object containing hyperparameters.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_targets = batch["start_targets"].to(device)
        end_targets = batch["end_targets"].to(device)

        batch_size = input_ids.size(0)

        # --- R-Drop: Two Forward Passes ---
        # Pass 1
        start_logits_1, end_logits_1 = model(input_ids, attention_mask)
        # Pass 2
        start_logits_2, end_logits_2 = model(input_ids, attention_mask)

        # --- Task Loss Calculation ---
        # Calculate loss for both passes and average them
        loss_start_1 = criterion_task(start_logits_1, start_targets)
        loss_end_1 = criterion_task(end_logits_1, end_targets)
        loss_1 = loss_start_1 + loss_end_1

        loss_start_2 = criterion_task(start_logits_2, start_targets)
        loss_end_2 = criterion_task(end_logits_2, end_targets)
        loss_2 = loss_start_2 + loss_end_2

        task_loss = 0.5 * (loss_1 + loss_2)

        # --- Consistency Loss Calculation (R-Drop) ---
        # Bidirectional KL Divergence between the two distributions
        loss_rdrop_start = criterion_rdrop(start_logits_1, start_logits_2)
        loss_rdrop_end = criterion_rdrop(end_logits_1, end_logits_2)
        rdrop_loss = 0.5 * (loss_rdrop_start + loss_rdrop_end)

        # --- Total Loss ---
        loss = task_loss + config.r_drop_alpha * rdrop_loss

        # --- Optimization Step ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), batch_size)

    return losses.avg


def eval_fn(dataloader, model, device, criterion_task, config):
    """
    Evaluates the model on the validation set.
    Computes Loss and Jaccard Score.

    Args:
        dataloader: PyTorch DataLoader for validation data.
        model: The neural network model.
        device: Calculation device.
        criterion_task: Loss function for the main task.
        config: Configuration object.

    Returns:
        tuple: (Average Loss, Average Jaccard Score)
    """
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()

    # Load validation metadata to get ground truth 'selected_text'
    # Filter out neutrals to align with the dataloader provided by library.data.get_loaders
    valid_df = pd.read_csv(config.VAL_META_PATH)
    valid_df = valid_df[valid_df["sentiment"] != "neutral"].reset_index(drop=True)

    current_idx = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_targets = batch["start_targets"].to(device)
            end_targets = batch["end_targets"].to(device)

            # Retrieve auxiliary data for post-processing
            offsets = batch["offsets"].numpy()
            texts = batch["text"]

            batch_size = input_ids.size(0)

            # Forward Pass
            start_logits, end_logits = model(input_ids, attention_mask)

            # Loss Calculation
            loss_start = criterion_task(start_logits, start_targets)
            loss_end = criterion_task(end_logits, end_targets)
            loss = loss_start + loss_end
            losses.update(loss.item(), batch_size)

            # --- Decoding and Metric Calculation ---
            start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

            for i in range(batch_size):
                start_p = start_probs[i]
                end_p = end_probs[i]
                offset = offsets[i]
                original_text = texts[i]

                # Retrieve ground truth selected_text
                if current_idx < len(valid_df):
                    target_text = valid_df.iloc[current_idx]["selected_text"]
                else:
                    # Fallback if index out of bounds (should not happen if aligned)
                    target_text = ""

                # Span Selection Logic: Maximize joint probability P(start) * P(end)
                # subject to start <= end
                score_mat = np.outer(start_p, end_p)
                score_mat = np.triu(score_mat)  # Mask invalid spans (end < start)

                best_idx = np.argmax(score_mat)
                best_start_idx, best_end_idx = np.unravel_index(
                    best_idx, score_mat.shape
                )

                # Extract predicted substring
                if best_start_idx >= len(offset) or best_end_idx >= len(offset):
                    pred_text = original_text
                else:
                    char_start = offset[best_start_idx][0]
                    char_end = offset[best_end_idx][1]

                    # Handle special tokens (0,0) or invalid spans
                    if char_start == 0 and char_end == 0:
                        pred_text = original_text
                    else:
                        pred_text = original_text[char_start:char_end]

                # Compute Jaccard Score
                score = jaccard(target_text, pred_text)
                jaccards.update(score, 1)

                current_idx += 1

    return losses.avg, jaccards.avg
