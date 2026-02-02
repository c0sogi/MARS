import torch
import torch.nn as nn
from library.config import Config
from library.utils import FGM


def loss_fn(
    start_logits,
    end_logits,
    relevance_logits,
    start_labels,
    end_labels,
    relevance_labels,
):
    """
    Computes the total loss as a weighted sum of Span Loss and Relevance Loss.

    Args:
        start_logits: (Batch, Seq_Len)
        end_logits: (Batch, Seq_Len)
        relevance_logits: (Batch, 1)
        start_labels: (Batch)
        end_labels: (Batch)
        relevance_labels: (Batch)
    """
    # 1. Span Loss (Cross Entropy)
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_labels)
    end_loss = loss_fct(end_logits, end_labels)
    span_loss = (start_loss + end_loss) / 2

    # 2. Relevance Loss (Binary Cross Entropy)
    # Ensure labels match logits shape (Batch, 1)
    rel_loss_fct = nn.BCEWithLogitsLoss()
    relevance_loss = rel_loss_fct(relevance_logits, relevance_labels.view(-1, 1))

    # 3. Total Loss
    total_loss = span_loss + (Config.RELEVANCE_LOSS_WEIGHT * relevance_loss)
    return total_loss


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    """
    Executes the training loop for one epoch.
    Integrates Adversarial Training (FGM) with Loss Normalization.
    """
    model.train()
    final_loss = 0

    # Initialize FGM if enabled in Config
    fgm = None
    if Config.USE_FGM:
        fgm = FGM(model)

    for batch_idx, data in enumerate(data_loader):
        # Move data to device
        for k, v in data.items():
            data[k] = v.to(device)

        optimizer.zero_grad()

        # --- 1. Clean Forward Pass ---
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        start_labels = data["start_labels"]
        end_labels = data["end_labels"]
        relevance_labels = data["relevance_labels"]

        start_logits, end_logits, relevance_logits = model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        loss = loss_fn(
            start_logits,
            end_logits,
            relevance_logits,
            start_labels,
            end_labels,
            relevance_labels,
        )

        # --- 2. Backward Pass (Clean) ---
        # If using FGM, we average the clean and adversarial losses to maintain gradient scale.
        # Loss Normalization: Scale by 0.5 if FGM is active.
        if Config.USE_FGM:
            (loss / 2.0).backward()
        else:
            loss.backward()

        # --- 3. Adversarial Pass (FGM) ---
        if Config.USE_FGM:
            # Perturb embeddings
            fgm.attack(epsilon=Config.ADV_EPSILON)

            # Forward pass with perturbed embeddings
            start_logits_adv, end_logits_adv, relevance_logits_adv = model(
                input_ids=input_ids, attention_mask=attention_mask
            )

            loss_adv = loss_fn(
                start_logits_adv,
                end_logits_adv,
                relevance_logits_adv,
                start_labels,
                end_labels,
                relevance_labels,
            )

            # Backward pass (Adversarial)
            # Scale by 0.5 to average with clean loss
            (loss_adv / 2.0).backward()

            # Restore original embeddings
            fgm.restore()

        # --- 4. Optimization ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        final_loss += loss.item()

    avg_loss = final_loss / len(data_loader)
    return avg_loss


def eval_fn(data_loader, model, device):
    """
    Executes the evaluation loop.
    Returns the average loss on the validation set.
    """
    model.eval()
    final_loss = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            for k, v in data.items():
                data[k] = v.to(device)

            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            start_labels = data["start_labels"]
            end_labels = data["end_labels"]
            relevance_labels = data["relevance_labels"]

            start_logits, end_logits, relevance_logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            )

            loss = loss_fn(
                start_logits,
                end_logits,
                relevance_logits,
                start_labels,
                end_labels,
                relevance_labels,
            )

            final_loss += loss.item()

    avg_loss = final_loss / len(data_loader)
    return avg_loss
