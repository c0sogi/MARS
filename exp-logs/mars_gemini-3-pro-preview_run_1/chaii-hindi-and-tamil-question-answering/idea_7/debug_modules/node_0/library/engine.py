import torch
import torch.nn as nn
import numpy as np
import sys
from library.config import Config
from library.model import FGM


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch, config):
    """
    Trains the model for one epoch using Multi-Task Learning and Adversarial Training.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        dataloader: Training dataloader.
        device: 'cuda' or 'cpu'.
        epoch: Current epoch number (0-indexed).
        config: Configuration object.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()

    # Loss functions
    # Span Loss: Cross Entropy for start and end positions
    criterion_span = nn.CrossEntropyLoss()
    # Relevance Loss: Binary Cross Entropy for answer presence (1.0 or 0.0)
    criterion_rel = nn.BCEWithLogitsLoss()

    # Initialize FGM for adversarial training if enabled
    fgm = FGM(model) if config.USE_FGM else None

    total_loss = 0.0
    dataset_size = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)
        target_mapping = batch["target_mapping"].to(device)

        # --- 1. Standard Forward Pass ---
        outputs = model(input_ids, attention_mask=attention_mask)

        start_logits = outputs["start_logits"]
        end_logits = outputs["end_logits"]
        relevance_logits = outputs["relevance_logits"]

        # Calculate Losses
        loss_start = criterion_span(start_logits, start_positions)
        loss_end = criterion_span(end_logits, end_positions)
        loss_rel = criterion_rel(relevance_logits, target_mapping)

        # Total Loss = Average of Span Losses + Relevance Loss
        loss = (loss_start + loss_end) / 2 + loss_rel

        # Backward Pass
        loss.backward()

        # --- 2. Adversarial Training (FGM) ---
        if config.USE_FGM:
            # Perturb embeddings
            fgm.attack(epsilon=config.FGM_EPSILON)

            # Adversarial Forward Pass
            outputs_adv = model(input_ids, attention_mask=attention_mask)

            loss_start_adv = criterion_span(
                outputs_adv["start_logits"], start_positions
            )
            loss_end_adv = criterion_span(outputs_adv["end_logits"], end_positions)
            loss_rel_adv = criterion_rel(
                outputs_adv["relevance_logits"], target_mapping
            )

            loss_adv = (loss_start_adv + loss_end_adv) / 2 + loss_rel_adv

            # Adversarial Backward Pass
            loss_adv.backward()

            # Restore original embeddings
            fgm.restore()

        # --- 3. Optimization Step ---
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / dataset_size
    print(f"Epoch {epoch + 1} | Average Train Loss: {avg_loss}")

    return avg_loss


def predict_test(model, dataloader, device):
    """
    Runs inference on the test set.

    Args:
        model: The trained PyTorch model.
        dataloader: Test dataloader.
        device: 'cuda' or 'cpu'.

    Returns:
        dict: A dictionary containing numpy arrays of logits and metadata.
    """
    model.eval()

    # Containers for accumulating results
    all_start_logits = []
    all_end_logits = []
    all_relevance_logits = []

    all_example_ids = []
    all_offset_mappings = []
    all_sequence_ids = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)

            # Move logits to CPU and convert to numpy
            start_logits = outputs["start_logits"].detach().cpu().numpy()
            end_logits = outputs["end_logits"].detach().cpu().numpy()
            relevance_logits = outputs["relevance_logits"].detach().cpu().numpy()

            all_start_logits.append(start_logits)
            all_end_logits.append(end_logits)
            all_relevance_logits.append(relevance_logits)

            # Collect metadata
            # example_id is a list of strings
            all_example_ids.extend(batch["example_id"])

            # offset_mapping and sequence_ids are tensors, convert to numpy
            all_offset_mappings.append(batch["offset_mapping"].numpy())
            all_sequence_ids.append(batch["sequence_ids"].numpy())

    # Concatenate all batches
    predictions = {
        "start_logits": np.concatenate(all_start_logits, axis=0),
        "end_logits": np.concatenate(all_end_logits, axis=0),
        "relevance_logits": np.concatenate(all_relevance_logits, axis=0),
        "example_ids": all_example_ids,
        "offset_mappings": np.concatenate(all_offset_mappings, axis=0),
        "sequence_ids": np.concatenate(all_sequence_ids, axis=0),
    }

    return predictions
