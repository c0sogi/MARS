import os
import torch
import numpy as np
import random
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_metrics(loc_logits, word_logits, loc_labels, word_labels):
    """
    Computes accuracy metrics for location and word predictions.

    Args:
        loc_logits (torch.Tensor): (Batch, Seq_Len) - Unnormalized location scores.
        word_logits (torch.Tensor): (Batch, Seq_Len, Vocab_Size) - Unnormalized word scores.
        loc_labels (torch.Tensor): (Batch, Seq_Len) - One-hot or binary location labels.
        word_labels (torch.Tensor): (Batch,) - Target word IDs.

    Returns:
        dict: Dictionary containing 'loc_acc' and 'word_acc'.
    """
    # Move tensors to CPU for metric calculation
    loc_logits = loc_logits.detach().cpu()
    word_logits = word_logits.detach().cpu()
    loc_labels = loc_labels.detach().cpu()
    word_labels = word_labels.detach().cpu()

    # ----------------------------------------------------------------------
    # 1. Location Accuracy
    # ----------------------------------------------------------------------
    # Predicted location: Index with the highest score
    loc_preds = torch.argmax(loc_logits, dim=1)

    # Target location: Index of the 1.0 in the one-hot label
    loc_targets = torch.argmax(loc_labels, dim=1)

    loc_correct = (loc_preds == loc_targets).sum().item()
    loc_acc = loc_correct / loc_labels.size(0)

    # ----------------------------------------------------------------------
    # 2. Word Accuracy (at Ground Truth Location)
    # ----------------------------------------------------------------------
    # We evaluate word prediction accuracy specifically at the ground truth location.
    # This separates the capability of "knowing what word" from "knowing where".

    batch_size = word_labels.size(0)
    batch_indices = torch.arange(batch_size)

    # Extract the word logits corresponding to the true location
    # Shape: (Batch, Vocab_Size)
    target_loc_logits = word_logits[batch_indices, loc_targets, :]

    # Predicted word ID
    word_preds = torch.argmax(target_loc_logits, dim=1)

    word_correct = (word_preds == word_labels).sum().item()
    word_acc = word_correct / batch_size

    return {"loc_acc": loc_acc, "word_acc": word_acc}


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    """
    Saves the model and optimizer state to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): Current epoch number.
        val_loss (float): Validation loss at this checkpoint.
        path (str): File path to save the checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "val_loss": val_loss,
    }

    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None):
    """
    Loads the model and optimizer state from a checkpoint file.

    Args:
        path (str): File path to load the checkpoint from.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.

    Returns:
        dict: The loaded checkpoint dictionary, or None if file not found.
    """
    if not os.path.exists(path):
        return None

    checkpoint = torch.load(path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint["optimizer_state_dict"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def insert_word_in_sentence(sentence, word, loc_token_idx, tokenizer):
    """
    Inserts a word into the original sentence based on the predicted token index.

    Args:
        sentence (str): The original sentence text (with missing word).
        word (str): The predicted word to insert.
        loc_token_idx (int): The index of the token (in the tokenized sequence)
                             after which the word should be inserted.
        tokenizer: The tokenizer used for the model (to map tokens to char offsets).

    Returns:
        str: The reconstructed sentence.
    """
    # Re-tokenize to get character offsets mapping.
    # We use truncation to match the model's input logic, ensuring indices align.
    encoding = tokenizer(
        sentence,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=Config.MAX_SEQ_LEN,
    )

    offsets = encoding["offset_mapping"]

    # Safety check: Clamp index to valid range
    if loc_token_idx >= len(offsets):
        loc_token_idx = len(offsets) - 1
    if loc_token_idx < 0:
        loc_token_idx = 0

    # Get the character span of the token at the predicted location.
    # The model predicts insertion *after* this token.
    # offsets[i] returns (start_char, end_char)
    _, end_char = offsets[loc_token_idx]

    # Split the sentence at the end of the token
    # Note: offsets are relative to the original string
    prefix = sentence[:end_char]
    suffix = sentence[end_char:]

    # Construct the new sentence.
    # We add a space before the inserted word to ensure separation.
    # We assume the suffix retains its original spacing (e.g., " ." or " nextword").
    reconstructed = f"{prefix} {word}{suffix}"

    return reconstructed
