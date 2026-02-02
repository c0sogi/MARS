import torch
import random
from torch.utils.data import Dataset
from library.data_utils import clean_and_tokenize


class ToxicityDataset(Dataset):
    """
    Dataset class for toxicity text data.
    Implements Stochastic Identity Masking during training to mitigate bias.
    """

    def __init__(
        self,
        texts,
        targets=None,
        vocab=None,
        identity_indices=None,
        mask_prob=0.0,
        is_training=False,
    ):
        """
        Args:
            texts (list): List of text strings.
            targets (list, optional): List of target values (floats).
            vocab (Vocabulary): Vocabulary object for token-to-index mapping.
            identity_indices (set, optional): Set of token indices corresponding to identity terms.
            mask_prob (float): Probability to mask an identity term during training.
            is_training (bool): Flag to enable/disable augmentation (masking).
        """
        self.texts = texts
        self.targets = targets
        self.vocab = vocab
        self.identity_indices = (
            identity_indices if identity_indices is not None else set()
        )
        self.mask_prob = mask_prob
        self.is_training = is_training

        # Pre-fetch special tokens
        # If vocab is None, we default to 0, but vocab should be provided.
        self.mask_index = vocab.get_mask_index() if vocab else 0

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = clean_and_tokenize(text)

        if self.vocab:
            indices = self.vocab.lookup_indices(tokens)
        else:
            # Fallback if no vocab provided (should not happen in proper pipeline)
            indices = []

        # Stochastic Identity Masking
        # Only apply if training, probability > 0, and we have identity terms identified
        if self.is_training and self.mask_prob > 0 and self.identity_indices:
            masked_indices = []
            for i in indices:
                # Check if the token index is an identity term
                if i in self.identity_indices and random.random() < self.mask_prob:
                    masked_indices.append(self.mask_index)
                else:
                    masked_indices.append(i)
            indices = masked_indices

        # Handle empty sequences to prevent runtime errors in EmbeddingBag
        if len(indices) == 0:
            # Use 0 (often PAD or UNK) as a placeholder
            indices = [0]

        indices_tensor = torch.tensor(indices, dtype=torch.long)

        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return indices_tensor, target
        else:
            return indices_tensor


def collate_batch(batch):
    """
    Collate function for EmbeddingBag.
    Concatenates indices into a single 1D tensor and computes offsets.

    Args:
        batch: List of items returned by ToxicityDataset.__getitem__
               Either (indices_tensor, target_tensor) or indices_tensor

    Returns:
        tuple: (text_list, offsets, label_list) if labels exist
               (text_list, offsets) if labels do not exist
    """
    label_list, text_list, offsets = [], [], [0]

    # Check if the batch contains labels (tuples) or just text (tensors)
    has_labels = isinstance(batch[0], tuple)

    if has_labels:
        for _text, _label in batch:
            label_list.append(_label)
            text_list.append(_text)
            # Record the length of the current sequence for offset calculation
            offsets.append(_text.size(0))
    else:
        for _text in batch:
            text_list.append(_text)
            offsets.append(_text.size(0))

    # The offsets are cumulative sums of lengths.
    # We need the start position of each sequence.
    # offsets currently contains [0, len1, len2, ...].
    # We take all but the last length, convert to tensor, and cumsum.
    # Example: lengths [5, 3] -> offsets input [0, 5, 3] -> cumsum -> [0, 5, 8]
    # But we only need [0, 5] for the two sequences.
    # Actually, the standard PyTorch EmbeddingBag expects offsets to be the starting indices.
    # So for lengths [L1, L2, L3], offsets are [0, L1, L1+L2].

    # Using torch.cumsum on the list of lengths (excluding the last unused accumulator if we built it that way)
    # Strategy: offsets list initialized with 0. Append lengths.
    # Then cumsum.

    # Correct logic for EmbeddingBag offsets:
    # If batch has 3 items of len 2, 4, 3.
    # text_list is concat of all.
    # offsets should be [0, 2, 6].

    # My offsets list construction above: [0, len1, len2, len3, ...]
    # If I take offsets[:-1], I get [0, len1, len2 ...].
    # Then cumsum gives [0, len1, len1+len2 ...].

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)

    if has_labels:
        label_list = torch.stack(label_list)
        return text_list, offsets, label_list
    else:
        return text_list, offsets
