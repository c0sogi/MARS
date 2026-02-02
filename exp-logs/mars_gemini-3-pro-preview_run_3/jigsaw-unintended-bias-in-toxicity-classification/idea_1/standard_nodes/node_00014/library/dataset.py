import torch
import random
import numpy as np
from torch.utils.data import Dataset
from library.config import Config
from library.data_utils import clean_and_tokenize


class ToxicityDataset(Dataset):
    """
    Dataset class for toxicity text data using Transformer tokenizer.
    Implements Stochastic Identity Masking and Sample Weighting.
    """

    def __init__(
        self,
        texts,
        targets=None,
        aux_targets=None,
        weights=None,
        tokenizer=None,
        identity_indices=None,
        mask_prob=0.0,
        is_training=False,
    ):
        self.texts = texts
        self.targets = targets
        self.aux_targets = aux_targets
        self.weights = weights
        self.tokenizer = tokenizer
        self.identity_indices = (
            identity_indices if identity_indices is not None else set()
        )
        self.mask_prob = mask_prob
        self.is_training = is_training
        self.mask_token_id = tokenizer.mask_token_id if tokenizer else 103

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = clean_and_tokenize(self.texts[idx])

        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=Config.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Stochastic Identity Masking (Cite solution_lesson_node_00001)
        if self.is_training and self.mask_prob > 0 and self.identity_indices:
            masked_ids = []
            for i in input_ids:
                if i in self.identity_indices and random.random() < self.mask_prob:
                    masked_ids.append(self.mask_token_id)
                else:
                    masked_ids.append(i)
            input_ids = masked_ids

        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

        if self.targets is not None:
            # Main target
            target = self.targets[idx]

            # Aux targets (Cite solution_lesson_node_00008)
            aux = self.aux_targets[idx] if self.aux_targets is not None else []

            # Combine into one tensor: [main, aux1, aux2, ...]
            # Cite solution_lesson_node_00009 (Unified Projection)
            combined_target = [target] + list(aux)
            item["targets"] = torch.tensor(combined_target, dtype=torch.float32)

            # Sample Weights (Cite solution_lesson_node_00010)
            if self.weights is not None:
                item["weights"] = torch.tensor(self.weights[idx], dtype=torch.float32)
            else:
                item["weights"] = torch.tensor(1.0, dtype=torch.float32)

        return item


def collate_batch(batch):
    """
    Collate function for Transformer batches.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    result = {"input_ids": input_ids, "attention_mask": attention_mask}

    if "targets" in batch[0]:
        targets = torch.stack([item["targets"] for item in batch])
        result["targets"] = targets

    if "weights" in batch[0]:
        weights = torch.stack([item["weights"] for item in batch])
        result["weights"] = weights

    return result
