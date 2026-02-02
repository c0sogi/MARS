import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from library.config import Config
from library.data_utils import build_vocab, prepare_ranker_data, prepare_reader_data


class NQRankerDataset(Dataset):
    """
    Dataset for the Siamese Ranker model.
    Provides triplets: (Question, Positive Candidate, [Negative Candidates]).
    """

    def __init__(
        self, config: Config, split: str = "train", load_cached_data: bool = True
    ):
        self.config = config
        self.split = split

        # Build/Load Tokenizer
        self.tokenizer = build_vocab(config, load_cached_data=load_cached_data)
        self.pad_token_id = self.tokenizer.pad_token_id

        # Load Data
        self.data = prepare_ranker_data(
            config, self.tokenizer, split=split, load_cached_data=load_cached_data
        )

        # Identify negative columns dynamically
        self.neg_cols = [c for c in self.data.columns if c.startswith("neg_ids_")]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        q_ids = torch.tensor(row["q_ids"], dtype=torch.long)
        pos_ids = torch.tensor(row["pos_ids"], dtype=torch.long)

        neg_ids_list = []
        for col in self.neg_cols:
            neg_ids = row[col]
            # Ensure it's a list before converting
            if isinstance(neg_ids, np.ndarray):
                neg_ids = neg_ids.tolist()
            neg_ids_list.append(torch.tensor(neg_ids, dtype=torch.long))

        return q_ids, pos_ids, neg_ids_list


def ranker_collate_fn(batch):
    """
    Collates batch for Ranker.
    Pads sequences to max length in batch.
    Returns:
        q_batch: (B, Max_Q_Len)
        pos_batch: (B, Max_Doc_Len)
        neg_batch: (B, Num_Neg, Max_Doc_Len)
    """
    # Unpack batch
    q_list, pos_list, neg_list_of_lists = zip(*batch)

    # Pad token ID is 1 based on Tokenizer implementation in data_utils
    pad_id = 1

    # Pad Questions
    q_batch = pad_sequence(q_list, batch_first=True, padding_value=pad_id)

    # Pad Positive Candidates
    pos_batch = pad_sequence(pos_list, batch_first=True, padding_value=pad_id)

    # Process Negatives
    # Flatten list of lists to pad them all together, then reshape
    flat_negs = [item for sublist in neg_list_of_lists for item in sublist]
    padded_negs = pad_sequence(flat_negs, batch_first=True, padding_value=pad_id)

    # Reshape back to (Batch, Num_Neg, Max_Len)
    # Assuming constant num_negatives per sample as per config
    num_negs = len(neg_list_of_lists[0])
    batch_size = len(batch)
    max_len = padded_negs.shape[1]

    neg_batch = padded_negs.view(batch_size, num_negs, max_len)

    return q_batch, pos_batch, neg_batch


class NQReaderDataset(Dataset):
    """
    Dataset for the Separable ConvNet Reader model.
    Provides: (Input IDs [Q+Context], Start Index, End Index).
    """

    def __init__(
        self, config: Config, split: str = "train", load_cached_data: bool = True
    ):
        self.config = config
        self.split = split

        # Build/Load Tokenizer
        self.tokenizer = build_vocab(config, load_cached_data=load_cached_data)
        self.pad_token_id = self.tokenizer.pad_token_id

        # Load Data
        self.data = prepare_reader_data(
            config, self.tokenizer, split=split, load_cached_data=load_cached_data
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        input_ids = torch.tensor(row["input_ids"], dtype=torch.long)
        start_idx = torch.tensor(row["start_token_idx"], dtype=torch.long)
        end_idx = torch.tensor(row["end_token_idx"], dtype=torch.long)

        return input_ids, start_idx, end_idx


def reader_collate_fn(batch):
    """
    Collates batch for Reader.
    Pads input sequences. Stacks targets.
    Returns:
        input_batch: (B, Max_Seq_Len)
        start_batch: (B,)
        end_batch: (B,)
    """
    input_list, start_list, end_list = zip(*batch)

    pad_id = 1

    input_batch = pad_sequence(input_list, batch_first=True, padding_value=pad_id)
    start_batch = torch.stack(start_list)
    end_batch = torch.stack(end_list)

    return input_batch, start_batch, end_batch
