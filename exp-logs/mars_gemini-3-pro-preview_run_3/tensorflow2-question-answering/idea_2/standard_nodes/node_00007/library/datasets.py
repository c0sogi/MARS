import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from library.config import Config
from library.text_processing import HTMLParser, build_vocab, Tokenizer

# Set fixed random seed
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)


class RankerDataset(Dataset):
    """
    PyTorch Dataset for the Long Answer Ranker.
    Serves pairs of (Question, Paragraph) with a binary label (Match/No-Match).
    """

    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Convert list/array to tensor
        q_seq = torch.tensor(row["q_seq"], dtype=torch.long)
        p_seq = torch.tensor(row["p_seq"], dtype=torch.long)
        label = torch.tensor(row["label"], dtype=torch.float)

        return {"question": q_seq, "paragraph": p_seq, "label": label}


class ReaderDataset(Dataset):
    """
    PyTorch Dataset for the Short Answer Reader.
    Serves (Question, Paragraph) pairs with start/end token indices for the answer span.
    """

    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        q_seq = torch.tensor(row["q_seq"], dtype=torch.long)
        p_seq = torch.tensor(row["p_seq"], dtype=torch.long)
        start_idx = torch.tensor(row["start_idx"], dtype=torch.long)
        end_idx = torch.tensor(row["end_idx"], dtype=torch.long)

        return {
            "question": q_seq,
            "paragraph": p_seq,
            "start_idx": start_idx,
            "end_idx": end_idx,
        }


def _calculate_overlap(q_tokens, p_tokens):
    """
    Heuristic for Hard Negative Mining: Count token intersection.
    """
    q_set = set(q_tokens)
    p_set = set(p_tokens)
    return len(q_set.intersection(p_set))


def _process_ranker_data_from_source(
    metadata_path, raw_data_path, tokenizer, sample_size=None
):
    """
    Internal function to process raw data into ranker training examples.
    """
    print(
        f"Processing Ranker data from {raw_data_path} using metadata {metadata_path}..."
    )

    metadata = pd.read_csv(metadata_path)

    # Filter for examples that actually have a long answer for training
    metadata = metadata[metadata["has_long_answer"] == True].copy()

    if sample_size is not None:
        metadata = metadata.head(sample_size)

    parser = HTMLParser()

    processed_rows = []

    with open(raw_data_path, "rb") as f:
        for _, row in metadata.iterrows():
            offset = row["byte_offset"]
            f.seek(offset)
            line = f.readline()
            if not line:
                continue

            try:
                entry = json.loads(line.decode("utf-8"))

                # Get Question
                q_text = entry.get("question_text", "")
                q_tokens = q_text.split()
                q_seq = tokenizer.text_to_sequence(q_text)
                q_seq_padded = tokenizer.pad_sequence(q_seq, Config.MAX_Q_LEN)

                # Get Document and Candidates
                doc_text = entry.get("document_text", "")
                candidates_data = entry.get("long_answer_candidates", [])
                candidates = parser.extract_candidates(doc_text, candidates_data)

                # Identify Ground Truth
                # In training data, annotations exist. We take the first valid long answer.
                annotations = entry.get("annotations", [])
                correct_candidate_idx = -1

                for ann in annotations:
                    la = ann.get("long_answer", {})
                    start = la.get("start_token", -1)
                    if start != -1:
                        # Find which candidate matches this start token
                        for i, cand in enumerate(candidates):
                            if cand["start_token"] == start:
                                correct_candidate_idx = i
                                break
                    if correct_candidate_idx != -1:
                        break

                if correct_candidate_idx == -1:
                    continue

                # 1. Positive Sample
                pos_cand = candidates[correct_candidate_idx]
                pos_text = pos_cand["text"]
                pos_seq = tokenizer.text_to_sequence(pos_text)
                pos_seq_padded = tokenizer.pad_sequence(pos_seq, Config.MAX_DOC_LEN)

                processed_rows.append(
                    {"q_seq": q_seq_padded, "p_seq": pos_seq_padded, "label": 1.0}
                )

                # 2. Negative Sample (Hard Negative Mining)
                best_neg_idx = -1
                max_overlap = -1

                for i, cand in enumerate(candidates):
                    if i == correct_candidate_idx:
                        continue

                    cand_tokens = cand["text"].split()
                    overlap = _calculate_overlap(q_tokens, cand_tokens)

                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_neg_idx = i

                if best_neg_idx != -1:
                    neg_cand = candidates[best_neg_idx]
                    neg_text = neg_cand["text"]
                    neg_seq = tokenizer.text_to_sequence(neg_text)
                    neg_seq_padded = tokenizer.pad_sequence(neg_seq, Config.MAX_DOC_LEN)

                    processed_rows.append(
                        {"q_seq": q_seq_padded, "p_seq": neg_seq_padded, "label": 0.0}
                    )

            except json.JSONDecodeError:
                continue

    return pd.DataFrame(processed_rows)


def prepare_ranker_data(split="train", load_cached_data=True, sample_size=None):
    """
    Prepares data for the Ranker model.

    Args:
        split (str): 'train' or 'val'.
        load_cached_data (bool): Whether to load from parquet cache.
        sample_size (int): Limit number of source examples processed.

    Returns:
        RankerDataset: The ready-to-use PyTorch dataset.
    """
    Config.setup_directories()

    if split == "train":
        metadata_path = Config.TRAIN_METADATA_PATH
        cache_path = Config.RANKER_TRAIN_CACHE
        raw_data_path = os.path.join(Config.INPUT_DIR, Config.TRAIN_FILE)
    elif split == "val":
        metadata_path = Config.VAL_METADATA_PATH
        cache_path = Config.RANKER_VAL_CACHE
        raw_data_path = os.path.join(Config.INPUT_DIR, Config.TRAIN_FILE)
    else:
        raise ValueError("split must be 'train' or 'val'")

    # Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached Ranker data from {cache_path}...")
        df = pd.read_parquet(cache_path)
        # Parquet stores lists as numpy arrays usually, but let's ensure integrity
        return RankerDataset(df)

    # Build Tokenizer (loads from its own cache if available)
    tokenizer = build_vocab(load_cached_data=True)

    # Process Data
    if sample_size is None:
        sample_size = Config.DEBUG_SAMPLE_SIZE

    df = _process_ranker_data_from_source(
        metadata_path, raw_data_path, tokenizer, sample_size
    )

    # Save Cache
    print(f"Saving Ranker data to {cache_path}...")
    df.to_parquet(cache_path, index=False)

    return RankerDataset(df)


def _process_reader_data_from_source(
    metadata_path, raw_data_path, tokenizer, sample_size=None
):
    """
    Internal function to process raw data into reader training examples.
    """
    print(
        f"Processing Reader data from {raw_data_path} using metadata {metadata_path}..."
    )

    metadata = pd.read_csv(metadata_path)

    # Filter for examples that have a short answer
    metadata = metadata[metadata["has_short_answer"] == True].copy()

    if sample_size is not None:
        metadata = metadata.head(sample_size)

    parser = HTMLParser()
    processed_rows = []

    with open(raw_data_path, "rb") as f:
        for _, row in metadata.iterrows():
            offset = row["byte_offset"]
            f.seek(offset)
            line = f.readline()
            if not line:
                continue

            try:
                entry = json.loads(line.decode("utf-8"))

                q_text = entry.get("question_text", "")
                q_seq = tokenizer.text_to_sequence(q_text)
                q_seq_padded = tokenizer.pad_sequence(q_seq, Config.MAX_Q_LEN)

                doc_text = entry.get("document_text", "")
                candidates_data = entry.get("long_answer_candidates", [])
                candidates = parser.extract_candidates(doc_text, candidates_data)

                annotations = entry.get("annotations", [])

                # We need a valid short answer span that is contained within a long answer candidate
                for ann in annotations:
                    short_answers = ann.get("short_answers", [])
                    la = ann.get("long_answer", {})
                    la_start = la.get("start_token", -1)
                    la_end = la.get("end_token", -1)

                    if not short_answers or la_start == -1:
                        continue

                    # Find the candidate corresponding to the long answer
                    target_cand = None
                    for cand in candidates:
                        if (
                            cand["start_token"] == la_start
                            and cand["end_token"] == la_end
                        ):
                            target_cand = cand
                            break

                    if target_cand is None:
                        continue

                    # Process the first valid short answer span
                    sa = short_answers[0]
                    sa_start = sa["start_token"]
                    sa_end = sa["end_token"]

                    # Calculate relative indices within the candidate paragraph
                    rel_start = sa_start - la_start
                    rel_end = sa_end - la_start

                    # Tokenize paragraph
                    p_text = target_cand["text"]
                    p_seq = tokenizer.text_to_sequence(p_text)

                    # Validate indices against MAX_DOC_LEN
                    # Note: We truncate the sequence, so if the answer is outside the truncation, we skip
                    if rel_end < Config.MAX_DOC_LEN:
                        p_seq_padded = tokenizer.pad_sequence(p_seq, Config.MAX_DOC_LEN)

                        processed_rows.append(
                            {
                                "q_seq": q_seq_padded,
                                "p_seq": p_seq_padded,
                                "start_idx": rel_start,
                                "end_idx": rel_end,  # In NQ end index is exclusive, but for classification we usually want the index of the last token.
                                # However, standard span prediction often predicts inclusive start and exclusive end or inclusive end.
                                # Let's assume inclusive start and exclusive end for now, but usually CrossEntropy expects class indices.
                                # If rel_end is exclusive, the last token is rel_end - 1.
                                # Let's store the index of the last token for classification (inclusive).
                                # So target index = rel_end - 1.
                            }
                        )
                        # Fix end index to be inclusive for classification target
                        processed_rows[-1]["end_idx"] = rel_end - 1
                        break  # Only take one valid annotation per example

            except json.JSONDecodeError:
                continue

    return pd.DataFrame(processed_rows)


def prepare_reader_data(split="train", load_cached_data=True, sample_size=None):
    """
    Prepares data for the Reader model.

    Args:
        split (str): 'train' or 'val'.
        load_cached_data (bool): Whether to load from parquet cache.
        sample_size (int): Limit number of source examples processed.

    Returns:
        ReaderDataset: The ready-to-use PyTorch dataset.
    """
    Config.setup_directories()

    if split == "train":
        metadata_path = Config.TRAIN_METADATA_PATH
        cache_path = Config.READER_TRAIN_CACHE
        raw_data_path = os.path.join(Config.INPUT_DIR, Config.TRAIN_FILE)
    elif split == "val":
        metadata_path = Config.VAL_METADATA_PATH
        cache_path = Config.READER_VAL_CACHE
        raw_data_path = os.path.join(Config.INPUT_DIR, Config.TRAIN_FILE)
    else:
        raise ValueError("split must be 'train' or 'val'")

    # Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached Reader data from {cache_path}...")
        df = pd.read_parquet(cache_path)
        return ReaderDataset(df)

    # Build Tokenizer
    tokenizer = build_vocab(load_cached_data=True)

    # Process Data
    if sample_size is None:
        sample_size = Config.DEBUG_SAMPLE_SIZE

    df = _process_reader_data_from_source(
        metadata_path, raw_data_path, tokenizer, sample_size
    )

    # Save Cache
    print(f"Saving Reader data to {cache_path}...")
    df.to_parquet(cache_path, index=False)

    return ReaderDataset(df)
