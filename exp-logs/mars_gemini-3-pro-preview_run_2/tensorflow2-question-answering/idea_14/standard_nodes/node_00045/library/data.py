import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter
from library.config import Config
from library.utils import seed_everything

# Ensure deterministic behavior
seed_everything(Config.SEED)


class Vocabulary:
    def __init__(self):
        self.stoi = {}
        self.itos = []

    def build(self, texts, max_size=Config.VOCAB_SIZE):
        print("Building vocabulary...")
        counter = Counter()
        for text in texts:
            counter.update(text.split())

        # Start with special tokens
        # <PAD>: 0, <UNK>: 1, <NULL>: 2 (used for start of candidate/no answer)
        self.itos = [Config.PAD_TOKEN, Config.UNK_TOKEN, "<NULL>"]

        # Add most common words
        most_common = counter.most_common(max_size - len(self.itos))
        self.itos.extend([word for word, count in most_common])

        self.stoi = {word: i for i, word in enumerate(self.itos)}
        print(f"Vocabulary built with {len(self.itos)} tokens.")

    def save(self, path):
        np.save(path, np.array(self.itos))
        print(f"Vocabulary saved to {path}")

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found at {path}")
        self.itos = np.load(path).tolist()
        self.stoi = {word: i for i, word in enumerate(self.itos)}
        print(f"Vocabulary loaded from {path}. Size: {len(self.itos)}")

    def encode(self, text, max_len=None):
        if isinstance(text, str):
            tokens = text.split()
        else:
            tokens = text

        indices = [
            self.stoi.get(token, self.stoi[Config.UNK_TOKEN]) for token in tokens
        ]

        if max_len is not None:
            if len(indices) > max_len:
                indices = indices[:max_len]
            else:
                indices += [self.stoi[Config.PAD_TOKEN]] * (max_len - len(indices))
        return indices

    def __len__(self):
        return len(self.itos)


def get_vocab(load_cached_data=True):
    vocab = Vocabulary()
    if load_cached_data and os.path.exists(Config.VOCAB_PATH):
        vocab.load(Config.VOCAB_PATH)
    else:
        # Build from training data sample to save memory/time
        # In a real scenario, we might stream the whole file
        texts = []
        sample_size = 50000
        print(f"Sampling {sample_size} records for vocabulary construction...")
        with open(Config.TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                entry = json.loads(line)
                texts.append(entry.get("question_text", ""))
                texts.append(entry.get("document_text", ""))

        vocab.build(texts)
        # Ensure cache directory exists
        os.makedirs(os.path.dirname(Config.VOCAB_PATH), exist_ok=True)
        vocab.save(Config.VOCAB_PATH)
    return vocab


def precompute_features(
    metadata_path,
    raw_data_path,
    output_path,
    vocab,
    load_cached_data=True,
    is_train=True,
):
    """
    Tokenizes text and saves indices to Parquet.
    """
    if load_cached_data and os.path.exists(output_path):
        print(f"Loading cached features from {output_path}")
        return pd.read_parquet(output_path)

    print(f"Precomputing features for {raw_data_path}...")

    # Load metadata to know which examples to process
    meta_df = pd.read_csv(metadata_path)
    # Create a set of example_ids for fast lookup
    valid_ids = set(meta_df["example_id"].astype(str))

    # We will store processed rows here
    processed_rows = []

    # Read raw JSONL
    # Since JSONL is not random access, we iterate and filter
    with open(raw_data_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            ex_id = str(entry["example_id"])

            if ex_id not in valid_ids:
                continue

            # 1. Process Question
            q_text = entry.get("question_text", "")
            q_ids = vocab.encode(q_text, max_len=Config.MAX_Q_LEN)

            # 2. Process Document Text & Candidates
            doc_text = entry.get("document_text", "")
            doc_tokens = doc_text.split()

            candidates_data = []

            # Extract candidates
            # We need to map document tokens to candidate tokens
            # candidate['start_token'] is index in doc_tokens

            raw_candidates = entry.get("long_answer_candidates", [])

            # Get annotations if training
            target_long_idx = -1
            short_starts = []
            short_ends = []
            yes_no = "NONE"

            if is_train:
                anns = entry.get("annotations", [])
                if anns:
                    ann = anns[0]
                    target_long_idx = ann.get("long_answer", {}).get(
                        "candidate_index", -1
                    )
                    yes_no = ann.get("yes_no_answer", "NONE")

                    shorts = ann.get("short_answers", [])
                    for s in shorts:
                        short_starts.append(s["start_token"])
                        short_ends.append(s["end_token"])

            # Process each candidate
            # We store: token_ids, original_start_token (for mapping short answers)
            for idx, cand in enumerate(raw_candidates):
                start = cand["start_token"]
                end = cand["end_token"]

                # Extract text span
                # Guard against indices out of bounds
                c_tokens = doc_tokens[start:end]

                # Encode
                # We prepend <NULL> (index 2) to represent "no short answer here" or "CLS"
                # This shifts indices by 1
                c_ids = [vocab.stoi["<NULL>"]] + vocab.encode(
                    c_tokens, max_len=Config.MAX_SEQ_LEN - 1
                )

                # Determine labels for this candidate
                is_correct_long = idx == target_long_idx

                # Determine short answer targets relative to this candidate
                # Default 0 (<NULL>)
                s_start_target = 0
                s_end_target = 0

                if is_correct_long and short_starts:
                    # Use the first short answer
                    s_s_doc = short_starts[0]
                    s_e_doc = short_ends[0]

                    # Check if short answer is actually inside this candidate
                    if s_s_doc >= start and s_e_doc <= end:
                        # Relative index + 1 for NULL token
                        rel_s = s_s_doc - start + 1
                        rel_e = s_e_doc - start + 1

                        # Check if within truncated length
                        if rel_e < len(c_ids):
                            s_start_target = rel_s
                            s_end_target = rel_e

                candidates_data.append(
                    {
                        "cand_ids": c_ids,
                        "is_correct": is_correct_long,
                        "s_start": s_start_target,
                        "s_end": s_end_target,
                    }
                )

            # Map yes_no to int
            yn_map = {"NONE": 0, "YES": 1, "NO": 2}
            yn_label = yn_map.get(yes_no, 0)

            processed_rows.append(
                {
                    "example_id": ex_id,
                    "q_ids": q_ids,
                    "candidates": candidates_data,
                    "yes_no_label": yn_label,
                }
            )

            # Debug break
            if (
                Config.DEBUG_SAMPLE_SIZE
                and len(processed_rows) >= Config.DEBUG_SAMPLE_SIZE
            ):
                break

    # Convert to DataFrame
    df = pd.DataFrame(processed_rows)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)
    print(f"Saved features to {output_path}")

    return df


class NQDataset(Dataset):
    def __init__(
        self,
        features_df,
        is_train=True,
        neg_ratio=Config.NEGATIVE_SAMPLING_RATIO,
        filter_negatives=True,
    ):
        self.data = features_df
        self.is_train = is_train
        self.neg_ratio = neg_ratio
        # Cite debug_lesson_11: Decouple data filtering from execution mode to ensure validation sees full distribution.
        self.filter_negatives = filter_negatives

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        q_ids = row["q_ids"]
        candidates = row["candidates"]  # List of dicts
        yn_label = row["yes_no_label"]
        example_id = row["example_id"]

        # If inference, return all candidates
        if not self.is_train:
            # Return format: question, list of candidates, example_id
            # We will batch this in collate
            return {"q_ids": q_ids, "candidates": candidates, "example_id": example_id}

        # Training: Negative Sampling
        # Find positive candidate
        pos_indices = [i for i, c in enumerate(candidates) if c["is_correct"]]

        selected_candidates = []

        # Add positive(s)
        if pos_indices:
            # Usually only one long answer, but code handles list
            for i in pos_indices:
                cand = candidates[i]
                selected_candidates.append(
                    {
                        "ids": cand["cand_ids"],
                        "label_long": 1.0,
                        "label_start": cand["s_start"],
                        "label_end": cand["s_end"],
                        "label_yn": yn_label,  # Only positive candidate gets Y/N supervision
                    }
                )

        # Add negatives
        # Identify negative pool
        neg_indices = [i for i in range(len(candidates)) if i not in pos_indices]

        sampled_neg_indices = []
        if neg_indices:
            if self.filter_negatives:
                # Determine how many negatives to sample
                num_pos = max(
                    1, len(pos_indices)
                )  # If no positive, we treat it as 1 "slot" for ratio
                num_neg = int(num_pos * self.neg_ratio)

                if len(neg_indices) > num_neg:
                    sampled_neg_indices = random.sample(neg_indices, num_neg)
                else:
                    sampled_neg_indices = (
                        neg_indices  # Take all if fewer than requested
                    )
            else:
                # Cite debug_lesson_11: Retain all negatives for validation to preserve data distribution.
                sampled_neg_indices = neg_indices

            for i in sampled_neg_indices:
                cand = candidates[i]
                selected_candidates.append(
                    {
                        "ids": cand["cand_ids"],
                        "label_long": 0.0,
                        "label_start": 0,  # Points to NULL
                        "label_end": 0,  # Points to NULL
                        "label_yn": 0,  # NONE
                    }
                )

        # If no candidates at all (edge case), return empty or dummy?
        # NQ data usually has candidates. If no positive and no negative (empty doc?), skip.
        # But we must return something.
        if not selected_candidates:
            # Dummy candidate
            selected_candidates.append(
                {
                    "ids": [2] + [0] * (Config.MAX_SEQ_LEN - 1),  # NULL + PAD
                    "label_long": 0.0,
                    "label_start": 0,
                    "label_end": 0,
                    "label_yn": 0,
                }
            )

        return {"q_ids": q_ids, "selected_candidates": selected_candidates}


def collate_fn(batch):
    """
    Batches data.
    For training: Flattens the (Question, Candidate) pairs.
    For inference: Keeps structure or flattens but tracks counts.
    """
    # Check if training mode based on keys
    is_train = "selected_candidates" in batch[0]

    batch_q_ids = []
    batch_c_ids = []

    # Targets
    batch_long_labels = []
    batch_start_labels = []
    batch_end_labels = []
    batch_yn_labels = []

    # Inference tracking
    batch_example_ids = []
    batch_candidate_counts = []  # How many candidates per question

    pad_id = 0  # Assuming 0 is PAD in vocab

    for item in batch:
        q = item["q_ids"]

        if is_train:
            cands = item["selected_candidates"]
            for c in cands:
                batch_q_ids.append(q)
                batch_c_ids.append(c["ids"])
                batch_long_labels.append(c["label_long"])
                batch_start_labels.append(c["label_start"])
                batch_end_labels.append(c["label_end"])
                batch_yn_labels.append(c["label_yn"])
        else:
            # Inference
            ex_id = item["example_id"]
            cands = item["candidates"]
            batch_example_ids.append(ex_id)
            batch_candidate_counts.append(len(cands))

            for c in cands:
                batch_q_ids.append(q)
                batch_c_ids.append(c["cand_ids"])

    # Convert to tensors and pad
    # Questions
    # Find max len in this batch
    max_q = max(len(x) for x in batch_q_ids)
    q_tensor = torch.full((len(batch_q_ids), max_q), pad_id, dtype=torch.long)
    for i, seq in enumerate(batch_q_ids):
        q_tensor[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    # Candidates
    max_c = max(len(x) for x in batch_c_ids)
    c_tensor = torch.full((len(batch_c_ids), max_c), pad_id, dtype=torch.long)
    for i, seq in enumerate(batch_c_ids):
        c_tensor[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    output = {"q_input": q_tensor, "c_input": c_tensor}

    if is_train:
        output["label_long"] = torch.tensor(batch_long_labels, dtype=torch.float32)
        output["label_start"] = torch.tensor(batch_start_labels, dtype=torch.long)
        output["label_end"] = torch.tensor(batch_end_labels, dtype=torch.long)
        output["label_yn"] = torch.tensor(batch_yn_labels, dtype=torch.long)
    else:
        output["example_ids"] = batch_example_ids
        output["candidate_counts"] = batch_candidate_counts

    return output


def get_dataloaders(vocab, load_cached_data=True):
    # 1. Prepare Train Features
    train_df = precompute_features(
        Config.TRAIN_META_PATH,
        Config.TRAIN_DATA_PATH,
        Config.TRAIN_FEATURES_PATH,
        vocab,
        load_cached_data=load_cached_data,
        is_train=True,
    )

    # 2. Prepare Val Features
    val_df = precompute_features(
        Config.VAL_META_PATH,
        Config.TRAIN_DATA_PATH,  # Val comes from train file
        Config.VAL_FEATURES_PATH,
        vocab,
        load_cached_data=load_cached_data,
        is_train=True,
    )

    # 3. Datasets
    train_ds = NQDataset(train_df, is_train=True, filter_negatives=True)
    # Cite debug_lesson_11: Disable negative filtering for validation to use full dataset.
    val_ds = NQDataset(val_df, is_train=True, filter_negatives=False)

    # 4. Loaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(vocab, load_cached_data=True):
    test_df = precompute_features(
        Config.TEST_META_PATH,
        Config.TEST_DATA_PATH,
        Config.TEST_FEATURES_PATH,
        vocab,
        load_cached_data=load_cached_data,
        is_train=False,
    )

    test_ds = NQDataset(test_df, is_train=False)

    # Batch size can be larger for inference, but collate flattens candidates
    # A single question can have 20+ candidates. Batch size 16 questions -> ~320 sequences
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
