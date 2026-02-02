import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.vocab import Vocabulary


class NQDataset(Dataset):
    def __init__(
        self, split="train", vocab=None, load_cached_data=True, sample_size=None
    ):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            vocab (Vocabulary): Vocabulary object for tokenization.
            load_cached_data (bool): If True, try to load processed features from parquet cache.
            sample_size (int, optional): Limit number of examples for debugging.
        """
        self.split = split
        self.vocab = vocab if vocab else Vocabulary()
        self.load_cached_data = load_cached_data
        self.sample_size = sample_size

        # Resolve paths based on split
        if split == "train":
            self.metadata_path = Config.TRAIN_META_PATH
            self.raw_data_path = Config.TRAIN_DATA_PATH
            self.cache_path = Config.TRAIN_FEATURES_CACHE_PATH
        elif split == "val":
            self.metadata_path = Config.VAL_META_PATH
            self.raw_data_path = (
                Config.TRAIN_DATA_PATH
            )  # Validation is a subset of train file
            self.cache_path = Config.VAL_FEATURES_CACHE_PATH
        elif split == "test":
            self.metadata_path = Config.TEST_META_PATH
            self.raw_data_path = Config.TEST_DATA_PATH
            self.cache_path = Config.TEST_FEATURES_CACHE_PATH
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")

        # Load data
        self.data = self._load_or_process_data()

    def _load_or_process_data(self):
        # 1. Check Cache
        if self.load_cached_data and os.path.exists(self.cache_path):
            try:
                df = pd.read_parquet(self.cache_path)
                # If debugging with a smaller sample size than cached, slice it
                if self.sample_size is not None and len(df) > self.sample_size:
                    df = df.head(self.sample_size)
                return df
            except Exception as e:
                print(f"Error loading cache for {self.split}: {e}. Recomputing...")

        # 2. Process from scratch
        df = self._process_raw_data()

        # 3. Save to Cache
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        df.to_parquet(self.cache_path, index=False)

        return df

    def _process_raw_data(self):
        # Load metadata to know which examples belong to this split and their labels
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        meta_df = pd.read_csv(self.metadata_path)

        if self.sample_size:
            meta_df = meta_df.head(self.sample_size)

        # Create a lookup for targets
        # Structure: example_id -> {long_idx, yes_no}
        target_map = {}
        target_ids = set()

        # Test metadata does not have labels
        has_labels = self.split != "test"

        for _, row in meta_df.iterrows():
            eid = str(row["example_id"])
            target_ids.add(eid)
            if has_labels:
                target_map[eid] = {
                    "long_idx": int(row["long_answer_index"]),
                    "yes_no": row["yes_no_answer"],
                }

        processed_samples = []

        # Stream raw JSONL
        with open(self.raw_data_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                eid = str(entry["example_id"])

                # Skip if not in this split
                if eid not in target_ids:
                    continue

                # Basic Extraction
                q_text = entry["question_text"]
                doc_text = entry["document_text"]
                doc_tokens = doc_text.split()  # NQ uses whitespace tokenization
                candidates = entry["long_answer_candidates"]

                # Encode Question (Global for all candidates in this example)
                q_indices = self.vocab.text_to_indices(q_text, Config.MAX_SEQ_LEN_Q)

                # Determine Labels (if available)
                correct_long_idx = -1
                short_answers = []
                yes_no_label = "NONE"

                if has_labels:
                    labels = target_map[eid]
                    correct_long_idx = labels["long_idx"]
                    yes_no_label = labels["yes_no"]

                    # Extract short answer spans from raw annotations
                    # We assume first annotation is ground truth
                    anns = entry.get("annotations", [])
                    if anns:
                        short_answers = anns[0].get("short_answers", [])

                # --- Processing Logic ---

                if self.split == "train":
                    # Training: Negative Sampling

                    if correct_long_idx != -1:
                        # 1. Positive Sample
                        pos_cand = candidates[correct_long_idx]
                        self._add_sample(
                            processed_samples,
                            eid,
                            q_indices,
                            doc_tokens,
                            pos_cand,
                            label=1,
                            short_answers=short_answers,
                            yes_no=yes_no_label,
                        )

                        # 2. Negative Samples
                        neg_indices = [
                            i for i in range(len(candidates)) if i != correct_long_idx
                        ]
                        if neg_indices:
                            # Sample k negatives
                            k = min(len(neg_indices), Config.NEG_SAMPLING_RATIO)
                            chosen_negs = random.sample(neg_indices, k)
                            for neg_idx in chosen_negs:
                                neg_cand = candidates[neg_idx]
                                self._add_sample(
                                    processed_samples,
                                    eid,
                                    q_indices,
                                    doc_tokens,
                                    neg_cand,
                                    label=0,
                                    short_answers=[],
                                    yes_no="NONE",
                                )
                    else:
                        # No long answer. Sample one negative to provide "background" examples
                        # of non-answers, or skip. To maintain balance, we sample 1 negative.
                        if candidates:
                            neg_idx = random.choice(range(len(candidates)))
                            neg_cand = candidates[neg_idx]
                            self._add_sample(
                                processed_samples,
                                eid,
                                q_indices,
                                doc_tokens,
                                neg_cand,
                                label=0,
                                short_answers=[],
                                yes_no="NONE",
                            )

                else:
                    # Validation / Test: Process ALL candidates
                    # We need to rank every candidate to find the best one.
                    for idx, cand in enumerate(candidates):
                        # Determine label for validation metrics
                        is_correct = idx == correct_long_idx
                        lbl = 1 if is_correct else 0
                        s_ans = short_answers if is_correct else []
                        yn = yes_no_label if is_correct else "NONE"

                        self._add_sample(
                            processed_samples,
                            eid,
                            q_indices,
                            doc_tokens,
                            cand,
                            label=lbl,
                            short_answers=s_ans,
                            yes_no=yn,
                            cand_idx=idx,
                        )

        return pd.DataFrame(processed_samples)

    def _add_sample(
        self,
        rows,
        eid,
        q_indices,
        doc_tokens,
        candidate,
        label,
        short_answers,
        yes_no,
        cand_idx=-1,
    ):
        """
        Helper to process a single candidate and append to list.
        """
        start = candidate["start_token"]
        end = candidate["end_token"]

        # Safety check for indices
        if start < 0 or end > len(doc_tokens) or start >= end:
            return

        # Extract text and encode
        cand_tokens_list = doc_tokens[start:end]
        cand_text = " ".join(cand_tokens_list)
        c_indices = self.vocab.text_to_indices(cand_text, Config.MAX_SEQ_LEN_C)

        # Generate Attention Mask (Ground Truth for Short Answer)
        # Mask is 1.0 if token is part of short answer, 0.0 otherwise
        attn_mask = np.zeros(Config.MAX_SEQ_LEN_C, dtype=np.float32)

        if label == 1 and short_answers:
            for s_ans in short_answers:
                s_start = s_ans["start_token"]
                s_end = s_ans["end_token"]

                # Calculate intersection between candidate and short answer
                inter_start = max(start, s_start)
                inter_end = min(end, s_end)

                if inter_start < inter_end:
                    # Convert global indices to candidate-relative indices
                    rel_start = inter_start - start
                    rel_end = inter_end - start

                    # Clip to max sequence length
                    rel_start = min(rel_start, Config.MAX_SEQ_LEN_C)
                    rel_end = min(rel_end, Config.MAX_SEQ_LEN_C)

                    if rel_start < rel_end:
                        attn_mask[rel_start:rel_end] = 1.0

        # Encode Yes/No
        yn_map = {"NONE": 0, "YES": 1, "NO": 2}
        yn_int = yn_map.get(yes_no, 0)

        rows.append(
            {
                "example_id": eid,
                "q_indices": q_indices,
                "c_indices": c_indices,
                "label": float(label),
                "attn_mask": attn_mask.tolist(),  # Store as list for Parquet
                "yes_no": yn_int,
                "cand_global_start": start,
                "cand_global_end": end,
                "cand_index_in_list": cand_idx,
            }
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        return {
            "example_id": row["example_id"],
            "q_indices": torch.tensor(row["q_indices"], dtype=torch.long),
            "c_indices": torch.tensor(row["c_indices"], dtype=torch.long),
            "label": torch.tensor(row["label"], dtype=torch.float),
            "attn_mask": torch.tensor(row["attn_mask"], dtype=torch.float),
            "yes_no": torch.tensor(row["yes_no"], dtype=torch.long),
            "cand_global_start": row["cand_global_start"],
            "cand_global_end": row["cand_global_end"],
            "cand_index_in_list": row["cand_index_in_list"],
        }


def collate_fn(batch):
    """
    Custom collate function to stack tensors and collect metadata.
    """
    q_indices = torch.stack([b["q_indices"] for b in batch])
    c_indices = torch.stack([b["c_indices"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    attn_masks = torch.stack([b["attn_mask"] for b in batch])
    yes_nos = torch.stack([b["yes_no"] for b in batch])

    # Metadata lists (not tensors)
    example_ids = [b["example_id"] for b in batch]
    starts = [b["cand_global_start"] for b in batch]
    ends = [b["cand_global_end"] for b in batch]
    cand_indices = [b["cand_index_in_list"] for b in batch]

    return {
        "q_indices": q_indices,
        "c_indices": c_indices,
        "labels": labels,
        "attn_masks": attn_masks,
        "yes_nos": yes_nos,
        "example_ids": example_ids,
        "cand_global_starts": starts,
        "cand_global_ends": ends,
        "cand_indices": cand_indices,
    }
