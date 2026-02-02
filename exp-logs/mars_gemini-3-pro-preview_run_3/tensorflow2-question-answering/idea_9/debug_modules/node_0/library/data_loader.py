import os
import json
import random
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.preprocessing import TextPreprocessor


def collate_fn(batch):
    """
    Custom collate function to handle dynamic padding.
    """
    batch_type = batch[0].get("type")

    if batch_type == "ranker":
        q_seqs = [item["q_indices"] for item in batch]
        doc_seqs = [item["doc_indices"] for item in batch]
        labels = [item["label"] for item in batch]

        q_padded = pad_sequence(q_seqs, batch_first=True, padding_value=0)
        doc_padded = pad_sequence(doc_seqs, batch_first=True, padding_value=0)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return {
            "q_indices": q_padded,
            "doc_indices": doc_padded,
            "labels": labels_tensor,
        }

    elif batch_type == "reader":
        input_seqs = [item["input_indices"] for item in batch]
        start_positions = [item["start_idx"] for item in batch]
        end_positions = [item["end_idx"] for item in batch]

        input_padded = pad_sequence(input_seqs, batch_first=True, padding_value=0)
        start_tensor = torch.tensor(start_positions, dtype=torch.long)
        end_tensor = torch.tensor(end_positions, dtype=torch.long)

        return {
            "input_indices": input_padded,
            "start_positions": start_tensor,
            "end_positions": end_tensor,
        }

    elif batch_type == "ranker_test":
        q_seqs = [item["q_indices"] for item in batch]
        doc_seqs = [item["doc_indices"] for item in batch]
        example_ids = [item["example_id"] for item in batch]
        cand_indices = [item["candidate_index"] for item in batch]
        cand_texts = [item["candidate_text"] for item in batch]

        q_padded = pad_sequence(q_seqs, batch_first=True, padding_value=0)
        doc_padded = pad_sequence(doc_seqs, batch_first=True, padding_value=0)

        return {
            "q_indices": q_padded,
            "doc_indices": doc_padded,
            "example_ids": example_ids,
            "candidate_indices": cand_indices,
            "candidate_texts": cand_texts,
        }

    return batch


class NQRankerDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        raw_file,
        preprocessor,
        is_train=True,
        load_cached_data=True,
        sample_size=None,
    ):
        self.preprocessor = preprocessor
        self.is_train = is_train
        self.raw_file = raw_file
        self.metadata_path = metadata_path
        self.sample_size = sample_size

        # Determine cache path based on split
        if is_train:
            self.cache_path = Config.RANKER_TRAIN_CACHE
        else:
            self.cache_path = Config.RANKER_VAL_CACHE

        self.data = self._process_data(load_cached_data)

    def _process_data(self, load_cached_data):
        # 1. Try loading from cache
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading Ranker data from {self.cache_path}")
            try:
                df = pd.read_parquet(self.cache_path)
                # Ensure lists are lists (sometimes parquet might store as strings or arrays)
                # Pandas read_parquet usually handles lists correctly if pyarrow is used
                return df.to_dict("records")
            except Exception as e:
                print(f"Failed to load Ranker cache: {e}. Recomputing...")

        # 2. Compute from scratch
        print(f"Processing Ranker data (is_train={self.is_train})...")

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")

        metadata_df = pd.read_csv(self.metadata_path)

        # Filter for samples that actually have long answers if training
        # If validating, we might want to evaluate on all, but for Ranker training we need positives.
        if self.is_train:
            metadata_df = metadata_df[metadata_df["has_long_answer"] == True]

        if self.sample_size:
            if len(metadata_df) > self.sample_size:
                metadata_df = metadata_df.sample(
                    n=self.sample_size, random_state=Config.SEED
                )

        processed_samples = []

        with open(self.raw_file, "rb") as f:
            for _, row in metadata_df.iterrows():
                offset = row["byte_offset"]
                f.seek(offset)
                line = f.readline()
                if not line:
                    continue

                try:
                    data = json.loads(line.decode("utf-8"))
                    q_text = data.get("question_text", "")
                    doc_text = data.get("document_text", "")
                    doc_tokens = self.preprocessor.tokenize(doc_text)

                    candidates = data.get("long_answer_candidates", [])
                    annotations = data.get("annotations", [])

                    # Find ground truth candidate index
                    gt_candidate_idx = -1
                    for ann in annotations:
                        la = ann.get("long_answer", {})
                        start_token = la.get("start_token", -1)
                        if start_token != -1:
                            # Find which candidate matches this start token
                            for idx, cand in enumerate(candidates):
                                if (
                                    cand["top_level"]
                                    and cand["start_token"] == start_token
                                ):
                                    gt_candidate_idx = idx
                                    break
                        if gt_candidate_idx != -1:
                            break

                    # If no ground truth found and we are training, skip
                    if gt_candidate_idx == -1 and self.is_train:
                        continue

                    # Collect valid top-level candidates
                    valid_candidates = []
                    for idx, cand in enumerate(candidates):
                        if cand["top_level"]:
                            start = cand["start_token"]
                            end = cand["end_token"]
                            cand_text = " ".join(doc_tokens[start:end])
                            valid_candidates.append((idx, cand_text))

                    if not valid_candidates:
                        continue

                    # Create Positive Sample
                    if gt_candidate_idx != -1:
                        # Find the text for the ground truth candidate
                        gt_text = ""
                        for idx, text in valid_candidates:
                            if idx == gt_candidate_idx:
                                gt_text = text
                                break

                        if gt_text:
                            q_indices = self.preprocessor.text_to_indices(
                                q_text, Config.MAX_Q_LEN
                            ).tolist()
                            doc_indices = self.preprocessor.text_to_indices(
                                gt_text, Config.MAX_DOC_LEN
                            ).tolist()

                            processed_samples.append(
                                {
                                    "q_indices": q_indices,
                                    "doc_indices": doc_indices,
                                    "label": 1.0,
                                }
                            )

                    # Create Negative Samples
                    # Filter out the ground truth index
                    neg_candidates = [
                        c for c in valid_candidates if c[0] != gt_candidate_idx
                    ]

                    if neg_candidates:
                        # Randomly sample negatives
                        num_neg = Config.NEG_RATIO
                        if len(neg_candidates) > num_neg:
                            selected_negs = random.sample(neg_candidates, num_neg)
                        else:
                            selected_negs = neg_candidates

                        for _, neg_text in selected_negs:
                            q_indices = self.preprocessor.text_to_indices(
                                q_text, Config.MAX_Q_LEN
                            ).tolist()
                            doc_indices = self.preprocessor.text_to_indices(
                                neg_text, Config.MAX_DOC_LEN
                            ).tolist()

                            processed_samples.append(
                                {
                                    "q_indices": q_indices,
                                    "doc_indices": doc_indices,
                                    "label": 0.0,
                                }
                            )

                except json.JSONDecodeError:
                    continue

        # 3. Save to cache
        print(f"Saving {len(processed_samples)} Ranker samples to {self.cache_path}")
        df = pd.DataFrame(processed_samples)
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        df.to_parquet(self.cache_path, index=False)

        return processed_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "type": "ranker",
            "q_indices": torch.tensor(item["q_indices"], dtype=torch.long),
            "doc_indices": torch.tensor(item["doc_indices"], dtype=torch.long),
            "label": item["label"],
        }


class NQReaderDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        raw_file,
        preprocessor,
        is_train=True,
        load_cached_data=True,
        sample_size=None,
    ):
        self.preprocessor = preprocessor
        self.is_train = is_train
        self.raw_file = raw_file
        self.metadata_path = metadata_path
        self.sample_size = sample_size

        if is_train:
            self.cache_path = Config.READER_TRAIN_CACHE
        else:
            self.cache_path = Config.READER_VAL_CACHE

        self.data = self._process_data(load_cached_data)

    def _process_data(self, load_cached_data):
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading Reader data from {self.cache_path}")
            try:
                df = pd.read_parquet(self.cache_path)
                return df.to_dict("records")
            except Exception as e:
                print(f"Failed to load Reader cache: {e}. Recomputing...")

        print(f"Processing Reader data (is_train={self.is_train})...")

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")

        metadata_df = pd.read_csv(self.metadata_path)

        # Only use samples with short answers
        metadata_df = metadata_df[metadata_df["has_short_answer"] == True]

        if self.sample_size:
            if len(metadata_df) > self.sample_size:
                metadata_df = metadata_df.sample(
                    n=self.sample_size, random_state=Config.SEED
                )

        processed_samples = []

        with open(self.raw_file, "rb") as f:
            for _, row in metadata_df.iterrows():
                offset = row["byte_offset"]
                f.seek(offset)
                line = f.readline()
                if not line:
                    continue

                try:
                    data = json.loads(line.decode("utf-8"))
                    q_text = data.get("question_text", "")
                    doc_text = data.get("document_text", "")
                    doc_tokens = self.preprocessor.tokenize(doc_text)
                    q_tokens = self.preprocessor.tokenize(q_text)

                    annotations = data.get("annotations", [])
                    candidates = data.get("long_answer_candidates", [])

                    # Find the first valid short answer
                    short_ans_start = -1
                    short_ans_end = -1

                    for ann in annotations:
                        if len(ann["short_answers"]) > 0:
                            short_ans_start = ann["short_answers"][0]["start_token"]
                            short_ans_end = ann["short_answers"][0]["end_token"]
                            break
                        # We could also handle Yes/No here, but sticking to spans for simplicity of this architecture
                        # If Yes/No, typically mapped to specific tokens or a special classification head.
                        # For this specific task description ("predict start:end"), we focus on spans.

                    if short_ans_start == -1:
                        continue

                    # Find containing long answer candidate
                    containing_cand = None
                    for cand in candidates:
                        if cand["top_level"]:
                            if (
                                cand["start_token"] <= short_ans_start
                                and cand["end_token"] >= short_ans_end
                            ):
                                containing_cand = cand
                                break

                    if not containing_cand:
                        continue

                    # Extract paragraph text
                    cand_start = containing_cand["start_token"]
                    cand_end = containing_cand["end_token"]
                    para_tokens = doc_tokens[cand_start:cand_end]

                    # Calculate relative indices
                    # Input sequence: Q tokens + Para tokens
                    # Start index = len(Q) + (short_start - cand_start)
                    # End index = len(Q) + (short_end - cand_start) - 1 (inclusive)

                    rel_start = len(q_tokens) + (short_ans_start - cand_start)
                    rel_end = len(q_tokens) + (short_end - cand_start) - 1

                    # Construct input text
                    input_text = " ".join(q_tokens + para_tokens)

                    # Convert to indices
                    # Note: We need to be careful about max length.
                    # If the answer is truncated, we should probably discard the sample.
                    input_indices = self.preprocessor.text_to_indices(
                        input_text, Config.MAX_Q_LEN + Config.MAX_DOC_LEN
                    ).tolist()

                    # Check if answer is within bounds
                    if rel_end >= len(input_indices):
                        continue

                    processed_samples.append(
                        {
                            "input_indices": input_indices,
                            "start_idx": rel_start,
                            "end_idx": rel_end,
                        }
                    )

                except json.JSONDecodeError:
                    continue

        print(f"Saving {len(processed_samples)} Reader samples to {self.cache_path}")
        df = pd.DataFrame(processed_samples)
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        df.to_parquet(self.cache_path, index=False)

        return processed_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "type": "reader",
            "input_indices": torch.tensor(item["input_indices"], dtype=torch.long),
            "start_idx": item["start_idx"],
            "end_idx": item["end_idx"],
        }


class NQRankerTestDataset(Dataset):
    def __init__(self, metadata_path, raw_file, preprocessor, load_cached_data=True):
        self.preprocessor = preprocessor
        self.raw_file = raw_file
        self.metadata_path = metadata_path
        self.cache_path = Config.RANKER_TEST_FEATURES_CACHE

        self.data = self._process_data(load_cached_data)

    def _process_data(self, load_cached_data):
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading Ranker Test data from {self.cache_path}")
            try:
                df = pd.read_parquet(self.cache_path)
                return df.to_dict("records")
            except Exception as e:
                print(f"Failed to load Ranker Test cache: {e}. Recomputing...")

        print("Processing Ranker Test data...")

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")

        metadata_df = pd.read_csv(self.metadata_path)
        processed_samples = []

        with open(self.raw_file, "rb") as f:
            for _, row in metadata_df.iterrows():
                offset = row["byte_offset"]
                example_id = row["example_id"]
                f.seek(offset)
                line = f.readline()
                if not line:
                    continue

                try:
                    data = json.loads(line.decode("utf-8"))
                    q_text = data.get("question_text", "")
                    doc_text = data.get("document_text", "")
                    doc_tokens = self.preprocessor.tokenize(doc_text)
                    candidates = data.get("long_answer_candidates", [])

                    q_indices = self.preprocessor.text_to_indices(
                        q_text, Config.MAX_Q_LEN
                    ).tolist()

                    for idx, cand in enumerate(candidates):
                        if cand["top_level"]:
                            start = cand["start_token"]
                            end = cand["end_token"]
                            cand_text = " ".join(doc_tokens[start:end])

                            doc_indices = self.preprocessor.text_to_indices(
                                cand_text, Config.MAX_DOC_LEN
                            ).tolist()

                            processed_samples.append(
                                {
                                    "example_id": str(example_id),
                                    "candidate_index": idx,
                                    "q_indices": q_indices,
                                    "doc_indices": doc_indices,
                                    "candidate_text": cand_text,  # Kept for Reader inference later
                                }
                            )

                except json.JSONDecodeError:
                    continue

        print(
            f"Saving {len(processed_samples)} Ranker Test samples to {self.cache_path}"
        )
        df = pd.DataFrame(processed_samples)
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        df.to_parquet(self.cache_path, index=False)

        return processed_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "type": "ranker_test",
            "q_indices": torch.tensor(item["q_indices"], dtype=torch.long),
            "doc_indices": torch.tensor(item["doc_indices"], dtype=torch.long),
            "example_id": item["example_id"],
            "candidate_index": item["candidate_index"],
            "candidate_text": item["candidate_text"],
        }
