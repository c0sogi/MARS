import os
import json
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from library.config import Config
from library.utils import tokenize, parse_annotation_record


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_vocab(metadata_df, load_cached_data=True):
    """
    Builds a vocabulary from the training data or loads it from cache.
    """
    cache_path = Config.VOCAB_CACHE_FILE

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading vocabulary from {cache_path}...")
        try:
            vocab = np.load(cache_path, allow_pickle=True).item()
            return vocab
        except Exception as e:
            print(f"Failed to load vocab cache: {e}. Rebuilding...")

    print("Building vocabulary from training data...")
    vocab_counts = Counter()

    # Sample data for vocab building if dataset is large
    sample_size = 10000
    if len(metadata_df) > sample_size:
        sample_df = metadata_df.sample(n=sample_size, random_state=Config.SEED)
    else:
        sample_df = metadata_df

    data_file = Config.TRAIN_DATA_FILE

    with open(data_file, "rb") as f:
        for _, row in sample_df.iterrows():
            f.seek(row["byte_offset"])
            line = f.readline()
            if not line:
                continue

            try:
                entry = json.loads(line)
                # Add question tokens
                q_tokens = tokenize(entry.get("question_text", ""))
                vocab_counts.update(q_tokens)

                # Add document tokens (sample first 1000 to save time/memory)
                doc_text = entry.get("document_text", "")
                doc_tokens = tokenize(doc_text)[:1000]
                vocab_counts.update(doc_tokens)
            except json.JSONDecodeError:
                continue

    # Create vocab dictionary
    # Start with special tokens
    vocab = {Config.PAD_TOKEN: 0, Config.UNK_TOKEN: 1}

    # Add most frequent words up to VOCAB_SIZE - 2
    most_common = vocab_counts.most_common(Config.VOCAB_SIZE - 2)
    for word, _ in most_common:
        vocab[word] = len(vocab)

    # Save to cache
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, vocab)
        print(f"Saved vocabulary to {cache_path}")
    except Exception as e:
        print(f"Failed to save vocab cache: {e}")

    return vocab


class NQDataset(Dataset):
    """
    Base dataset class for Natural Questions that handles file seeking.
    """

    def __init__(self, metadata_df, data_file_path, vocab):
        self.metadata_df = metadata_df
        self.data_file_path = data_file_path
        self.vocab = vocab
        # We don't open the file here to be safe with multiprocessing workers.
        # We open it in __getitem__ or use a worker_init_fn logic if optimization is needed.
        # For simplicity and robustness within 24h, we open per read or keep a handle if possible.

    def _read_json_at_offset(self, byte_offset):
        with open(self.data_file_path, "rb") as f:
            f.seek(byte_offset)
            line = f.readline()
            return json.loads(line)

    def _text_to_indices(self, text, max_len):
        tokens = tokenize(text)
        indices = [self.vocab.get(t, self.vocab[Config.UNK_TOKEN]) for t in tokens]
        # Truncate
        if len(indices) > max_len:
            indices = indices[:max_len]
        # Pad
        if len(indices) < max_len:
            indices += [self.vocab[Config.PAD_TOKEN]] * (max_len - len(indices))
        return np.array(indices, dtype=np.int64)

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        raise NotImplementedError


class LongAnswerDataset(NQDataset):
    """
    Dataset for the Long Answer Ranking task (Siamese 1D-CNN).
    """

    def __init__(
        self, metadata_df, data_file_path, vocab, split="train", load_cached_data=True
    ):
        super().__init__(metadata_df, data_file_path, vocab)
        self.split = split
        self.samples = self._prepare_samples(metadata_df, split, load_cached_data)

    def _prepare_samples(self, metadata_df, split, load_cached_data):
        """
        Prepares a list of (byte_offset, candidate_index, label) tuples.
        Handles negative sampling for training.
        """
        cache_file = os.path.join(Config.WORKING_DIR, f"{split}_long_samples.parquet")

        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading {split} samples from {cache_file}...")
            return pd.read_parquet(cache_file).to_dict("records")

        print(f"Processing {split} samples for Long Answer task...")
        samples = []

        # If debugging, limit size
        if Config.TRAIN_SAMPLE_SIZE and split == "train":
            metadata_df = metadata_df.head(Config.TRAIN_SAMPLE_SIZE)

        for _, row in metadata_df.iterrows():
            offset = row["byte_offset"]

            # For test set, we don't have annotations, so we can't determine labels.
            # We just need to know how many candidates there are.
            # However, reading every line to count candidates is slow.
            # For training/val, we rely on annotations.

            if split == "test":
                # For test, we need to read the file to know candidates.
                # This is unavoidable unless we pre-processed test metadata deeper.
                # Given strict constraints, we do it here.
                try:
                    data = self._read_json_at_offset(offset)
                    candidates = data.get("long_answer_candidates", [])
                    for i in range(len(candidates)):
                        samples.append(
                            {
                                "byte_offset": offset,
                                "candidate_index": i,
                                "label": 0.0,  # Dummy label
                                "example_id": row["example_id"],
                            }
                        )
                except:
                    continue
                continue

            # For Train/Val
            try:
                anns = json.loads(row["annotations"])
            except:
                continue

            # Identify correct candidate indices
            correct_indices = set()
            for ann in anns:
                la = ann.get("long_answer", {})
                idx = la.get("candidate_index", -1)
                if idx != -1:
                    correct_indices.add(idx)

            # We need to read the file to know how many candidates exist to generate negatives
            # Optimization: We only need to read if we are generating negatives or if we need text later.
            # We read here to build the index.
            try:
                data = self._read_json_at_offset(offset)
                candidates = data.get("long_answer_candidates", [])
            except:
                continue

            num_candidates = len(candidates)

            # Positive samples
            for idx in correct_indices:
                if idx < num_candidates:
                    samples.append(
                        {
                            "byte_offset": offset,
                            "candidate_index": idx,
                            "label": 1.0,
                            "example_id": row["example_id"],
                        }
                    )

            # Negative samples
            if split == "train":
                # Randomly sample negatives
                neg_indices = [
                    i for i in range(num_candidates) if i not in correct_indices
                ]
                if neg_indices:
                    # Determine how many negatives to keep
                    num_neg = max(1, int(len(neg_indices) * Config.NEG_SAMPLE_RATIO))
                    selected_negs = random.sample(
                        neg_indices, min(len(neg_indices), num_neg)
                    )
                    for idx in selected_negs:
                        samples.append(
                            {
                                "byte_offset": offset,
                                "candidate_index": idx,
                                "label": 0.0,
                                "example_id": row["example_id"],
                            }
                        )
            else:
                # Validation: keep all or a deterministic subset.
                # Usually for Val we want to evaluate ranking, so we might keep all.
                # But to save memory/time in this baseline, let's keep a fixed ratio or all.
                # Let's keep all for validation to compute accurate metrics.
                for i in range(num_candidates):
                    label = 1.0 if i in correct_indices else 0.0
                    samples.append(
                        {
                            "byte_offset": offset,
                            "candidate_index": i,
                            "label": label,
                            "example_id": row["example_id"],
                        }
                    )

        # Save to cache
        df_samples = pd.DataFrame(samples)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        df_samples.to_parquet(cache_file, index=False)
        print(f"Saved {len(df_samples)} samples to {cache_file}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        offset = sample["byte_offset"]
        cand_idx = sample["candidate_index"]
        label = sample["label"]

        data = self._read_json_at_offset(offset)

        # Question
        q_text = data.get("question_text", "")
        q_indices = self._text_to_indices(q_text, Config.MAX_QUES_LEN)

        # Candidate Text
        candidates = data.get("long_answer_candidates", [])
        if cand_idx < len(candidates):
            cand = candidates[cand_idx]
            # Extract text based on tokens (simplified approximation: slice word list)
            # The dataset provides document_text as a string. We need to slice it.
            # However, token indices in candidates refer to whitespace-split tokens.
            doc_tokens = tokenize(data.get("document_text", ""))
            start = cand["start_token"]
            end = cand["end_token"]

            # Safety check
            start = max(0, start)
            end = min(len(doc_tokens), end)

            cand_tokens_list = doc_tokens[start:end]
            cand_text = " ".join(cand_tokens_list)
        else:
            cand_text = ""

        c_indices = self._text_to_indices(cand_text, Config.MAX_SEQ_LEN)

        return {
            "question": torch.tensor(q_indices, dtype=torch.long),
            "candidate": torch.tensor(c_indices, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32),
            "example_id": sample["example_id"],
            "candidate_index": cand_idx,
        }


def process_short_answer_data(
    metadata_df, data_file_path, vocab, load_cached_data=True
):
    """
    Generates features (X) and labels (y) for the Short Answer Logistic Regression.
    Only processes training data where a valid long answer exists.
    """
    cache_path = Config.SHORT_ANSWER_DATA_CACHE

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading short answer data from {cache_path}...")
        try:
            data = pd.read_parquet(cache_path)
            # Assuming last column is y, rest are X
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
            return X, y
        except Exception as e:
            print(f"Failed to load short answer cache: {e}. Recomputing...")

    print("Processing short answer data (Sliding Windows)...")

    features_list = []
    labels_list = []

    # Use a subset for feature generation to avoid OOM if dataset is huge
    if Config.TRAIN_SAMPLE_SIZE:
        metadata_df = metadata_df.head(Config.TRAIN_SAMPLE_SIZE)

    with open(data_file_path, "rb") as f:
        for _, row in metadata_df.iterrows():
            try:
                anns = json.loads(row["annotations"])
            except:
                continue

            # Find the ground truth long answer and short answer
            valid_long_ann = None
            short_answers = []

            for ann in anns:
                la = ann.get("long_answer", {})
                if la.get("candidate_index", -1) != -1:
                    valid_long_ann = la
                    short_answers = ann.get("short_answers", [])
                    break

            if not valid_long_ann or not short_answers:
                continue

            # Read data
            f.seek(row["byte_offset"])
            line = f.readline()
            if not line:
                continue
            data = json.loads(line)

            doc_tokens = tokenize(data.get("document_text", ""))
            q_text = data.get("question_text", "")
            q_tokens_set = set(tokenize(q_text.lower()))

            # Get Long Answer Text
            la_start = valid_long_ann["start_token"]
            la_end = valid_long_ann["end_token"]
            la_tokens = doc_tokens[la_start:la_end]

            # Generate Sliding Windows
            # Window size and stride
            w_size = Config.WINDOW_SIZE
            stride = Config.WINDOW_STRIDE

            # Ground truth short answer spans relative to the document
            gt_spans = [(sa["start_token"], sa["end_token"]) for sa in short_answers]

            for i in range(0, len(la_tokens) - w_size + 1, stride):
                window_tokens = la_tokens[i : i + w_size]

                # Absolute positions in document
                w_abs_start = la_start + i
                w_abs_end = w_abs_start + w_size

                # Compute Label (IoU > 0.5 with any ground truth short answer)
                is_positive = 0
                for gt_start, gt_end in gt_spans:
                    # Intersection
                    inter_start = max(w_abs_start, gt_start)
                    inter_end = min(w_abs_end, gt_end)
                    inter = max(0, inter_end - inter_start)

                    # Union
                    union = (w_abs_end - w_abs_start) + (gt_end - gt_start) - inter

                    if (
                        union > 0 and (inter / union) > 0.3
                    ):  # Threshold for positive window
                        is_positive = 1
                        break

                # Compute Features
                # 1. Contains Digit
                has_digit = 1 if any(t.isdigit() for t in window_tokens) else 0

                # 2. Is Capitalized (first token)
                is_cap = 1 if window_tokens and window_tokens[0][0].isupper() else 0

                # 3. Exact Match Count with Question
                match_count = sum(1 for t in window_tokens if t.lower() in q_tokens_set)

                # 4. Relative Position in Long Answer
                rel_pos = i / len(la_tokens) if len(la_tokens) > 0 else 0

                features_list.append([has_digit, is_cap, match_count, rel_pos])
                labels_list.append(is_positive)

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.float32)

    # Save to cache
    df_out = pd.DataFrame(X, columns=["has_digit", "is_cap", "match_count", "rel_pos"])
    df_out["label"] = y

    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df_out.to_parquet(cache_path, index=False)
        print(f"Saved short answer data to {cache_path}. Shape: {df_out.shape}")
    except Exception as e:
        print(f"Failed to save short answer cache: {e}")

    return X, y


def get_long_answer_loader(
    metadata_df,
    data_file_path,
    vocab,
    split="train",
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    load_cached_data=True,
):
    """
    Returns a DataLoader for the Long Answer task.
    """
    dataset = LongAnswerDataset(
        metadata_df,
        data_file_path,
        vocab,
        split=split,
        load_cached_data=load_cached_data,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Avoid multiprocessing complexity for file handles in this baseline
        pin_memory=True,
    )
    return loader
