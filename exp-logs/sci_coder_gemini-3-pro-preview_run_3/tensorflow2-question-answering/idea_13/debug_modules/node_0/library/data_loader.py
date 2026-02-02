import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from library.config import Config
from library.text_utils import TextProcessor, Vocabulary

# --------------------------------------------------------------------------
# Dataset Classes
# --------------------------------------------------------------------------


class RankerDataset(Dataset):
    def __init__(self, data_df):
        """
        Args:
            data_df (pd.DataFrame): DataFrame containing 'q_indices', 'p_indices', 'label'.
        """
        self.data = data_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            "q_indices": row["q_indices"],
            "p_indices": row["p_indices"],
            "label": float(row["label"]),
        }


class ReaderDataset(Dataset):
    def __init__(self, data_df):
        """
        Args:
            data_df (pd.DataFrame): DataFrame containing 'q_indices', 'p_indices', 'start_token', 'end_token'.
        """
        self.data = data_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            "q_indices": row["q_indices"],
            "p_indices": row["p_indices"],
            "start_token": int(row["start_token"]),
            "end_token": int(row["end_token"]),
        }


# --------------------------------------------------------------------------
# Collate Functions
# --------------------------------------------------------------------------


def pad_sequence(sequences, max_len=None, padding_value=0):
    """Pads a list of sequences to the same length."""
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)

    padded_seqs = []
    for seq in sequences:
        if len(seq) < max_len:
            padded_seqs.append(list(seq) + [padding_value] * (max_len - len(seq)))
        else:
            padded_seqs.append(list(seq)[:max_len])
    return torch.tensor(padded_seqs, dtype=torch.long)


def collate_ranker(batch):
    q_seqs = [item["q_indices"] for item in batch]
    p_seqs = [item["p_indices"] for item in batch]
    labels = [item["label"] for item in batch]

    # Pad dynamically to max length in batch, but cap at Config limits
    q_tensor = pad_sequence(q_seqs, max_len=Config.MAX_Q_LEN, padding_value=0)
    p_tensor = pad_sequence(p_seqs, max_len=Config.MAX_DOC_LEN, padding_value=0)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    return q_tensor, p_tensor, labels_tensor


def collate_reader(batch):
    q_seqs = [item["q_indices"] for item in batch]
    p_seqs = [item["p_indices"] for item in batch]
    starts = [item["start_token"] for item in batch]
    ends = [item["end_token"] for item in batch]

    q_tensor = pad_sequence(q_seqs, max_len=Config.MAX_Q_LEN, padding_value=0)
    p_tensor = pad_sequence(p_seqs, max_len=Config.MAX_DOC_LEN, padding_value=0)

    # Ensure start/end indices do not exceed the truncated paragraph length
    max_p_len = p_tensor.size(1)
    starts_tensor = torch.clamp(
        torch.tensor(starts, dtype=torch.long), max=max_p_len - 1
    )
    ends_tensor = torch.clamp(torch.tensor(ends, dtype=torch.long), max=max_p_len - 1)

    return q_tensor, p_tensor, starts_tensor, ends_tensor


# --------------------------------------------------------------------------
# Data Processing Logic
# --------------------------------------------------------------------------


def _process_ranker_data(metadata_df, vocab, source_file, sample_size=None):
    """
    Generates training pairs for the ranker: (Question, Paragraph, Label).
    Uses TF-IDF to find hard negatives.
    """
    processor = TextProcessor()
    data_records = []

    # Limit sample size if specified
    if sample_size is not None and len(metadata_df) > sample_size:
        metadata_df = metadata_df.sample(n=sample_size, random_state=Config.SEED)

    with open(source_file, "rb") as f:
        for _, row in metadata_df.iterrows():
            f.seek(row["byte_offset"])
            line = f.readline()
            if not line:
                continue

            try:
                entry = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue

            question_text = entry["question_text"]
            document_text = entry["document_text"]
            annotations = entry.get("annotations", [])

            # Identify Ground Truth Long Answer Span
            # We look for the first valid long answer annotation
            target_start = -1
            target_end = -1
            for ann in annotations:
                la = ann["long_answer"]
                if la["start_token"] != -1:
                    target_start = la["start_token"]
                    target_end = la["end_token"]
                    break

            # Segment document into paragraphs
            # segment_document returns list of strings. We need to map these back to global token indices
            # to check against ground truth.
            # Since TextProcessor.tokenize splits on whitespace, we can track cumulative length.
            candidates_text = processor.segment_document(document_text)

            # Re-tokenize to get lengths and map to vocab
            q_indices = vocab.transform(question_text, max_len=Config.MAX_Q_LEN)

            candidate_objs = []
            current_global_idx = 0

            for cand_text in candidates_text:
                cand_tokens = processor.tokenize(cand_text)
                cand_len = len(cand_tokens)
                cand_global_start = current_global_idx
                cand_global_end = current_global_idx + cand_len

                # Check if this candidate matches the target
                # A simple heuristic: if the candidate range exactly matches the annotation
                is_positive = False
                if target_start != -1:
                    # Allow some flexibility or exact match?
                    # The annotation typically aligns with HTML tags, so exact match usually works
                    # if segmentation logic aligns.
                    # Here we check for significant overlap or exact match.
                    if (
                        cand_global_start == target_start
                        and cand_global_end == target_end
                    ):
                        is_positive = True

                candidate_objs.append(
                    {
                        "text": cand_text,
                        "indices": vocab.transform(
                            cand_text, max_len=Config.MAX_DOC_LEN
                        ),
                        "is_positive": is_positive,
                    }
                )

                current_global_idx += cand_len

            # If we found a positive candidate, we can generate training data
            positives = [c for c in candidate_objs if c["is_positive"]]
            negatives = [c for c in candidate_objs if not c["is_positive"]]

            if not positives:
                continue

            # For each positive, add it to dataset
            for pos in positives:
                data_records.append(
                    {"q_indices": q_indices, "p_indices": pos["indices"], "label": 1}
                )

            # Hard Negative Mining via TF-IDF
            if negatives:
                corpus = [question_text] + [n["text"] for n in negatives]
                # Handle empty text cases
                if any(len(t.strip()) == 0 for t in corpus):
                    continue

                try:
                    vectorizer = TfidfVectorizer().fit_transform(corpus)
                    vectors = vectorizer.toarray()
                    q_vec = vectors[0].reshape(1, -1)
                    cand_vecs = vectors[1:]

                    similarities = cosine_similarity(q_vec, cand_vecs)[0]

                    # Get indices of top k negatives
                    # k = len(positives) to balance, or fixed number
                    num_neg = min(
                        len(negatives), 2 * len(positives)
                    )  # 2 negatives per positive
                    hard_neg_indices = np.argsort(similarities)[-num_neg:]

                    for idx in hard_neg_indices:
                        data_records.append(
                            {
                                "q_indices": q_indices,
                                "p_indices": negatives[idx]["indices"],
                                "label": 0,
                            }
                        )
                except ValueError:
                    # Skip if vocabulary empty or other tfidf issues
                    continue

    return pd.DataFrame(data_records)


def _process_reader_data(metadata_df, vocab, source_file, sample_size=None):
    """
    Generates training data for the reader: (Question, Positive Paragraph, Start, End).
    Only uses samples with short answers.
    """
    processor = TextProcessor()
    data_records = []

    if sample_size is not None and len(metadata_df) > sample_size:
        metadata_df = metadata_df.sample(n=sample_size, random_state=Config.SEED)

    with open(source_file, "rb") as f:
        for _, row in metadata_df.iterrows():
            # Filter based on metadata flag if available
            if "has_short_answer" in row and not row["has_short_answer"]:
                continue

            f.seek(row["byte_offset"])
            line = f.readline()
            if not line:
                continue

            try:
                entry = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue

            annotations = entry.get("annotations", [])
            short_ans_start = -1
            short_ans_end = -1

            # Find first valid short answer
            for ann in annotations:
                if ann["short_answers"]:
                    short_ans_start = ann["short_answers"][0]["start_token"]
                    short_ans_end = ann["short_answers"][0]["end_token"]
                    break
                elif ann["yes_no_answer"] != "NONE":
                    # For YES/NO, we often point to the long answer span or specific tokens.
                    # Simplified strategy: Use the long answer span as the target for reader context,
                    # but reader predicts specific tokens.
                    # If strictly YES/NO without span, we might skip for span extraction training
                    # or point to CLS token. Here we skip strictly YES/NO without span.
                    pass

            if short_ans_start == -1:
                continue

            question_text = entry["question_text"]
            document_text = entry["document_text"]

            # Segment and find the paragraph containing the short answer
            candidates_text = processor.segment_document(document_text)

            current_global_idx = 0
            q_indices = vocab.transform(question_text, max_len=Config.MAX_Q_LEN)

            for cand_text in candidates_text:
                cand_tokens = processor.tokenize(cand_text)
                cand_len = len(cand_tokens)
                cand_global_start = current_global_idx
                cand_global_end = current_global_idx + cand_len

                # Check if short answer is fully contained in this paragraph
                if (
                    short_ans_start >= cand_global_start
                    and short_ans_end <= cand_global_end
                ):
                    # Calculate relative indices
                    rel_start = short_ans_start - cand_global_start
                    rel_end = (
                        short_ans_end - cand_global_start
                    )  # exclusive in annotation, but we usually want inclusive or exclusive consistent.
                    # NQ annotations are [start, end).
                    # Let's use inclusive for last token index for classification.
                    # So end index is rel_end - 1

                    final_end = rel_end - 1

                    # Safety check
                    if final_end < cand_len:
                        p_indices = vocab.transform(
                            cand_text, max_len=Config.MAX_DOC_LEN
                        )

                        # Adjust indices if truncation happens
                        if (
                            rel_start < Config.MAX_DOC_LEN
                            and final_end < Config.MAX_DOC_LEN
                        ):
                            data_records.append(
                                {
                                    "q_indices": q_indices,
                                    "p_indices": p_indices,
                                    "start_token": rel_start,
                                    "end_token": final_end,
                                }
                            )
                    break  # Found the paragraph

                current_global_idx += cand_len

    return pd.DataFrame(data_records)


def create_ranker_data(metadata_path, vocab, load_cached_data=True):
    """Orchestrates creation/loading of ranker dataset."""
    cache_path = (
        Config.RANKER_TRAIN_DATA if "train" in metadata_path else Config.RANKER_VAL_DATA
    )

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached ranker data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing ranker data from {metadata_path}...")
    df_meta = pd.read_csv(metadata_path)

    # Determine source file based on metadata
    # Assuming metadata contains 'file_path' column which points to filename in INPUT_DIR
    # We process the whole metadata DF, which might mix files if we combined them,
    # but usually train/val come from train file.
    # To be safe, we group by file_path.

    all_data = []
    for file_name, group in df_meta.groupby("file_path"):
        full_path = os.path.join(Config.INPUT_DIR, file_name)
        sample_size = Config.SAMPLE_SIZE if Config.DEBUG else None
        df_chunk = _process_ranker_data(group, vocab, full_path, sample_size)
        all_data.append(df_chunk)

    final_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    final_df.to_parquet(cache_path, index=False)
    print(f"Ranker data saved to {cache_path}. Size: {len(final_df)}")

    return final_df


def create_reader_data(metadata_path, vocab, load_cached_data=True):
    """Orchestrates creation/loading of reader dataset."""
    cache_path = (
        Config.READER_TRAIN_DATA if "train" in metadata_path else Config.READER_VAL_DATA
    )

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached reader data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing reader data from {metadata_path}...")
    df_meta = pd.read_csv(metadata_path)

    all_data = []
    for file_name, group in df_meta.groupby("file_path"):
        full_path = os.path.join(Config.INPUT_DIR, file_name)
        sample_size = Config.SAMPLE_SIZE if Config.DEBUG else None
        df_chunk = _process_reader_data(group, vocab, full_path, sample_size)
        all_data.append(df_chunk)

    final_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    final_df.to_parquet(cache_path, index=False)
    print(f"Reader data saved to {cache_path}. Size: {len(final_df)}")

    return final_df


def create_inference_data(metadata_path, vocab):
    """
    Creates dataset for inference (Test set).
    Returns a list of dicts: {example_id, q_indices, candidates: [{text, indices}]}
    Does not cache because inference is usually one-off or handled differently.
    """
    processor = TextProcessor()
    inference_data = []

    df_meta = pd.read_csv(metadata_path)

    # Process by file
    for file_name, group in df_meta.groupby("file_path"):
        full_path = os.path.join(Config.INPUT_DIR, file_name)

        with open(full_path, "rb") as f:
            for _, row in group.iterrows():
                f.seek(row["byte_offset"])
                line = f.readline()
                if not line:
                    continue

                try:
                    entry = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                example_id = entry["example_id"]
                question_text = entry["question_text"]
                document_text = entry["document_text"]

                q_indices = vocab.transform(question_text, max_len=Config.MAX_Q_LEN)
                candidates_text = processor.segment_document(document_text)

                candidates_processed = []
                current_global_idx = 0

                for cand_text in candidates_text:
                    cand_indices = vocab.transform(
                        cand_text, max_len=Config.MAX_DOC_LEN
                    )
                    cand_len = len(processor.tokenize(cand_text))

                    candidates_processed.append(
                        {
                            "text": cand_text,
                            "indices": cand_indices,
                            "global_start": current_global_idx,
                            "global_end": current_global_idx + cand_len,
                        }
                    )
                    current_global_idx += cand_len

                inference_data.append(
                    {
                        "example_id": example_id,
                        "q_indices": q_indices,
                        "candidates": candidates_processed,
                    }
                )

    return inference_data


# --------------------------------------------------------------------------
# Main Loader Interface
# --------------------------------------------------------------------------


def get_data_loaders(vocab, load_cached_data=True):
    """
    Returns (ranker_train_loader, ranker_val_loader, reader_train_loader, reader_val_loader).
    """
    # Ranker Data
    ranker_train_df = create_ranker_data(Config.TRAIN_METADATA, vocab, load_cached_data)
    ranker_val_df = create_ranker_data(Config.VAL_METADATA, vocab, load_cached_data)

    ranker_train_ds = RankerDataset(ranker_train_df)
    ranker_val_ds = RankerDataset(ranker_val_df)

    ranker_train_loader = DataLoader(
        ranker_train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_ranker,
        num_workers=2,
    )
    ranker_val_loader = DataLoader(
        ranker_val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_ranker,
        num_workers=2,
    )

    # Reader Data
    reader_train_df = create_reader_data(Config.TRAIN_METADATA, vocab, load_cached_data)
    reader_val_df = create_reader_data(Config.VAL_METADATA, vocab, load_cached_data)

    reader_train_ds = ReaderDataset(reader_train_df)
    reader_val_ds = ReaderDataset(reader_val_df)

    reader_train_loader = DataLoader(
        reader_train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_reader,
        num_workers=2,
    )
    reader_val_loader = DataLoader(
        reader_val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_reader,
        num_workers=2,
    )

    return (
        ranker_train_loader,
        ranker_val_loader,
        reader_train_loader,
        reader_val_loader,
    )
