import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from library import config
from library import text_utils

# Set fixed seeds for reproducibility
random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)


class TFIDFNegativeSampler:
    """
    Selects hard negatives based on TF-IDF similarity between the question
    and candidate paragraphs.
    """

    def __init__(self):
        pass

    def sample(self, question_text, candidates, positive_idx):
        """
        Selects a negative candidate that is similar to the question but is not the positive one.

        Args:
            question_text (str): The query text.
            candidates (list): List of candidate dictionaries (from segment_document).
            positive_idx (int): Index of the positive candidate in the list.

        Returns:
            dict: The selected negative candidate dictionary, or None if no suitable negative found.
        """
        if len(candidates) < 2:
            return None

        # Extract text from candidates
        candidate_texts = [c["text"] for c in candidates]

        # Corpus = Question + Candidates
        corpus = [question_text] + candidate_texts

        try:
            vectorizer = TfidfVectorizer(
                stop_words="english", token_pattern=r"(?u)\b\w+\b"
            )
            tfidf_matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            # Handle cases with empty vocabulary (e.g. only stop words)
            return random.choice(
                [c for i, c in enumerate(candidates) if i != positive_idx]
            )

        # Compute cosine similarity between Question (index 0) and all candidates (indices 1..)
        # shape: (1, n_candidates)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # We want the candidate with highest similarity that is NOT the positive_idx
        # Mask the positive index by setting its score to -1
        similarities[positive_idx] = -1.0

        # Get index of best negative
        best_neg_idx = np.argmax(similarities)

        # If the similarity is 0, it's basically a random negative, which is fine
        return candidates[best_neg_idx]


def process_ranker_data(metadata_path, vocab, load_cached_data=True, is_train=True):
    """
    Generates or loads training data for the Ranker model.
    Format: Question, Positive Paragraph, Negative Paragraph.
    """
    # Determine cache path based on split
    if is_train:
        cache_path = config.RANKER_TRAIN_CACHE
    else:
        cache_path = config.RANKER_VAL_CACHE

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading ranker data from {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Processing ranker data from {metadata_path}...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    # Debugging limit
    if config.DEBUG_SAMPLE_SIZE:
        df_meta = df_meta.iloc[: config.DEBUG_SAMPLE_SIZE]

    sampler = TFIDFNegativeSampler()
    data_records = []

    # Open the raw data file once
    # Note: We assume all data comes from TRAIN_DATA_FILE based on metadata logic
    # even for validation split (as it was split from train).
    source_file = config.TRAIN_DATA_FILE

    with open(source_file, "rb") as f:
        for _, row in df_meta.iterrows():
            # Skip if no long answer (ranker needs positive example)
            if not row.get("has_long_answer", False):
                continue

            offset = row["byte_offset"]
            f.seek(offset)
            line = f.readline()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            question_text = record["question_text"]
            doc_tokens = text_utils.tokenize(record["document_text"])

            # Get Ground Truth Long Answer info
            # The annotation gives start/end token indices in the document
            # We take the first annotation that has a long answer
            la_start = -1
            for ann in record["annotations"]:
                if ann["long_answer"]["start_token"] != -1:
                    la_start = ann["long_answer"]["start_token"]
                    break

            if la_start == -1:
                continue

            # Segment document
            candidates = text_utils.segment_document(doc_tokens)

            # Find which candidate matches the ground truth start token
            pos_candidate_idx = -1
            for i, cand in enumerate(candidates):
                if cand["start_token"] == la_start:
                    pos_candidate_idx = i
                    break

            if pos_candidate_idx == -1:
                # Ground truth span might not align perfectly with heuristic segmentation
                # For simplicity, skip these or find closest overlap. Skipping for now.
                continue

            pos_candidate = candidates[pos_candidate_idx]

            # Sample Negative
            neg_candidate = sampler.sample(question_text, candidates, pos_candidate_idx)
            if neg_candidate is None:
                continue

            # Convert to indices
            q_indices = text_utils.text_to_indices(
                question_text, vocab, config.MAX_Q_LEN
            )
            pos_indices = text_utils.text_to_indices(
                pos_candidate["text"], vocab, config.MAX_DOC_LEN
            )
            neg_indices = text_utils.text_to_indices(
                neg_candidate["text"], vocab, config.MAX_DOC_LEN
            )

            data_records.append(
                {
                    "q_indices": q_indices,
                    "pos_indices": pos_indices,
                    "neg_indices": neg_indices,
                }
            )

    df_processed = pd.DataFrame(data_records)

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_processed.to_parquet(cache_path, index=False)
    print(f"Saved {len(df_processed)} ranker samples to {cache_path}")

    return df_processed


def process_reader_data(metadata_path, vocab, load_cached_data=True, is_train=True):
    """
    Generates or loads training data for the Reader model.
    Format: Question, Paragraph, Start Token Index, End Token Index.
    """
    if is_train:
        cache_path = config.READER_TRAIN_CACHE
    else:
        cache_path = config.READER_VAL_CACHE

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading reader data from {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Processing reader data from {metadata_path}...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    if config.DEBUG_SAMPLE_SIZE:
        df_meta = df_meta.iloc[: config.DEBUG_SAMPLE_SIZE]

    data_records = []
    source_file = config.TRAIN_DATA_FILE

    with open(source_file, "rb") as f:
        for _, row in df_meta.iterrows():
            # Reader needs short answers
            if not row.get("has_short_answer", False):
                continue

            offset = row["byte_offset"]
            f.seek(offset)
            line = f.readline()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            question_text = record["question_text"]
            doc_tokens = text_utils.tokenize(record["document_text"])

            # Find valid short answer annotation
            target_ann = None
            for ann in record["annotations"]:
                if ann["short_answers"] or ann["yes_no_answer"] != "NONE":
                    # We prioritize span answers for this reader implementation
                    # Handling YES/NO is complex, we focus on spans here or treat YES/NO as special tokens if architecture allowed.
                    # For this task, we extract spans.
                    if ann["short_answers"]:
                        target_ann = ann
                        break

            if not target_ann:
                continue

            # Get the long answer (paragraph) containing the short answer
            la_start = target_ann["long_answer"]["start_token"]
            la_end = target_ann["long_answer"]["end_token"]

            # Extract paragraph text
            if la_start == -1 or la_end == -1:
                continue

            para_tokens = doc_tokens[la_start:la_end]
            para_text = " ".join(para_tokens)

            # Get short answer span (first one if multiple)
            sa_span = target_ann["short_answers"][0]
            sa_start_global = sa_span["start_token"]
            sa_end_global = sa_span["end_token"]

            # Calculate relative indices
            rel_start = sa_start_global - la_start
            rel_end = (
                sa_end_global - la_start - 1
            )  # Inclusive index for model target usually

            # Validate indices
            if (
                rel_start < 0
                or rel_end >= len(para_tokens)
                or rel_end >= config.MAX_DOC_LEN
            ):
                continue

            q_indices = text_utils.text_to_indices(
                question_text, vocab, config.MAX_Q_LEN
            )
            para_indices = text_utils.text_to_indices(
                para_text, vocab, config.MAX_DOC_LEN
            )

            data_records.append(
                {
                    "q_indices": q_indices,
                    "para_indices": para_indices,
                    "start_idx": rel_start,
                    "end_idx": rel_end,
                }
            )

    df_processed = pd.DataFrame(data_records)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_processed.to_parquet(cache_path, index=False)
    print(f"Saved {len(df_processed)} reader samples to {cache_path}")

    return df_processed


class RankerDataset(Dataset):
    def __init__(self, data_df):
        """
        Args:
            data_df (pd.DataFrame): DataFrame with columns 'q_indices', 'pos_indices', 'neg_indices'.
                                    Each element is a list of integers.
        """
        self.data = data_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        return (
            torch.tensor(row["q_indices"], dtype=torch.long),
            torch.tensor(row["pos_indices"], dtype=torch.long),
            torch.tensor(row["neg_indices"], dtype=torch.long),
        )


class ReaderDataset(Dataset):
    def __init__(self, data_df):
        """
        Args:
            data_df (pd.DataFrame): DataFrame with columns 'q_indices', 'para_indices', 'start_idx', 'end_idx'.
        """
        self.data = data_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        return (
            torch.tensor(row["q_indices"], dtype=torch.long),
            torch.tensor(row["para_indices"], dtype=torch.long),
            torch.tensor(row["start_idx"], dtype=torch.long),
            torch.tensor(row["end_idx"], dtype=torch.long),
        )


def process_test_ranker_inputs(metadata_path, vocab, load_cached_data=True):
    """
    Pre-processes test data for the ranking phase.
    Since we don't have labels, we generate (Question, Candidate) pairs for ALL candidates.
    Returns a DataFrame where each row is a candidate to be scored.
    """
    cache_path = config.RANKER_TEST_INPUTS_CACHE

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading test ranker inputs from {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass

    print(f"Processing test ranker inputs from {metadata_path}...")
    df_meta = pd.read_csv(metadata_path)

    # In inference, we usually process all test data, but respect debug flag if needed
    if config.DEBUG_SAMPLE_SIZE and len(df_meta) > config.DEBUG_SAMPLE_SIZE:
        df_meta = df_meta.iloc[: config.DEBUG_SAMPLE_SIZE]

    data_records = []
    source_file = config.TEST_DATA_FILE

    with open(source_file, "rb") as f:
        for _, row in df_meta.iterrows():
            offset = row["byte_offset"]
            # Cite debug_lesson_1: Do not use row["example_id"] from metadata
            f.seek(offset)
            line = f.readline()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Cite debug_lesson_1 & debug_lesson_4: Use raw ID and sanitize
            raw_id = str(record["example_id"])
            example_id = raw_id.replace(",", "").replace("\n", "").strip()

            question_text = record["question_text"]
            doc_tokens = text_utils.tokenize(record["document_text"])
            candidates = text_utils.segment_document(doc_tokens)

            q_indices = text_utils.text_to_indices(
                question_text, vocab, config.MAX_Q_LEN
            )

            for i, cand in enumerate(candidates):
                # Filter very short candidates to save compute
                if len(cand["text"].split()) < config.MIN_DOC_LEN:
                    continue

                cand_indices = text_utils.text_to_indices(
                    cand["text"], vocab, config.MAX_DOC_LEN
                )

                data_records.append(
                    {
                        "example_id": str(example_id),
                        "candidate_idx": i,
                        "q_indices": q_indices,
                        "cand_indices": cand_indices,
                        "cand_text": cand[
                            "text"
                        ],  # Needed for final submission text extraction
                        "start_token": cand["start_token"],  # Needed for index mapping
                        "end_token": cand["end_token"],
                    }
                )

    df_processed = pd.DataFrame(data_records)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_processed.to_parquet(cache_path, index=False)
    print(f"Saved {len(df_processed)} test candidates to {cache_path}")

    return df_processed
