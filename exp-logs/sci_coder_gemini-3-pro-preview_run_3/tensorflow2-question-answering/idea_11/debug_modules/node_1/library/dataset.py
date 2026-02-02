import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from library.config import Config
from library.preprocessing import HTMLParser


class NQDataset(Dataset):
    """
    PyTorch Dataset wrapper for the NQ data.
    Handles deserialization of complex fields if stored as JSON strings.
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx].to_dict()

        # Deserialize candidates if they are stored as JSON strings (common in Parquet)
        if "candidates" in row and isinstance(row["candidates"], str):
            row["candidates"] = json.loads(row["candidates"])

        return row


def ranker_collate_fn(batch):
    """
    Collates a batch of ranker training triplets.
    Returns padded tensors for query, positive, and negative sequences.
    """
    q_seqs = [torch.tensor(x["q_ids"], dtype=torch.long) for x in batch]
    pos_seqs = [torch.tensor(x["pos_ids"], dtype=torch.long) for x in batch]
    neg_seqs = [torch.tensor(x["neg_ids"], dtype=torch.long) for x in batch]

    q_padded = torch.nn.utils.rnn.pad_sequence(
        q_seqs, batch_first=True, padding_value=0
    )
    pos_padded = torch.nn.utils.rnn.pad_sequence(
        pos_seqs, batch_first=True, padding_value=0
    )
    neg_padded = torch.nn.utils.rnn.pad_sequence(
        neg_seqs, batch_first=True, padding_value=0
    )

    return q_padded, pos_padded, neg_padded


def reader_collate_fn(batch):
    """
    Collates a batch of reader training samples.
    Returns padded input sequences and start/end target tensors.
    """
    input_seqs = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    starts = torch.tensor([x["start_token"] for x in batch], dtype=torch.long)
    ends = torch.tensor([x["end_token"] for x in batch], dtype=torch.long)

    input_padded = torch.nn.utils.rnn.pad_sequence(
        input_seqs, batch_first=True, padding_value=0
    )

    return input_padded, starts, ends


def get_ranker_dataset(split, tokenizer, load_cached_data=True, debug=Config.DEBUG):
    """
    Generates or loads the dataset for the Ranker model.
    Constructs triplets: (Question, Positive Paragraph, Hard Negative Paragraph).
    """
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
        out_path = Config.RANKER_TRAIN_DATA
    elif split == "val":
        meta_path = Config.VAL_METADATA_PATH
        out_path = Config.RANKER_VAL_DATA
    else:
        raise ValueError("Split must be 'train' or 'val'")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(out_path):
        print(f"Loading cached ranker data from {out_path}")
        try:
            df = pd.read_parquet(out_path)
            return NQDataset(df)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing ranker data for split: {split}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    meta_df = pd.read_csv(meta_path)

    # Debugging subset
    if debug:
        subset_size = (
            Config.TRAIN_SUBSET_SIZE if split == "train" else Config.VAL_SUBSET_SIZE
        )
        if subset_size:
            meta_df = meta_df.head(subset_size)

    # Filter for examples that actually have a long answer
    meta_df = meta_df[meta_df["has_long_answer"] == True]

    parser = HTMLParser()
    data_list = []

    with open(Config.TRAIN_DATA_FILE, "rb") as f:
        for _, row in meta_df.iterrows():
            f.seek(row["byte_offset"])
            line = f.readline()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            question_text = record["question_text"]
            doc_text = record["document_text"]

            # Identify Ground Truth Long Answer Span
            gt_start = -1
            gt_end = -1
            for ann in record["annotations"]:
                if ann["long_answer"]["start_token"] != -1:
                    gt_start = ann["long_answer"]["start_token"]
                    gt_end = ann["long_answer"]["end_token"]
                    break

            if gt_start == -1:
                continue

            # Segment document into candidates
            candidates = parser.segment(doc_text)

            # Find Positive Candidate
            pos_idx = -1
            for i, cand in enumerate(candidates):
                # Check if GT span is contained within candidate span
                if cand["start_token"] <= gt_start and cand["end_token"] >= gt_end:
                    pos_idx = i
                    break

            if pos_idx == -1:
                continue

            # TF-IDF Hard Negative Mining
            # We build a mini-corpus of [Question, Cand_0, Cand_1, ...]
            corpus = [question_text] + [c["text"] for c in candidates]

            try:
                # Fit TF-IDF on this single document context
                vectorizer = TfidfVectorizer().fit_transform(corpus)
                vectors = vectorizer.toarray()

                # Calculate similarity between Question (index 0) and all Candidates (indices 1..)
                # vectors[0:1] is (1, dim), vectors[1:] is (num_cand, dim)
                cosine_sims = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

                # Mask the positive candidate so we don't select it as negative
                cosine_sims[pos_idx] = -1.0

                # Select the candidate with highest similarity to question among negatives
                neg_idx = np.argmax(cosine_sims)

                # Tokenize
                q_ids = tokenizer.texts_to_sequences([question_text])[0][
                    : Config.MAX_Q_LEN
                ]
                pos_ids = tokenizer.texts_to_sequences([candidates[pos_idx]["text"]])[
                    0
                ][: Config.MAX_CTX_LEN]
                neg_ids = tokenizer.texts_to_sequences([candidates[neg_idx]["text"]])[
                    0
                ][: Config.MAX_CTX_LEN]

                # Ensure sequences are not empty
                if not q_ids or not pos_ids or not neg_ids:
                    continue

                data_list.append(
                    {
                        "example_id": row["example_id"],
                        "q_ids": q_ids,
                        "pos_ids": pos_ids,
                        "neg_ids": neg_ids,
                    }
                )

            except ValueError:
                # Can happen if vocab is empty or document is weird
                continue

    # 3. Save and Return
    df = pd.DataFrame(data_list)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved ranker data to {out_path}. Count: {len(df)}")

    return NQDataset(df)


def get_reader_dataset(split, tokenizer, load_cached_data=True, debug=Config.DEBUG):
    """
    Generates or loads the dataset for the Reader model.
    Constructs samples: (Concatenated Q+P, Start Index, End Index).
    """
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
        out_path = Config.READER_TRAIN_DATA
    elif split == "val":
        meta_path = Config.VAL_METADATA_PATH
        out_path = Config.READER_VAL_DATA
    else:
        raise ValueError("Split must be 'train' or 'val'")

    if load_cached_data and os.path.exists(out_path):
        print(f"Loading cached reader data from {out_path}")
        try:
            df = pd.read_parquet(out_path)
            return NQDataset(df)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Processing reader data for split: {split}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    meta_df = pd.read_csv(meta_path)

    if debug:
        subset_size = (
            Config.TRAIN_SUBSET_SIZE if split == "train" else Config.VAL_SUBSET_SIZE
        )
        if subset_size:
            meta_df = meta_df.head(subset_size)

    # Filter for examples with Short Answers
    meta_df = meta_df[meta_df["has_short_answer"] == True]

    parser = HTMLParser()
    data_list = []

    with open(Config.TRAIN_DATA_FILE, "rb") as f:
        for _, row in meta_df.iterrows():
            f.seek(row["byte_offset"])
            line = f.readline()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            question_text = record["question_text"]
            doc_text = record["document_text"]

            # Find the annotation with a short answer
            target_ann = None
            for ann in record["annotations"]:
                if ann["short_answers"]:
                    target_ann = ann
                    break

            if not target_ann:
                continue

            # Use the first short answer span
            sa = target_ann["short_answers"][0]
            s_start = sa["start_token"]
            s_end = sa["end_token"]

            # Find the candidate paragraph containing this short answer
            candidates = parser.segment(doc_text)
            cand_idx = -1
            for i, cand in enumerate(candidates):
                if cand["start_token"] <= s_start and cand["end_token"] >= s_end:
                    cand_idx = i
                    break

            if cand_idx == -1:
                continue

            candidate = candidates[cand_idx]
            cand_text = candidate["text"]

            # Tokenize Question
            q_ids = tokenizer.texts_to_sequences([question_text])[0]
            if len(q_ids) > Config.MAX_Q_LEN:
                q_ids = q_ids[: Config.MAX_Q_LEN]

            # Calculate indices relative to the start of the candidate paragraph
            # Note: Tokenizer splits by whitespace, consistent with HTMLParser/NQ data
            rel_start = s_start - candidate["start_token"]
            rel_end = s_end - candidate["start_token"]

            # Tokenize Paragraph
            p_ids = tokenizer.texts_to_sequences([cand_text])[0]

            # Check if answer is within the truncation window
            # rel_end is exclusive index. If rel_end > MAX, the answer is cut off.
            if rel_end > Config.MAX_CTX_LEN:
                continue

            p_ids = p_ids[: Config.MAX_CTX_LEN]

            # Concatenate Question and Paragraph
            input_ids = q_ids + p_ids

            # Adjust targets for the concatenated sequence
            # Start index shifts by length of Q
            # End index shifts by length of Q.
            # We convert end to inclusive index for classification (standard PyTorch CE target)
            final_start = len(q_ids) + rel_start
            final_end = len(q_ids) + rel_end - 1

            # Validation check
            if final_end >= len(input_ids):
                continue

            data_list.append(
                {
                    "example_id": row["example_id"],
                    "input_ids": input_ids,
                    "start_token": final_start,
                    "end_token": final_end,
                }
            )

    df = pd.DataFrame(data_list)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved reader data to {out_path}. Count: {len(df)}")

    return NQDataset(df)


def get_test_dataset(tokenizer, load_cached_data=True):
    """
    Generates or loads the dataset for Inference (Test set).
    Returns samples containing Question and List of Candidates.
    """
    out_path = Config.TEST_FEATURES_PATH
    meta_path = Config.TEST_METADATA_PATH

    if load_cached_data and os.path.exists(out_path):
        print(f"Loading cached test features from {out_path}")
        try:
            df = pd.read_parquet(out_path)
            return NQDataset(df)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print("Processing test data...")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    meta_df = pd.read_csv(meta_path)
    parser = HTMLParser()
    data_list = []

    with open(Config.TEST_DATA_FILE, "rb") as f:
        for _, row in meta_df.iterrows():
            f.seek(row["byte_offset"])
            line = f.readline()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            question_text = record["question_text"]
            doc_text = record["document_text"]

            # Tokenize Question
            q_ids = tokenizer.texts_to_sequences([question_text])[0][: Config.MAX_Q_LEN]

            # Segment Candidates
            candidates = parser.segment(doc_text)

            # Limit number of candidates to process per doc to save time/memory
            candidates = candidates[: Config.MAX_TEST_CANDIDATES]

            processed_candidates = []
            for cand in candidates:
                c_text = cand["text"]
                c_ids = tokenizer.texts_to_sequences([c_text])[0][: Config.MAX_CTX_LEN]

                processed_candidates.append(
                    {
                        "text": c_text,
                        "token_ids": c_ids,
                        "start_token": cand["start_token"],
                        "end_token": cand["end_token"],
                    }
                )

            # Serialize candidates to JSON string for storage in Parquet
            data_list.append(
                {
                    "example_id": row["example_id"],
                    "q_ids": q_ids,
                    "candidates": json.dumps(processed_candidates),
                }
            )

    df = pd.DataFrame(data_list)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved test features to {out_path}. Count: {len(df)}")

    return NQDataset(df)
