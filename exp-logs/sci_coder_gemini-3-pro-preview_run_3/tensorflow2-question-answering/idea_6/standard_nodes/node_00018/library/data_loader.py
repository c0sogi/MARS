import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from library.config import PathConfig, ModelConfig, TrainingConfig
from library.utils import set_seed, setup_logger, parse_html, HTML_TAGS

# Initialize logger
logger = setup_logger("data_loader")


class NQProcessor:
    """
    Handles raw data processing, candidate generation, and label alignment.
    Implements caching mechanisms for processed datasets.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _read_json_line(self, file_path, offset):
        """Reads a single line from a JSONL file at a specific byte offset."""
        with open(file_path, "rb") as f:
            f.seek(offset)
            line = f.readline()
            if not line:
                return None
            return json.loads(line.decode("utf-8"))

    def _get_hard_negative(self, question, candidates, positive_idx):
        """
        Selects a hard negative candidate using TF-IDF similarity.
        Returns the text of the candidate most similar to the question that isn't the positive one.
        """
        if len(candidates) < 2:
            return None

        texts = [c["text"] for c in candidates]
        # Add question to the set to compute similarity
        corpus = [question] + texts

        try:
            vectorizer = TfidfVectorizer().fit_transform(corpus)
            vectors = vectorizer.toarray()

            # Cosine similarity between question (index 0) and all candidates (indices 1 to N)
            question_vec = vectors[0].reshape(1, -1)
            candidate_vecs = vectors[1:]

            similarities = cosine_similarity(question_vec, candidate_vecs)[0]

            # Mask the positive candidate
            similarities[positive_idx] = -1.0

            # Get index of highest similarity
            hard_negative_idx = np.argmax(similarities)
            return candidates[hard_negative_idx]["text"]

        except ValueError:
            # Fallback for empty vocabulary or other sklearn errors
            # Pick a random negative
            indices = list(range(len(candidates)))
            indices.remove(positive_idx)
            return candidates[np.random.choice(indices)]["text"]

    def process_ranker_data(
        self,
        metadata_path,
        raw_data_path,
        output_path,
        load_cached_data=True,
        subset_size=None,
    ):
        """
        Generates triplets (Question, Positive Context, Negative Context) for the Ranker.
        """
        if load_cached_data and os.path.exists(output_path):
            logger.info(f"Loading cached ranker data from {output_path}")
            return pd.read_parquet(output_path)

        logger.info(f"Processing ranker data from {raw_data_path}...")
        metadata = pd.read_csv(metadata_path)

        # Filter for examples that have a long answer
        if "has_long_answer" in metadata.columns:
            metadata = metadata[metadata["has_long_answer"] == True]

        if subset_size:
            metadata = metadata.head(subset_size)

        data = []

        for _, row in tqdm(
            metadata.iterrows(), total=len(metadata), desc="Processing Ranker"
        ):
            record = self._read_json_line(raw_data_path, row["byte_offset"])
            if not record:
                continue

            question = record["question_text"]
            doc_text = record["document_text"]
            candidates = parse_html(doc_text)

            # Find positive candidate index
            # The annotation gives the long answer index among the *original* candidates in the JSON.
            # However, parse_html might filter some. We need to match by token indices.
            # Actually, simplified-nq provides long_answer_candidates in the JSON.
            # But we are using parse_html to clean text. We need to map the annotation to our parsed candidates.

            # Ground truth long answer
            annotation = record["annotations"][0]["long_answer"]
            la_start = annotation["start_token"]
            la_end = annotation["end_token"]

            positive_text = None
            positive_idx = -1

            # Match annotation to parsed candidates
            for i, cand in enumerate(candidates):
                # We check for overlap or exact match.
                # The NQ long answer is usually one of the top-level candidates.
                if cand["start_token"] == la_start and cand["end_token"] == la_end:
                    positive_text = cand["text"]
                    positive_idx = i
                    break

            if positive_text is None:
                continue

            # Get Hard Negative
            negative_text = self._get_hard_negative(question, candidates, positive_idx)

            if negative_text:
                data.append(
                    {
                        "question": question,
                        "positive": positive_text,
                        "negative": negative_text,
                    }
                )

        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path)
        return df

    def process_reader_data(
        self,
        metadata_path,
        raw_data_path,
        output_path,
        load_cached_data=True,
        subset_size=None,
    ):
        """
        Generates (Question, Context, Start_Token, End_Token) for the Reader.
        Aligns whitespace-token spans to BERT sub-word tokens.
        """
        if load_cached_data and os.path.exists(output_path):
            logger.info(f"Loading cached reader data from {output_path}")
            return pd.read_parquet(output_path)

        logger.info(f"Processing reader data from {raw_data_path}...")
        metadata = pd.read_csv(metadata_path)

        # Filter for examples containing short answers
        if "has_short_answer" in metadata.columns:
            metadata = metadata[metadata["has_short_answer"] == True]

        if subset_size:
            metadata = metadata.head(subset_size)

        data = []

        for _, row in tqdm(
            metadata.iterrows(), total=len(metadata), desc="Processing Reader"
        ):
            record = self._read_json_line(raw_data_path, row["byte_offset"])
            if not record:
                continue

            question = record["question_text"]
            doc_text = record["document_text"]
            doc_tokens = doc_text.split()
            candidates = parse_html(doc_text)

            annotation = record["annotations"][0]
            short_answers = annotation["short_answers"]

            if not short_answers:
                continue

            # Take the first short answer
            sa_start_token = short_answers[0]["start_token"]
            sa_end_token = short_answers[0]["end_token"]

            # Find the candidate paragraph containing this short answer
            target_candidate = None
            for cand in candidates:
                if (
                    cand["start_token"] <= sa_start_token
                    and cand["end_token"] >= sa_end_token
                ):
                    target_candidate = cand
                    break

            if not target_candidate:
                continue

            context_text = target_candidate["text"]

            # Calculate character offsets within the cleaned context text
            # We need to reconstruct the text up to the answer start to find char offset
            # Since parse_html removes tags, we must be careful.
            # Strategy: Reconstruct the "clean" segment tokens and find the answer tokens within them.

            # Tokens of the candidate paragraph (cleaned)
            # Note: parse_html logic: " ".join(current_tokens)
            # We need to identify which tokens in the cleaned string correspond to the answer.

            # The indices in doc_tokens corresponding to the candidate
            cand_tokens_indices = []

            # Re-simulate parse logic briefly or map indices
            # parse_html returns start/end indices into the original doc_tokens list.
            # However, parse_html skips tags. We need to find the relative index of the answer
            # within the *non-tag* tokens of the candidate.

            # Extract tokens for the candidate from original doc
            raw_cand_tokens = doc_tokens[
                target_candidate["start_token"] : target_candidate["end_token"]
            ]

            # Filter tags to get the clean tokens list
            clean_cand_tokens = []
            clean_token_original_indices = []

            for i, t in enumerate(raw_cand_tokens):
                if t not in HTML_TAGS:
                    clean_cand_tokens.append(t)
                    # absolute index in doc
                    clean_token_original_indices.append(
                        target_candidate["start_token"] + i
                    )

            # Find where the answer starts/ends in this clean list
            try:
                rel_start = clean_token_original_indices.index(sa_start_token)
                # The end token in NQ is exclusive, so we look for end-1
                rel_end = clean_token_original_indices.index(sa_end_token - 1)
            except ValueError:
                # Token mismatch due to cleaning logic discrepancy
                continue

            # Now we have the answer as a span of tokens in `clean_cand_tokens`
            # We need to map this to character offsets in `context_text`
            # context_text is " ".join(clean_cand_tokens)

            # Calculate char start
            char_start = 0
            for k in range(rel_start):
                char_start += len(clean_cand_tokens[k]) + 1  # +1 for space

            # Calculate char end
            char_end = char_start
            for k in range(rel_start, rel_end + 1):
                char_end += len(clean_cand_tokens[k]) + 1
            char_end -= 1  # Remove trailing space

            # Tokenize with BERT tokenizer to get target labels
            # We tokenize Question + Context
            encodings = self.tokenizer(
                question,
                context_text,
                truncation="only_second",
                max_length=ModelConfig.MAX_CTX_LEN,
                stride=ModelConfig.DOC_STRIDE,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            # The answer might be in one of the overflow chunks (or none if truncated badly, though stride helps)
            # We try to find the answer in the first chunk that contains it fully

            found_span = False
            for encoding_idx, offsets in enumerate(encodings["offset_mapping"]):
                sequence_ids = encodings.sequence_ids(encoding_idx)

                # Find start and end of context in this sequence
                ctx_start_idx = 0
                while sequence_ids[ctx_start_idx] != 1:
                    ctx_start_idx += 1
                    if ctx_start_idx >= len(sequence_ids):
                        break

                if ctx_start_idx >= len(sequence_ids):
                    continue

                ctx_end_idx = len(sequence_ids) - 1
                while sequence_ids[ctx_end_idx] != 1:
                    ctx_end_idx -= 1

                # Check if char_start and char_end are within this chunk's context span
                # offsets[idx] returns (start_char, end_char) of the token

                # Boundary check
                if (
                    offsets[ctx_start_idx][0] > char_start
                    or offsets[ctx_end_idx][1] < char_end
                ):
                    continue

                # Find token start index
                idx = ctx_start_idx
                while idx <= ctx_end_idx and offsets[idx][0] <= char_start:
                    idx += 1
                start_token_idx = idx - 1

                # Find token end index
                idx = ctx_end_idx
                while idx >= ctx_start_idx and offsets[idx][1] >= char_end:
                    idx -= 1
                end_token_idx = idx + 1

                data.append(
                    {
                        "input_ids": encodings["input_ids"][encoding_idx],
                        "attention_mask": encodings["attention_mask"][encoding_idx],
                        "token_type_ids": (
                            encodings["token_type_ids"][encoding_idx]
                            if "token_type_ids" in encodings
                            else [0] * len(encodings["input_ids"][encoding_idx])
                        ),
                        "start_position": start_token_idx,
                        "end_position": end_token_idx,
                    }
                )
                found_span = True
                break  # Only take the first valid window

        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Lists in pandas dataframe can be tricky with parquet, ensure compatibility
        # Or just save as is, pyarrow handles lists usually
        df.to_parquet(output_path)
        return df


class RankerDataset(Dataset):
    def __init__(self, data_df, tokenizer, max_len=128):
        self.data = data_df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Tokenize Question
        q_enc = self.tokenizer(
            row["question"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize Positive
        pos_enc = self.tokenizer(
            row["positive"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize Negative
        neg_enc = self.tokenizer(
            row["negative"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "q_input_ids": q_enc["input_ids"].squeeze(0),
            "q_attention_mask": q_enc["attention_mask"].squeeze(0),
            "pos_input_ids": pos_enc["input_ids"].squeeze(0),
            "pos_attention_mask": pos_enc["attention_mask"].squeeze(0),
            "neg_input_ids": neg_enc["input_ids"].squeeze(0),
            "neg_attention_mask": neg_enc["attention_mask"].squeeze(0),
        }


class ReaderDataset(Dataset):
    def __init__(self, data_df):
        self.data = data_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Data is already tokenized and stored as lists in the dataframe
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(row["token_type_ids"], dtype=torch.long),
            "start_positions": torch.tensor(row["start_position"], dtype=torch.long),
            "end_positions": torch.tensor(row["end_position"], dtype=torch.long),
        }


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get dataloaders.
    Handles processing, caching, and dataset creation.
    """
    set_seed(TrainingConfig.SEED)

    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.MODEL_NAME)
    processor = NQProcessor(tokenizer)

    # --- Ranker Data ---
    ranker_train_df = processor.process_ranker_data(
        PathConfig.TRAIN_METADATA,
        PathConfig.TRAIN_FILE,
        PathConfig.RANKER_TRAIN_DATA,
        load_cached_data=load_cached_data,
        subset_size=TrainingConfig.SUBSET_SIZE,
    )

    ranker_val_df = processor.process_ranker_data(
        PathConfig.VAL_METADATA,
        PathConfig.TRAIN_FILE,
        PathConfig.RANKER_VAL_DATA,
        load_cached_data=load_cached_data,
        subset_size=TrainingConfig.SUBSET_SIZE,
    )

    ranker_train_ds = RankerDataset(ranker_train_df, tokenizer)
    ranker_val_ds = RankerDataset(ranker_val_df, tokenizer)

    ranker_train_loader = DataLoader(
        ranker_train_ds,
        batch_size=TrainingConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=TrainingConfig.NUM_WORKERS,
        pin_memory=True,
    )

    ranker_val_loader = DataLoader(
        ranker_val_ds,
        batch_size=TrainingConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=TrainingConfig.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Reader Data ---
    reader_train_df = processor.process_reader_data(
        PathConfig.TRAIN_METADATA,
        PathConfig.TRAIN_FILE,
        PathConfig.READER_TRAIN_DATA,
        load_cached_data=load_cached_data,
        subset_size=TrainingConfig.SUBSET_SIZE,
    )

    reader_val_df = processor.process_reader_data(
        PathConfig.VAL_METADATA,
        PathConfig.TRAIN_FILE,
        PathConfig.READER_VAL_DATA,
        load_cached_data=load_cached_data,
        subset_size=TrainingConfig.SUBSET_SIZE,
    )

    reader_train_ds = ReaderDataset(reader_train_df)
    reader_val_ds = ReaderDataset(reader_val_df)

    reader_train_loader = DataLoader(
        reader_train_ds,
        batch_size=TrainingConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=TrainingConfig.NUM_WORKERS,
        pin_memory=True,
    )

    reader_val_loader = DataLoader(
        reader_val_ds,
        batch_size=TrainingConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=TrainingConfig.NUM_WORKERS,
        pin_memory=True,
    )

    return (
        ranker_train_loader,
        ranker_val_loader,
        reader_train_loader,
        reader_val_loader,
    )
