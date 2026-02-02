import os
import json
import random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from library.config import Config


class Tokenizer:
    def __init__(self, config):
        self.config = config
        self.vocab = {}
        self.inv_vocab = {}
        self.unk_token_id = 0
        self.pad_token_id = 1

    def fit(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text)

        # Start after special tokens
        most_common = counter.most_common(self.config.MAX_VOCAB_SIZE - 2)

        self.vocab = {
            self.config.PAD_TOKEN: self.pad_token_id,
            self.config.UNK_TOKEN: self.unk_token_id,
        }
        for i, (token, _) in enumerate(most_common):
            self.vocab[token] = i + 2

        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, tokens, max_len):
        ids = [self.vocab.get(t, self.unk_token_id) for t in tokens]
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [self.pad_token_id] * (max_len - len(ids))
        return ids

    def save(self, path):
        df = pd.DataFrame([{"token": k, "id": v} for k, v in self.vocab.items()])
        df.to_parquet(path)

    def load(self, path):
        df = pd.read_parquet(path)
        self.vocab = dict(zip(df["token"], df["id"]))
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.unk_token_id = self.vocab.get(self.config.UNK_TOKEN, 0)
        self.pad_token_id = self.vocab.get(self.config.PAD_TOKEN, 1)


class HTMLParser:
    @staticmethod
    def extract_candidate_text(doc_tokens, candidate):
        """Extracts text for a specific candidate span."""
        start = candidate["start_token"]
        end = candidate["end_token"]
        # Basic bounds check
        if start < 0 or end > len(doc_tokens) or start >= end:
            return []
        return doc_tokens[start:end]

    @staticmethod
    def get_candidates(data):
        """Returns the list of long answer candidates from the JSON object."""
        return data.get("long_answer_candidates", [])


class HardNegativeMiner:
    def __init__(self, config):
        self.config = config

    def mine(self, question_tokens, positive_idx, candidates, doc_tokens):
        """
        Selects hard negatives based on TF-IDF similarity to the question.
        Returns a list of indices of the negative candidates.
        """
        if len(candidates) <= 1:
            return []

        # Extract text for all candidates
        candidate_texts = []
        valid_indices = []

        for i, cand in enumerate(candidates):
            if i == positive_idx:
                continue
            text = HTMLParser.extract_candidate_text(doc_tokens, cand)
            if not text:
                continue
            candidate_texts.append(" ".join(text))
            valid_indices.append(i)

        if not candidate_texts:
            return []

        # Add question to the set
        q_text = " ".join(question_tokens)
        corpus = [q_text] + candidate_texts

        try:
            vectorizer = TfidfVectorizer(min_df=1, stop_words="english")
            tfidf = vectorizer.fit_transform(corpus)

            # Compute similarity between Question (index 0) and Candidates (indices 1..)
            sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

            # Sort by similarity descending
            sorted_args = np.argsort(sims)[::-1]

            # Select top K hard negatives
            selected_indices = []
            for idx in sorted_args:
                original_idx = valid_indices[idx]
                selected_indices.append(original_idx)
                if len(selected_indices) >= self.config.NUM_NEGATIVES_PER_POSITIVE:
                    break

            return selected_indices

        except ValueError:
            # Fallback if vocabulary is empty or other TFIDF error
            return random.sample(
                valid_indices,
                min(len(valid_indices), self.config.NUM_NEGATIVES_PER_POSITIVE),
            )


def load_json_line(file_handle, offset):
    file_handle.seek(offset)
    line = file_handle.readline()
    if not line:
        return None
    return json.loads(line.decode("utf-8"))


def build_vocab(config, load_cached_data=True):
    """
    Builds or loads the vocabulary.
    """
    if load_cached_data and os.path.exists(config.VOCAB_CACHE_PATH):
        print(f"Loading vocab from {config.VOCAB_CACHE_PATH}")
        tokenizer = Tokenizer(config)
        tokenizer.load(config.VOCAB_CACHE_PATH)
        return tokenizer

    print("Building vocabulary from training data...")
    metadata = pd.read_csv(config.TRAIN_METADATA_PATH)

    # Sample if debugging
    if config.DEBUG_SAMPLE_SIZE:
        metadata = metadata.head(config.DEBUG_SAMPLE_SIZE)

    # We will sample a subset of text to build vocab to save time
    sample_size = min(len(metadata), 20000)
    sample_meta = metadata.sample(n=sample_size, random_state=config.SEED)

    corpus = []
    with open(config.TRAIN_FILE, "rb") as f:
        for _, row in sample_meta.iterrows():
            data = load_json_line(f, row["byte_offset"])
            if not data:
                continue

            # Add question
            q_tokens = data["question_text"].split()
            corpus.append(q_tokens)

            # Add a bit of document text (first 500 tokens)
            doc_tokens = data["document_text"].split()
            corpus.append(doc_tokens[:500])

    tokenizer = Tokenizer(config)
    tokenizer.fit(corpus)

    # Save
    os.makedirs(os.path.dirname(config.VOCAB_CACHE_PATH), exist_ok=True)
    tokenizer.save(config.VOCAB_CACHE_PATH)
    print(f"Vocabulary built and saved. Size: {len(tokenizer.vocab)}")
    return tokenizer


def prepare_ranker_data(config, tokenizer, split="train", load_cached_data=True):
    """
    Prepares data for the Ranker model.
    Output columns: [q_ids, pos_cand_ids, neg_cand_ids_1, neg_cand_ids_2, ...]
    """
    cache_path = (
        config.RANKER_TRAIN_CACHE if split == "train" else config.RANKER_VAL_CACHE
    )
    metadata_path = (
        config.TRAIN_METADATA_PATH if split == "train" else config.VAL_METADATA_PATH
    )

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading ranker {split} data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing ranker {split} data...")
    metadata = pd.read_csv(metadata_path)

    # Filter for examples with long answers
    metadata = metadata[metadata["has_long_answer"] == True]

    if config.DEBUG_SAMPLE_SIZE:
        metadata = metadata.head(config.DEBUG_SAMPLE_SIZE)

    miner = HardNegativeMiner(config)
    records = []

    with open(config.TRAIN_FILE, "rb") as f:
        for _, row in metadata.iterrows():
            data = load_json_line(f, row["byte_offset"])
            if not data:
                continue

            q_tokens = data["question_text"].split()
            doc_tokens = data["document_text"].split()
            candidates = HTMLParser.get_candidates(data)

            # Find positive candidate index
            # The annotation contains the long answer candidate index or we match spans
            # Simplified NQ annotations usually point to one of the candidates
            # We need to find which candidate matches the annotation long answer span
            ann = data["annotations"][0]  # Use first annotation
            la_start = ann["long_answer"]["start_token"]
            la_end = ann["long_answer"]["end_token"]

            pos_idx = -1
            for i, cand in enumerate(candidates):
                if cand["start_token"] == la_start and cand["end_token"] == la_end:
                    pos_idx = i
                    break

            if pos_idx == -1:
                continue

            # Mine negatives (only for training, val can just take random or none if evaluating differently)
            # For this pipeline, we assume validation also uses the ranking loss structure
            neg_indices = miner.mine(q_tokens, pos_idx, candidates, doc_tokens)

            # If not enough negatives found, pad with random valid candidates
            while len(neg_indices) < config.NUM_NEGATIVES_PER_POSITIVE:
                possible = [
                    i
                    for i in range(len(candidates))
                    if i != pos_idx and i not in neg_indices
                ]
                if not possible:
                    break  # Not enough candidates in document
                neg_indices.append(random.choice(possible))

            # If still not enough (small doc), duplicate
            if not neg_indices:
                continue  # Skip single-candidate docs if we need negatives

            while len(neg_indices) < config.NUM_NEGATIVES_PER_POSITIVE:
                neg_indices.append(neg_indices[0])

            # Encode
            q_enc = tokenizer.encode(q_tokens, config.MAX_Q_LEN)
            pos_text = HTMLParser.extract_candidate_text(
                doc_tokens, candidates[pos_idx]
            )
            pos_enc = tokenizer.encode(pos_text, config.MAX_DOC_LEN)

            record = {"q_ids": q_enc, "pos_ids": pos_enc}

            for i in range(config.NUM_NEGATIVES_PER_POSITIVE):
                neg_text = HTMLParser.extract_candidate_text(
                    doc_tokens, candidates[neg_indices[i]]
                )
                neg_enc = tokenizer.encode(neg_text, config.MAX_DOC_LEN)
                record[f"neg_ids_{i}"] = neg_enc

            records.append(record)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path)
    print(f"Ranker {split} data saved. Rows: {len(df)}")
    return df


def prepare_reader_data(config, tokenizer, split="train", load_cached_data=True):
    """
    Prepares data for the Reader model.
    Output columns: [input_ids, start_token_idx, end_token_idx]
    """
    cache_path = (
        config.READER_TRAIN_CACHE if split == "train" else config.READER_VAL_CACHE
    )
    metadata_path = (
        config.TRAIN_METADATA_PATH if split == "train" else config.VAL_METADATA_PATH
    )

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading reader {split} data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing reader {split} data...")
    metadata = pd.read_csv(metadata_path)

    # Filter for examples with short answers
    metadata = metadata[metadata["has_short_answer"] == True]

    if config.DEBUG_SAMPLE_SIZE:
        metadata = metadata.head(config.DEBUG_SAMPLE_SIZE)

    records = []

    with open(config.TRAIN_FILE, "rb") as f:
        for _, row in metadata.iterrows():
            data = load_json_line(f, row["byte_offset"])
            if not data:
                continue

            q_tokens = data["question_text"].split()
            doc_tokens = data["document_text"].split()

            ann = data["annotations"][0]

            # We need the long answer context that contains the short answer
            la_start = ann["long_answer"]["start_token"]
            la_end = ann["long_answer"]["end_token"]

            if la_start == -1:
                continue  # Should be filtered by metadata, but double check

            context_tokens = doc_tokens[la_start:la_end]

            # Determine Short Answer Span
            # Short answers are relative to document. We need them relative to context.
            # Usually short answers are inside the long answer.
            # If yes/no, we might handle differently, but for this task description:
            # "predict a) a set of start:end token indices, b) a YES/NO answer"
            # For simplicity in this reader, we focus on span extraction.
            # Yes/No would typically be a classification head.

            short_answers = ann["short_answers"]
            if not short_answers:
                # Could be YES/NO.
                # For this implementation, we skip YES/NO only examples for span training
                # or treat them as CLS task. Let's skip for span reader simplicity.
                continue

            sa_start_doc = short_answers[0]["start_token"]
            sa_end_doc = short_answers[0]["end_token"]

            # Calculate relative indices
            rel_start = sa_start_doc - la_start
            rel_end = sa_end_doc - la_start

            # Sanity check
            if rel_start < 0 or rel_end > len(context_tokens):
                continue

            # Construct Input: [Q] + [Context]
            # We don't have a SEP token in simple config, just concat
            # Ideally Tokenizer handles special tokens, but we used simple split
            # We will just concat lists.

            combined_tokens = q_tokens + context_tokens

            # Adjust targets for Question length
            final_start = len(q_tokens) + rel_start
            final_end = len(q_tokens) + rel_end

            # Truncate if necessary (Reader usually handles max seq len)
            if len(combined_tokens) > config.MAX_READER_SEQ_LEN:
                # If answer is cut off, skip
                if final_end >= config.MAX_READER_SEQ_LEN:
                    continue
                combined_tokens = combined_tokens[: config.MAX_READER_SEQ_LEN]

            input_ids = tokenizer.encode(combined_tokens, config.MAX_READER_SEQ_LEN)

            records.append(
                {
                    "input_ids": input_ids,
                    "start_token_idx": final_start,
                    "end_token_idx": final_end,
                }
            )

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path)
    print(f"Reader {split} data saved. Rows: {len(df)}")
    return df
