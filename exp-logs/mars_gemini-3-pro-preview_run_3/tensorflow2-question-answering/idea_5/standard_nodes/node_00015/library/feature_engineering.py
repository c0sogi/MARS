import os
import json
import math
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from library.configuration import Config
from library.text_utils import tokenize, extract_structural_features


class BM25:
    """
    Implementation of the BM25 retrieval algorithm.
    """

    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.idf = {}
        self.avg_dl = 0
        self.doc_count = 0

    def fit(self, corpus_tokens):
        """
        Calculates IDF and average document length from the corpus.

        Args:
            corpus_tokens (list of list of str): Tokenized documents.
        """
        self.doc_count = len(corpus_tokens)
        if self.doc_count == 0:
            return

        doc_lengths = []
        df = Counter()

        for tokens in corpus_tokens:
            doc_lengths.append(len(tokens))
            unique_tokens = set(tokens)
            df.update(unique_tokens)

        self.avg_dl = np.mean(doc_lengths) if doc_lengths else 0

        for term, freq in df.items():
            # Standard IDF formula with smoothing
            self.idf[term] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1)

    def score(self, query_tokens, doc_tokens):
        """
        Computes the BM25 score for a query against a document.
        """
        score = 0.0
        doc_len = len(doc_tokens)
        doc_counter = Counter(doc_tokens)

        for q in query_tokens:
            if q not in self.idf:
                continue

            f_q_D = doc_counter[q]
            idf = self.idf[q]

            numerator = f_q_D * (self.k1 + 1)
            denominator = f_q_D + self.k1 * (
                1 - self.b + self.b * (doc_len / (self.avg_dl + 1e-6))
            )

            score += idf * (numerator / denominator)

        return score


class FeatureExtractor:
    """
    Extracts lexical and structural features for the Ranker model.
    """

    def __init__(self):
        self.bm25 = BM25(k1=Config.BM25_K1, b=Config.BM25_B)
        # We use a simple whitespace tokenizer lambda to preserve our own tokenization
        self.tfidf = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=None,
            max_features=10000,
        )
        self.is_fitted = False

    def fit(self, questions, candidates):
        """
        Fits internal statistical models (BM25, TF-IDF) on a sample of data.

        Args:
            questions (list of list of str): Sample question tokens.
            candidates (list of list of str): Sample candidate tokens.
        """
        # Fit BM25 on candidates (the corpus to retrieve from)
        self.bm25.fit(candidates)

        # Fit TF-IDF on the combined corpus to ensure a shared vocabulary space
        corpus = questions + candidates
        if not corpus:
            print("Warning: Empty corpus provided to FeatureExtractor.fit")
            self.is_fitted = True
            return

        self.tfidf.fit(corpus)
        self.is_fitted = True

    def compute_features(
        self, query_tokens, candidate_tokens, candidate_idx, total_candidates
    ):
        """
        Generates a dictionary of scalar features.
        """
        if not self.is_fitted:
            # Fallback if not fitted (should not happen in proper pipeline)
            return {}

        features = {}

        # --- Lexical Features ---

        # 1. BM25 Score
        features["bm25_score"] = self.bm25.score(query_tokens, candidate_tokens)

        # 2. TF-IDF Cosine Similarity
        try:
            # transform expects a list of documents (lists of tokens)
            vecs = self.tfidf.transform([query_tokens, candidate_tokens])
            # dense output for cosine_similarity
            cosine = cosine_similarity(vecs[0], vecs[1])[0][0]
        except (ValueError, IndexError):
            cosine = 0.0
        features["tfidf_cosine"] = cosine

        # 3. N-gram Overlaps
        q_set = set(query_tokens)
        c_set = set(candidate_tokens)

        unigram_overlap = len(q_set.intersection(c_set))
        features["unigram_overlap_count"] = unigram_overlap

        # Precision-like and Recall-like ratios
        features["unigram_query_ratio"] = (
            unigram_overlap / len(q_set) if len(q_set) > 0 else 0.0
        )
        features["unigram_candidate_ratio"] = (
            unigram_overlap / len(c_set) if len(c_set) > 0 else 0.0
        )

        # Bigrams
        if len(query_tokens) > 1 and len(candidate_tokens) > 1:
            q_bigrams = set(zip(query_tokens[:-1], query_tokens[1:]))
            c_bigrams = set(zip(candidate_tokens[:-1], candidate_tokens[1:]))
            bigram_overlap = len(q_bigrams.intersection(c_bigrams))
            features["bigram_overlap_count"] = bigram_overlap
        else:
            features["bigram_overlap_count"] = 0

        # --- Structural Features ---
        struct_feats = extract_structural_features(candidate_tokens)
        features["is_paragraph"] = int(struct_feats["is_paragraph"])
        features["is_table"] = int(struct_feats["is_table"])
        features["is_list"] = int(struct_feats["is_list"])
        features["is_heading"] = int(struct_feats["is_heading"])
        features["is_other"] = int(struct_feats["is_other"])

        # --- Meta Features ---
        c_len = len(candidate_tokens)
        features["candidate_len"] = c_len
        features["log_candidate_len"] = math.log(c_len + 1)

        # Normalized position in document (0.0 = start, 1.0 = end)
        features["normalized_pos"] = candidate_idx / max(total_candidates, 1)

        return features


def get_candidates_from_json(data):
    """
    Extracts candidate text blocks from the parsed JSON object.

    Returns:
        list of dict: Each dict contains 'tokens', 'start_token', 'end_token', etc.
    """
    doc_text = data.get("document_text", "")
    doc_tokens = doc_text.split()  # Raw split as per text_utils.tokenize logic

    candidates_info = data.get("long_answer_candidates", [])

    candidates = []
    for idx, cand in enumerate(candidates_info):
        start = cand["start_token"]
        end = cand["end_token"]

        # Safety check indices
        if start < 0 or end > len(doc_tokens) or start >= end:
            continue

        # Extract tokens
        c_tokens = doc_tokens[start:end]

        # Filter very short candidates to reduce noise
        if len(c_tokens) < 5:  # Arbitrary small threshold for sanity
            continue

        candidates.append(
            {
                "tokens": c_tokens,
                "start_token": start,
                "end_token": end,
                "original_index": idx,
                "top_level": cand["top_level"],
            }
        )
    return candidates


def create_ranker_dataset(
    metadata_path, output_path, is_train=True, load_cached_data=True, sample_size=None
):
    """
    Generates the feature dataset for the Ranker model.

    Args:
        metadata_path (str): Path to the metadata CSV (train, val, or test).
        output_path (str): Path to save the resulting Parquet file.
        is_train (bool): If True, extracts labels and performs negative sampling.
        load_cached_data (bool): If True, attempts to load from output_path first.
        sample_size (int, optional): Limit processing to N examples for debugging.

    Returns:
        pd.DataFrame: The generated dataset.
    """
    # 1. Caching Check
    if load_cached_data and os.path.exists(output_path):
        print(f"Loading cached ranker data from {output_path}")
        return pd.read_parquet(output_path)

    print(f"Generating ranker data (is_train={is_train}) from {metadata_path}...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load Metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)
    if sample_size is not None:
        df_meta = df_meta.head(sample_size)

    # Determine source data file based on metadata filename logic
    # The metadata CSV contains 'file_path' column which points to the correct file in input/
    # We assume all rows in one metadata file point to the same source file type (train or test)
    # but we should handle reading row-by-row safely.

    # 2. Fit Feature Extractor
    # We need to fit the extractor on a representative sample of text.
    # We'll sample up to 5000 examples from the current split.
    extractor = FeatureExtractor()

    print("Sampling data to fit feature extractor...")
    fit_samples = df_meta.sample(n=min(len(df_meta), 5000), random_state=Config.SEED)
    fit_questions = []
    fit_candidates = []

    # We need to open files dynamically based on row['file_path']
    # To optimize, we group by file_path
    for file_name, group in fit_samples.groupby("file_path"):
        file_path = os.path.join(Config.INPUT_DIR, file_name)
        with open(file_path, "rb") as f:
            for _, row in group.iterrows():
                f.seek(row["byte_offset"])
                line = f.readline()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    q_tokens = tokenize(data["question_text"])
                    fit_questions.append(q_tokens)

                    cands = get_candidates_from_json(data)
                    # Sample a few candidates per doc to keep memory usage low
                    if cands:
                        fit_candidates.extend([c["tokens"] for c in cands[:5]])
                except json.JSONDecodeError:
                    continue

    print("Fitting feature extractor...")
    extractor.fit(fit_questions, fit_candidates)

    # 3. Generate Features
    records = []

    # Iterate through metadata to process all examples
    # Group by file_path to minimize file open/close operations
    for file_name, group in df_meta.groupby("file_path"):
        file_path = os.path.join(Config.INPUT_DIR, file_name)
        print(f"Processing file: {file_name} with {len(group)} examples...")

        with open(file_path, "rb") as f:
            for i, (_, row) in enumerate(group.iterrows()):
                f.seek(row["byte_offset"])
                line = f.readline()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                q_tokens = tokenize(data.get("question_text", ""))
                candidates = get_candidates_from_json(data)

                if not candidates:
                    continue

                # Pre-calculate BM25 scores for all candidates for ranking/sampling
                bm25_scores = []
                for c in candidates:
                    s = extractor.bm25.score(q_tokens, c["tokens"])
                    bm25_scores.append(s)
                bm25_scores = np.array(bm25_scores)

                selected_indices = []
                labels = []

                if is_train:
                    # Identify Ground Truth
                    positive_idx = -1
                    annotations = data.get("annotations", [])

                    # Look for a valid long answer
                    for ann in annotations:
                        la = ann.get("long_answer", {})
                        la_start = la.get("start_token", -1)
                        if la_start != -1:
                            # Match against candidates
                            # We allow a small tolerance or exact match. NQ usually exact.
                            for c_idx, c in enumerate(candidates):
                                if (
                                    c["start_token"] == la_start
                                    and c["end_token"] == la["end_token"]
                                ):
                                    positive_idx = c_idx
                                    break
                        if positive_idx != -1:
                            break

                    # Sampling Strategy
                    # 1. Positive Sample
                    if positive_idx != -1:
                        selected_indices.append(positive_idx)
                        labels.append(1)

                    # 2. Hard Negative Mining (Top BM25 scores that are NOT the positive)
                    # Sort indices by score descending
                    sorted_indices = np.argsort(bm25_scores)[::-1]

                    neg_count = 0
                    target_neg_count = Config.RANKER_NEG_SAMPLES

                    # If no positive exists (unanswerable), we might want more negatives or just standard amount
                    if positive_idx == -1:
                        target_neg_count += 1

                    for idx in sorted_indices:
                        if idx == positive_idx:
                            continue

                        selected_indices.append(idx)
                        labels.append(0)
                        neg_count += 1
                        if neg_count >= target_neg_count:
                            break

                else:
                    # Inference Mode: Select Top-K candidates for re-ranking
                    # We use BM25 as the initial retriever
                    top_k = min(len(candidates), Config.TOP_K_RETRIEVAL)
                    # Get indices of top k scores
                    # argsort is ascending, so slice from end
                    selected_indices = np.argsort(bm25_scores)[::-1][:top_k]
                    # Dummy labels for test set
                    labels = [-1] * len(selected_indices)

                # Compute Features for Selected Candidates
                for idx, label in zip(selected_indices, labels):
                    cand = candidates[idx]
                    feats = extractor.compute_features(
                        q_tokens, cand["tokens"], idx, len(candidates)
                    )

                    record = {
                        "example_id": row["example_id"],
                        "candidate_index": idx,
                        "start_token": cand["start_token"],
                        "end_token": cand["end_token"],
                        "label": label,
                    }
                    record.update(feats)
                    records.append(record)

                if (i + 1) % 5000 == 0:
                    print(f"  Processed {i + 1} examples in current file...")

    df_ranker = pd.DataFrame(records)

    # Save to cache
    df_ranker.to_parquet(output_path, index=False)
    print(f"Ranker dataset saved to {output_path}. Shape: {df_ranker.shape}")

    return df_ranker
