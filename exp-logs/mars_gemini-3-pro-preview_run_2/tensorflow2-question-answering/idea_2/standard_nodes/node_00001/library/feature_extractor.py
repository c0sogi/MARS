import os
import json
import math
import random
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Any, Tuple

from library.config import PathConfig, FeatureConfig, ModelConfig
from library.text_processing import TextPreprocessor
from library.corpus_stats import IDFIndex


class CandidateFeatureGenerator:
    """
    Generates numerical features for Question-Candidate pairs.
    Used for ranking long answer candidates.
    """

    def __init__(self, idf_index: IDFIndex = None):
        self.preprocessor = TextPreprocessor()

        # Load IDF Index
        if idf_index:
            self.idf_index = idf_index
        else:
            self.idf_index = IDFIndex()
            self.idf_index.build_from_corpus(load_cached_data=True)

        # BM25 Hyperparameters
        self.k1 = FeatureConfig.BM25_K1
        self.b = FeatureConfig.BM25_B

        # Heuristic average document length (in tokens) for BM25 normalization
        # Since candidates vary wildly (headers vs paragraphs), we use a fixed proxy
        # based on typical paragraph length to ensure stability.
        self.avg_doc_len = 150.0

    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Generates n-grams from a list of tokens."""
        if len(tokens) < n:
            return []
        return list(zip(*[tokens[i:] for i in range(n)]))

    def compute_bm25(self, q_tokens: List[str], c_tokens: List[str]) -> float:
        """
        Computes BM25 relevance score between question and candidate.
        """
        if not c_tokens or not q_tokens:
            return 0.0

        c_len = len(c_tokens)
        c_token_counts = Counter(c_tokens)
        score = 0.0

        for q_term in q_tokens:
            if q_term not in c_token_counts:
                continue

            idf = self.idf_index.get_idf(q_term)
            tf = c_token_counts[q_term]

            numerator = idf * tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (c_len / self.avg_doc_len)
            )
            score += numerator / denominator

        return score

    def compute_tfidf_cosine(self, q_tokens: List[str], c_tokens: List[str]) -> float:
        """
        Computes Cosine Similarity between TF-IDF vectors of question and candidate.
        """
        if not q_tokens or not c_tokens:
            return 0.0

        q_counts = Counter(q_tokens)
        c_counts = Counter(c_tokens)

        # Get unique terms in both
        all_terms = set(q_counts.keys()) | set(c_counts.keys())

        dot_product = 0.0
        q_norm = 0.0
        c_norm = 0.0

        for term in all_terms:
            idf = self.idf_index.get_idf(term)

            q_val = q_counts.get(term, 0) * idf
            c_val = c_counts.get(term, 0) * idf

            dot_product += q_val * c_val
            q_norm += q_val**2
            c_norm += c_val**2

        if q_norm == 0 or c_norm == 0:
            return 0.0

        return dot_product / (math.sqrt(q_norm) * math.sqrt(c_norm))

    def compute_lexical_overlap(
        self, q_tokens: List[str], c_tokens: List[str]
    ) -> List[float]:
        """
        Computes simple overlap features:
        1. Unigram overlap count
        2. Unigram overlap ratio (relative to Q)
        3. Bigram overlap count
        """
        if not q_tokens or not c_tokens:
            return [0.0, 0.0, 0.0]

        q_set = set(q_tokens)
        c_set = set(c_tokens)

        # Unigrams
        intersection = q_set.intersection(c_set)
        unigram_count = len(intersection)
        unigram_ratio = unigram_count / len(q_set) if len(q_set) > 0 else 0.0

        # Bigrams
        q_bigrams = set(self._get_ngrams(q_tokens, 2))
        c_bigrams = set(self._get_ngrams(c_tokens, 2))
        bigram_count = len(q_bigrams.intersection(c_bigrams))

        return [float(unigram_count), unigram_ratio, float(bigram_count)]

    def compute_positional_features(
        self, candidate_index: int, num_candidates: int
    ) -> List[float]:
        """
        Computes features related to the position of the candidate in the document.
        1. Absolute index (normalized)
        2. Relative index
        3. Is first candidate?
        """
        if num_candidates == 0:
            return [0.0, 0.0, 0.0]

        rel_index = candidate_index / num_candidates
        # Normalize absolute index roughly (e.g., assuming max 1000 candidates)
        abs_norm = min(candidate_index, 1000) / 1000.0
        is_first = 1.0 if candidate_index == 0 else 0.0

        return [abs_norm, rel_index, is_first]

    def compute_length_features(self, c_tokens: List[str]) -> List[float]:
        """
        Computes length-based features.
        1. Token count (log normalized)
        """
        count = len(c_tokens)
        log_count = math.log(count + 1)
        return [log_count]

    def extract_features(
        self,
        question_text: str,
        candidate_text: str,
        candidate_index: int,
        num_candidates: int,
    ) -> np.ndarray:
        """
        Main entry point to generate a feature vector for a single candidate.
        """
        q_tokens = self.preprocessor.preprocess(question_text)
        c_tokens = self.preprocessor.preprocess(candidate_text)

        # 1. BM25 (1 feature)
        f_bm25 = self.compute_bm25(q_tokens, c_tokens)

        # 2. TF-IDF Cosine (1 feature)
        f_tfidf = self.compute_tfidf_cosine(q_tokens, c_tokens)

        # 3. Lexical Overlap (3 features)
        f_lexical = self.compute_lexical_overlap(q_tokens, c_tokens)

        # 4. Positional (3 features)
        f_pos = self.compute_positional_features(candidate_index, num_candidates)

        # 5. Length (1 feature)
        f_len = self.compute_length_features(c_tokens)

        # Combine all features
        features = [f_bm25, f_tfidf] + f_lexical + f_pos + f_len

        return np.array(features, dtype=np.float32)

    def compute_jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Computes Jaccard similarity between two texts.
        Used for Short Answer selection heuristic.
        """
        t1 = set(self.preprocessor.preprocess(text1))
        t2 = set(self.preprocessor.preprocess(text2))

        if not t1 or not t2:
            return 0.0

        intersection = len(t1.intersection(t2))
        union = len(t1.union(t2))

        return intersection / union if union > 0 else 0.0


def build_features_for_dataset(
    jsonl_path: str,
    metadata_path: str,
    output_path: str,
    is_train: bool = False,
    load_cached_data: bool = True,
    sample_size: int = None,
) -> pd.DataFrame:
    """
    Generates features for an entire dataset (Train/Val/Test) and caches to Parquet.
    Handles negative sampling for training data.
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(output_path):
        print(f"Loading features from cache: {output_path}")
        try:
            return pd.read_parquet(output_path)
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # 2. Setup
    PathConfig.ensure_directories()
    feature_gen = CandidateFeatureGenerator()

    # Load Metadata to get labels/ids
    print(f"Loading metadata from {metadata_path}...")
    meta_df = pd.read_csv(metadata_path)

    # Create lookup for ground truth labels if training/validation
    # Map example_id -> long_answer_index
    gt_map = {}
    if "long_answer_index" in meta_df.columns:
        for _, row in meta_df.iterrows():
            gt_map[str(row["example_id"])] = row["long_answer_index"]

    # Filter IDs to process (in case metadata is a subset of JSONL)
    valid_ids = set(meta_df["example_id"].astype(str))

    data_rows = []
    processed_count = 0

    print(f"Processing data from {jsonl_path}...")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if sample_size is not None and processed_count >= sample_size:
                break

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            ex_id = str(entry.get("example_id"))
            if ex_id not in valid_ids:
                continue

            question_text = entry.get("question_text", "")
            doc_text = entry.get("document_text", "")
            doc_tokens = doc_text.split()  # Raw split for indexing
            candidates = entry.get("long_answer_candidates", [])

            # Get Ground Truth Index
            gt_index = gt_map.get(ex_id, -1)

            # Determine which candidates to process
            candidates_to_process = []

            if is_train:
                # Training Strategy: Positive + Sampled Negatives
                pos_candidate = None
                neg_candidates = []

                for idx, cand in enumerate(candidates):
                    # Extract text
                    start = cand["start_token"]
                    end = cand["end_token"]
                    c_text = " ".join(doc_tokens[start:end])

                    cand_obj = {
                        "idx": idx,
                        "text": c_text,
                        "label": 1 if idx == gt_index else 0,
                    }

                    if idx == gt_index:
                        pos_candidate = cand_obj
                    else:
                        neg_candidates.append(cand_obj)

                # Add positive if exists
                if pos_candidate:
                    candidates_to_process.append(pos_candidate)

                # Sample negatives
                # If we have a positive, we need NEG_SAMPLING_RATIO negatives
                # If no positive (unanswerable), we might still want negatives to learn "0"
                num_neg = ModelConfig.NEG_SAMPLING_RATIO
                if len(neg_candidates) > num_neg:
                    sampled_negs = random.sample(neg_candidates, num_neg)
                    candidates_to_process.extend(sampled_negs)
                else:
                    candidates_to_process.extend(neg_candidates)

            else:
                # Inference/Validation: Process ALL candidates
                # (Or top N if we implemented a pre-filter, but config says process all)
                for idx, cand in enumerate(candidates):
                    start = cand["start_token"]
                    end = cand["end_token"]
                    c_text = " ".join(doc_tokens[start:end])

                    candidates_to_process.append(
                        {
                            "idx": idx,
                            "text": c_text,
                            "label": 1 if idx == gt_index else 0,
                        }
                    )

            # Generate Features
            num_cands = len(candidates)
            for item in candidates_to_process:
                feats = feature_gen.extract_features(
                    question_text, item["text"], item["idx"], num_cands
                )

                row = {
                    "example_id": ex_id,
                    "candidate_index": item["idx"],
                    "label": item["label"],
                }

                # Expand features into columns f_0, f_1, ...
                for f_idx, f_val in enumerate(feats):
                    row[f"f_{f_idx}"] = f_val

                data_rows.append(row)

            processed_count += 1
            if processed_count % 1000 == 0:
                print(f"Processed {processed_count} documents...")

    # Create DataFrame
    df = pd.DataFrame(data_rows)

    # Save to Cache
    print(f"Saving {len(df)} feature rows to {output_path}...")
    try:
        df.to_parquet(output_path, index=False)
    except Exception as e:
        print(f"Failed to save cache: {e}")

    return df
