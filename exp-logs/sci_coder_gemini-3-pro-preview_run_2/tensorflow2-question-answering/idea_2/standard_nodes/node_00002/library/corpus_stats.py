import os
import json
import math
import pandas as pd
import numpy as np
from collections import Counter

from library.config import PathConfig
from library.text_processing import TextPreprocessor


class IDFIndex:
    """
    Manages Inverse Document Frequency (IDF) statistics for the corpus.
    Calculates global statistics required for TF-IDF and BM25 features.
    """

    def __init__(self):
        self.idf_map = {}
        # Default IDF for OOV terms (will be updated after build to max IDF)
        self.default_idf = 0.0
        self.preprocessor = TextPreprocessor()

        # Determine cache path.
        # Requirement: Do NOT use pickle. Use parquet.
        # We modify the extension from the config to ensure we use parquet.
        base_path = os.path.splitext(PathConfig.IDF_CACHE)[0]
        self.cache_path = base_path + ".parquet"

    def build_from_corpus(self, sample_size=None, load_cached_data=True):
        """
        Computes IDF statistics from the training corpus.

        Args:
            sample_size (int, optional): Max number of documents to process. Useful for debugging.
            load_cached_data (bool): If True, attempts to load from disk first.
        """
        # 1. Attempt to load from cache
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading IDF stats from {self.cache_path}...")
            try:
                df = pd.read_parquet(self.cache_path)
                self.idf_map = dict(zip(df["token"], df["idf"]))
                if self.idf_map:
                    self.default_idf = max(self.idf_map.values())
                print(f"Loaded {len(self.idf_map)} tokens from cache.")
                return
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing from scratch.")

        # 2. Compute from scratch
        print("Computing IDF stats from corpus...")

        # Load train metadata to filter for training examples only
        # This ensures we don't leak validation info into global stats
        train_ids = set()
        if os.path.exists(PathConfig.TRAIN_META):
            try:
                meta_df = pd.read_csv(PathConfig.TRAIN_META)
                # Ensure IDs are strings for consistent comparison
                train_ids = set(meta_df["example_id"].astype(str))
            except Exception as e:
                print(f"Warning: Could not load train metadata: {e}")

        doc_freqs = Counter()
        num_docs = 0

        # Stream the training file
        if not os.path.exists(PathConfig.TRAIN_JSONL):
            raise FileNotFoundError(
                f"Training data not found at {PathConfig.TRAIN_JSONL}"
            )

        with open(PathConfig.TRAIN_JSONL, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if sample_size is not None and num_docs >= sample_size:
                    break

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Filter: Only use examples in the training split
                # If metadata was missing (train_ids empty), we assume all data is train (fallback)
                ex_id = str(entry.get("example_id"))
                if train_ids and ex_id not in train_ids:
                    continue

                text = entry.get("document_text", "")
                if not text:
                    continue

                # Tokenize and get unique tokens for this document (binary document frequency)
                tokens = set(self.preprocessor.preprocess(text))
                doc_freqs.update(tokens)
                num_docs += 1

                if num_docs % 10000 == 0:
                    print(f"Processed {num_docs} documents...")

        print(f"Finished processing. Total documents: {num_docs}")
        print(f"Total unique tokens: {len(doc_freqs)}")

        # Calculate IDF
        # Formula: log( (N + 1) / (df + 1) ) + 1
        # Standard smoothing to handle potential division by zero (though df+1 prevents it)
        idf_data = []
        for token, df in doc_freqs.items():
            idf_val = math.log((num_docs + 1) / (df + 1)) + 1
            self.idf_map[token] = idf_val
            idf_data.append({"token": token, "idf": idf_val})

        # Set default IDF for unknown words to the max possible IDF (rare word behavior)
        # log((N+1)/1) + 1
        self.default_idf = math.log(num_docs + 1) + 1

        # 3. Save to cache
        print(f"Saving IDF stats to {self.cache_path}...")
        try:
            df_out = pd.DataFrame(idf_data)
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            df_out.to_parquet(self.cache_path, index=False)
            print("Cache saved successfully.")
        except Exception as e:
            print(f"Failed to save cache: {e}")

    def get_idf(self, token):
        """
        Retrieves the IDF value for a given token.
        Returns the default (max) IDF if the token is not in the vocabulary.
        """
        return self.idf_map.get(token, self.default_idf)
