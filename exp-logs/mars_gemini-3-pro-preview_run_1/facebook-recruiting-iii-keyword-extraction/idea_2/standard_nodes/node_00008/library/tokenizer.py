import os
import re
import json
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict, Union, Optional

from library.config import Config
from library.utils import set_seed


class TextProcessor:
    """
    Handles text preprocessing, vocabulary building, and label encoding
    for the Stack Exchange Tag Prediction task.
    """

    def __init__(self):
        self.word_to_idx: Dict[str, int] = {}
        self.tag_to_idx: Dict[str, int] = {}
        self.idx_to_tag: Dict[int, str] = {}

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_idx = 0
        self.unk_idx = 1

        # File paths for caching
        self.tokenizer_path = Config.TOKENIZER_PATH
        self.label_encoder_path = os.path.join(Config.WORKING_DIR, "tag_map.json")

    def _clean_text(self, text: str) -> List[str]:
        """
        Basic tokenization: lowercase and split by alphanumeric characters.
        """
        if not isinstance(text, str):
            return []
        # Convert to lowercase and find all alphanumeric sequences
        # This implicitly removes punctuation and special characters
        tokens = re.findall(r"\w+", text.lower())
        return tokens

    def fit(self, load_cached_data: bool = True) -> None:
        """
        Builds the vocabulary and tag mappings.

        Args:
            load_cached_data (bool): If True, attempts to load from disk first.
        """
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # 1. Try to load from cache
        if load_cached_data:
            if os.path.exists(self.tokenizer_path) and os.path.exists(
                self.label_encoder_path
            ):
                print(f"Loading cached tokenizer from {self.tokenizer_path}...")
                with open(self.tokenizer_path, "r") as f:
                    self.word_to_idx = json.load(f)

                print(f"Loading cached tag map from {self.label_encoder_path}...")
                with open(self.label_encoder_path, "r") as f:
                    data = json.load(f)
                    self.tag_to_idx = data["tag_to_idx"]
                    self.idx_to_tag = {int(k): v for k, v in data["idx_to_tag"].items()}

                print("Resources loaded successfully.")
                return

        print("Building vocabulary and tag maps from scratch...")

        # 2. Build Tag Vocabulary
        # We use metadata for tags as it is lighter and contains all labels
        print(f"Reading tags from {Config.TRAIN_METADATA}...")
        df_meta = pd.read_csv(Config.TRAIN_METADATA, usecols=["Tags"])
        df_meta["Tags"] = df_meta["Tags"].fillna("").astype(str)

        # Count all tags
        all_tags = df_meta["Tags"].str.split().explode()
        tag_counts = all_tags.value_counts()

        # Select top NUM_TAGS
        top_tags = tag_counts.head(Config.NUM_TAGS).index.tolist()

        self.tag_to_idx = {tag: i for i, tag in enumerate(top_tags)}
        self.idx_to_tag = {i: tag for i, tag in enumerate(top_tags)}

        print(
            f"Selected {len(self.tag_to_idx)} top tags from {len(tag_counts)} unique tags."
        )

        # Free memory
        del df_meta, all_tags, tag_counts

        # 3. Build Word Vocabulary
        # We need to read the actual text from train.csv
        print(f"Reading text from {Config.TRAIN_CSV}...")
        # To save time/memory, we can sample if the dataset is massive,
        # but with 220GB RAM we can try reading the whole text or a large chunk.
        # Given the time limit, we'll read the whole file but process efficiently.
        try:
            df_text = pd.read_csv(
                Config.TRAIN_CSV,
                usecols=["Title", "Body"],
                dtype={"Title": "object", "Body": "object"},
            )
        except Exception as e:
            print(
                f"Error reading train.csv: {e}. Fallback to chunking not implemented for brevity."
            )
            raise e

        print("Processing text to build vocabulary...")
        word_counter = Counter()

        # Process in batches to show some progress or just do it all
        # Vectorized operations are hard for tokenization, using list comp
        # Combine Title and Body for vocab building

        # Fill NaNs
        df_text["Title"] = df_text["Title"].fillna("")
        df_text["Body"] = df_text["Body"].fillna("")

        # We iterate and update counter
        # Sampling 50% of data for vocab building is usually sufficient and faster
        # if the dataset is extremely large, but here we aim for best quality.
        # Let's use a sample of 1M rows if the dataset is larger than 1M to speed up
        if len(df_text) > 1000000:
            print("Subsampling 1M rows for vocabulary building to save time...")
            df_sample = df_text.sample(n=1000000, random_state=Config.SEED)
        else:
            df_sample = df_text

        for title, body in zip(df_sample["Title"], df_sample["Body"]):
            tokens = self._clean_text(title + " " + body)
            word_counter.update(tokens)

        print(f"Total unique words found: {len(word_counter)}")

        # Select top VOCAB_SIZE words
        most_common = word_counter.most_common(Config.VOCAB_SIZE)

        # 0 is PAD, 1 is UNK
        self.word_to_idx = {self.pad_token: self.pad_idx, self.unk_token: self.unk_idx}
        start_idx = 2
        for word, _ in most_common:
            self.word_to_idx[word] = start_idx
            start_idx += 1

        print(f"Vocabulary built. Size: {len(self.word_to_idx)}")

        # 4. Save to cache
        print("Saving artifacts to cache...")
        with open(self.tokenizer_path, "w") as f:
            json.dump(self.word_to_idx, f)

        with open(self.label_encoder_path, "w") as f:
            json.dump({"tag_to_idx": self.tag_to_idx, "idx_to_tag": self.idx_to_tag}, f)

        print("Fit complete.")

    def encode_text(self, titles: List[str], bodies: List[str]) -> List[List[int]]:
        """
        Tokenizes and maps text to integers.
        Combines Title and Body.
        """
        encoded_batch = []

        for t, b in zip(titles, bodies):
            # Combine
            text = (str(t) if t else "") + " " + (str(b) if b else "")
            tokens = self._clean_text(text)

            # Map to indices
            indices = [self.word_to_idx.get(token, self.unk_idx) for token in tokens]

            # Handle empty sequences
            if not indices:
                indices = [self.pad_idx]

            # Truncate to MAX_LEN
            if len(indices) > Config.MAX_LEN:
                indices = indices[: Config.MAX_LEN]

            encoded_batch.append(indices)

        return encoded_batch

    def encode_tags(self, tags_list: List[str]) -> np.ndarray:
        """
        Converts a list of tag strings (space-delimited) to a multi-hot binary matrix.
        Shape: (batch_size, NUM_TAGS)
        """
        batch_size = len(tags_list)
        num_classes = len(self.tag_to_idx)

        # Initialize zero matrix
        targets = np.zeros((batch_size, num_classes), dtype=np.float32)

        for i, tags_str in enumerate(tags_list):
            if not isinstance(tags_str, str):
                continue

            tags = tags_str.split()
            for tag in tags:
                if tag in self.tag_to_idx:
                    idx = self.tag_to_idx[tag]
                    targets[i, idx] = 1.0

        return targets

    def decode_tags(
        self, preds: Union[np.ndarray, "torch.Tensor"], threshold: float = None
    ) -> List[str]:
        """
        Converts probability vectors or binary vectors to space-delimited tag strings.

        Args:
            preds: Array of shape (batch_size, NUM_TAGS) containing probs or binary.
            threshold: If provided, applies threshold to probabilities.
                       If None, assumes input is already binary or uses Config.TAG_THRESHOLD.
        """
        # Handle torch tensors
        if hasattr(preds, "cpu"):
            preds = preds.detach().cpu().numpy()

        if threshold is None:
            threshold = Config.TAG_THRESHOLD

        # Apply threshold if probabilities (assuming float input implies probs)
        if preds.dtype == np.float32 or preds.dtype == np.float64:
            binary_preds = (preds > threshold).astype(int)
        else:
            binary_preds = preds.astype(int)

        decoded_batch = []
        for row in binary_preds:
            # Get indices where value is 1
            indices = np.where(row == 1)[0]

            # Map to tags
            tags = [self.idx_to_tag[idx] for idx in indices if idx in self.idx_to_tag]

            # Join
            decoded_batch.append(" ".join(tags))

        return decoded_batch

    def get_vocab_size(self) -> int:
        return len(self.word_to_idx)

    def get_num_tags(self) -> int:
        return len(self.tag_to_idx)
