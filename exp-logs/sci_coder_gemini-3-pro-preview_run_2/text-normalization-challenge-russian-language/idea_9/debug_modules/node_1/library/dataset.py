import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import is_semiotic
from library.hfbb import HFBBModel


class ContextWindowDataset(Dataset):
    """
    PyTorch Dataset for the Tier 2 Transformer.
    Extracts a context window around a target token and tokenizes input/output.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        indices: np.ndarray,
        char_tokenizer,
        target_tokenizer=None,
        max_len_char: int = Config.MAX_LEN_CHAR,
        max_len_subword: int = Config.MAX_LEN_SUBWORD,
        mode: str = "train",
    ):
        """
        Args:
            df (pd.DataFrame): The full dataframe (train, val, or test).
            indices (np.ndarray): Array of indices (rows in df) to include in this dataset.
            char_tokenizer: Tokenizer for the character-level input.
            target_tokenizer: Tokenizer for the subword-level target (optional for test).
            max_len_char (int): Max sequence length for input.
            max_len_subword (int): Max sequence length for target.
            mode (str): 'train', 'val', or 'test'.
        """
        self.indices = indices
        self.char_tokenizer = char_tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_len_char = max_len_char
        self.max_len_subword = max_len_subword
        self.mode = mode
        self.window_size = Config.CONTEXT_WINDOW  # e.g., 2

        # Convert relevant columns to numpy arrays for faster access during __getitem__
        # Accessing dataframe rows one-by-one is slow.
        self.sentence_ids = df["sentence_id"].values
        self.before_tokens = df["before"].astype(str).values

        # 'after' is only present in train/val
        if "after" in df.columns:
            self.after_tokens = df["after"].astype(str).values
        else:
            self.after_tokens = None

        # Pre-compute sentence boundaries to avoid logic in loop
        # We can check if sentence_id[i] == sentence_id[i+k]
        self.data_len = len(df)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # real_idx is the index in the original dataframe
        real_idx = self.indices[idx]

        # 1. Extract Context
        current_sentence_id = self.sentence_ids[real_idx]
        target_token = self.before_tokens[real_idx]

        # Previous tokens
        prev_tokens = []
        for i in range(self.window_size, 0, -1):
            lookback_idx = real_idx - i
            if (
                lookback_idx >= 0
                and self.sentence_ids[lookback_idx] == current_sentence_id
            ):
                prev_tokens.append(self.before_tokens[lookback_idx])
            else:
                # Use a placeholder or just empty.
                # Usually empty string or specific token.
                # We'll skip adding anything to keep context clean, or use <start> if preferred.
                # Let's use specific tokens if needed, but here we just omit out-of-boundary text.
                pass

        # Next tokens
        next_tokens = []
        for i in range(1, self.window_size + 1):
            lookahead_idx = real_idx + i
            if (
                lookahead_idx < self.data_len
                and self.sentence_ids[lookahead_idx] == current_sentence_id
            ):
                next_tokens.append(self.before_tokens[lookahead_idx])
            else:
                pass

        # 2. Construct Input String
        # Format: "prev2 prev1 <sep> target <sep> next1 next2"
        # We join with spaces to separate words, assuming CharTokenizer handles spaces or treats them as chars.
        sep = self.char_tokenizer.sep_token

        prev_str = " ".join(prev_tokens)
        next_str = " ".join(next_tokens)

        # Ensure spaces around separator for clarity, though tokenizer might split by sep token specifically
        input_text = f"{prev_str} {sep} {target_token} {sep} {next_str}".strip()

        # Clean up potential double spaces if prev/next were empty
        input_text = " ".join(input_text.split())

        # 3. Tokenize Input
        src_ids = self.char_tokenizer.encode(
            input_text, max_len=self.max_len_char, add_special_tokens=True
        )

        # 4. Tokenize Target (if available)
        tgt_ids = []
        if self.after_tokens is not None and self.target_tokenizer is not None:
            target_text = self.after_tokens[real_idx]
            tgt_ids = self.target_tokenizer.encode(
                target_text, max_len=self.max_len_subword, add_special_tokens=True
            )

        # Convert to tensors
        src_tensor = torch.tensor(src_ids, dtype=torch.long)

        item = {
            "src_ids": src_tensor,
            "raw_before": target_token,
            "id": real_idx,  # Useful for tracking
        }

        if len(tgt_ids) > 0:
            item["tgt_ids"] = torch.tensor(tgt_ids, dtype=torch.long)
            item["raw_after"] = target_text

        return item


def prepare_curriculum_indices(
    df: pd.DataFrame, hfbb_model: HFBBModel, load_cached_data: bool = True
) -> np.ndarray:
    """
    Selects indices for the 'Confidence-Gated Curriculum' training strategy.

    Selection Logic:
    1. Residuals: Where HFBB prediction != Truth.
    2. Ambiguous: Where HFBB prediction == Truth BUT Confidence < Threshold.
    3. Anchors: Random 20% sample of High-Confidence Semiotic tokens.

    Also applies upsampling to rare classes within the selected set.

    Args:
        df (pd.DataFrame): Training dataframe.
        hfbb_model (HFBBModel): Trained statistical model.
        load_cached_data (bool): Whether to load indices from disk.

    Returns:
        np.ndarray: Array of selected indices.
    """
    # Cache path
    cache_path = os.path.join(Config.WORKING_DIR, "curriculum_indices.npy")

    # 1. Load Cache if requested
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading curriculum indices from {cache_path}...")
        return np.load(cache_path)

    print("Generating curriculum indices from scratch...")

    # 2. Prepare Data for Vectorized HFBB Prediction
    # We need prev/next tokens to query HFBB efficiently
    # Using shift logic similar to HFBB._compute_stats

    # Ensure string types
    before_series = df["before"].astype(str)
    after_series = df["after"].astype(str)
    sentence_ids = df["sentence_id"]

    # Shift for context
    prev_series = before_series.shift(1).fillna("<start>")
    # Mask out boundaries
    mask_start = sentence_ids != sentence_ids.shift(1)
    prev_series[mask_start] = "<start>"

    next_series = before_series.shift(-1).fillna("<end>")
    mask_end = sentence_ids != sentence_ids.shift(-1)
    next_series[mask_end] = "<end>"

    # 3. Run HFBB Predictions (Vectorized-ish via map)
    # Since HFBB uses dict lookups, we can apply a function row-wise or use map
    # For 7M rows, a loop might be slow, but map is decent.

    print("Running HFBB predictions on training set...")

    # Extract underlying dicts for faster direct lookup
    trigram_map = hfbb_model.trigram_map
    bigram_prev_map = hfbb_model.bigram_prev_map
    bigram_next_map = hfbb_model.bigram_next_map
    unigram_map = hfbb_model.unigram_map

    # Arrays for results
    n = len(df)
    is_correct = np.zeros(n, dtype=bool)
    confidence = np.zeros(n, dtype=np.float32)

    # Convert to list of tuples for iteration (faster than df.itertuples for simple types)
    # Zip is fast
    data_iter = zip(prev_series, before_series, next_series, after_series)

    # Iterate
    # We collect indices for different categories
    residual_indices = []
    ambiguous_indices = []
    anchor_candidates = []

    for idx, (p, curr, nxt, truth) in enumerate(data_iter):
        pred = None
        conf = 0.0

        # Hierarchy Lookup
        if (p, curr, nxt) in trigram_map:
            pred = trigram_map[(p, curr, nxt)]
            conf = 1.0
        elif (p, curr) in bigram_prev_map:
            pred = bigram_prev_map[(p, curr)]
            conf = 1.0
        elif (curr, nxt) in bigram_next_map:
            pred = bigram_next_map[(curr, nxt)]
            conf = 1.0
        elif curr in unigram_map:
            pred, conf = unigram_map[curr]

        # Check correctness
        # If pred is None, it's incorrect (unless truth is also None/Empty, which shouldn't happen)
        correct = pred == truth

        if not correct:
            # Residual: HFBB failed
            residual_indices.append(idx)
        else:
            # HFBB Correct
            if conf < Config.AMBIGUITY_THRESHOLD:
                # Ambiguous: Correct but lucky/uncertain
                ambiguous_indices.append(idx)
            else:
                # High Confidence Correct
                # Check if semiotic (we only care about anchors for semiotic tokens)
                # Simple check: contains digit or letter
                if is_semiotic(curr):
                    anchor_candidates.append(idx)

    print(f"Residuals found: {len(residual_indices)}")
    print(f"Ambiguous found: {len(ambiguous_indices)}")
    print(f"Anchor candidates: {len(anchor_candidates)}")

    # 4. Sample Anchors
    # We take a percentage of the high-confidence semiotic tokens
    rng = np.random.default_rng(Config.SEED)
    num_anchors = int(len(anchor_candidates) * Config.ANCHOR_RATIO)
    if num_anchors > 0:
        selected_anchors = rng.choice(
            anchor_candidates, size=num_anchors, replace=False
        )
    else:
        selected_anchors = []

    print(f"Selected anchors: {len(selected_anchors)}")

    # 5. Combine Indices
    base_indices = np.concatenate(
        [
            np.array(residual_indices, dtype=np.int64),
            np.array(ambiguous_indices, dtype=np.int64),
            np.array(selected_anchors, dtype=np.int64),
        ]
    )

    # 6. Class-Balanced Upsampling (Optional but recommended in config)
    if Config.UPSAMPLE_RARE_CLASSES and "class" in df.columns:
        print("Applying class-balanced upsampling...")
        # Get classes for the selected indices
        selected_classes = df.iloc[base_indices]["class"].values

        # Identify rare classes (anything not PLAIN or PUNCT)
        # We can just check counts within the selection
        unique, counts = np.unique(selected_classes, return_counts=True)
        # Simple heuristic: if count < mean count, upsample
        # Or specifically target known rare classes: MONEY, MEASURE, DECIMAL, ORDINAL
        target_rare = {
            "MONEY",
            "MEASURE",
            "DECIMAL",
            "ORDINAL",
            "TELEPHONE",
            "ELECTRONIC",
            "DIGIT",
        }

        upsample_indices = []

        # Map class to indices within base_indices
        # We iterate base_indices
        for i, cls in enumerate(selected_classes):
            if cls in target_rare:
                # Add this index again (e.g., 2x or 3x)
                # Let's add it 2 more times (total 3x)
                original_idx = base_indices[i]
                upsample_indices.extend([original_idx] * 2)

        if upsample_indices:
            print(f"Upsampled {len(upsample_indices)} rare tokens.")
            final_indices = np.concatenate(
                [base_indices, np.array(upsample_indices, dtype=np.int64)]
            )
        else:
            final_indices = base_indices
    else:
        final_indices = base_indices

    # Shuffle final set
    rng.shuffle(final_indices)

    print(f"Final dataset size: {len(final_indices)}")

    # 7. Save to Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, final_indices)
    print(f"Indices saved to {cache_path}")

    return final_indices
