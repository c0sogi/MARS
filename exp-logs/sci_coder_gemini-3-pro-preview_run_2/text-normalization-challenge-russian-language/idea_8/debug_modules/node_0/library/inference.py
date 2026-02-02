import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from library.config import Config
from library.utils import get_device, set_seed
from library.text_processing import build_tokenizers, is_semiotic
from library.hfbb_engine import HFBBModel
from library.transformer_model import get_model, generate_square_subsequent_mask


class InferenceDataset(Dataset):
    """
    Dataset for Transformer inference.
    """

    def __init__(self, df, char_tokenizer, max_len=128):
        self.df = df.reset_index(drop=True)
        self.char_tokenizer = char_tokenizer
        self.max_len = max_len
        self.sep_str = "<SEP>"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Construct input text: prev <SEP> before <SEP> next
        input_text = (
            f"{row['prev']}{self.sep_str}{row['before']}{self.sep_str}{row['next']}"
        )

        parts = input_text.split(self.sep_str)
        src_indices = []

        # Add SOS
        src_indices.append(self.char_tokenizer.sos_token_id)

        for i, part in enumerate(parts):
            part_indices = self.char_tokenizer.encode(part, add_special_tokens=False)
            src_indices.extend(part_indices)
            if i < len(parts) - 1:
                src_indices.append(self.char_tokenizer.sep_token_id)

        # Add EOS
        src_indices.append(self.char_tokenizer.eos_token_id)

        # Truncate
        if len(src_indices) > self.max_len:
            src_indices = src_indices[: self.max_len]
            src_indices[-1] = self.char_tokenizer.eos_token_id

        return torch.tensor(src_indices, dtype=torch.long), idx


def inference_collate_fn(batch):
    src_batch = [item[0] for item in batch]
    indices = [item[1] for item in batch]
    # Pad with 0 (assuming CharTokenizer PAD is 0, which is standard in this setup)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    return src_padded, indices


class HybridPredictor:
    def __init__(self):
        self.device = get_device()
        self.hfbb = HFBBModel()
        self.char_tokenizer = None
        self.target_tokenizer = None
        self.transformer = None
        self.sep = " "  # HFBB separator

    def load_resources(self):
        print("HybridPredictor: Loading resources...")
        # 1. Load HFBB
        self.hfbb.build(load_cached_data=True)

        # 2. Load Tokenizers
        # We assume they exist from training phase
        self.char_tokenizer, self.target_tokenizer = build_tokenizers(
            load_cached_data=True
        )

        # 3. Load Transformer
        if os.path.exists(Config.MODEL_BEST_PATH):
            print(f"HybridPredictor: Loading Transformer from {Config.MODEL_BEST_PATH}")
            self.transformer = get_model(
                src_vocab_size=self.char_tokenizer.vocab_size,
                tgt_vocab_size=self.target_tokenizer.vocab_size,
                device=self.device,
            )
            state_dict = torch.load(Config.MODEL_BEST_PATH, map_location=self.device)
            self.transformer.load_state_dict(state_dict)
            self.transformer.eval()
        else:
            print(
                "HybridPredictor: WARNING - Transformer checkpoint not found. Will fallback to HFBB/Identity."
            )

    def _prepare_context(self, df):
        print("HybridPredictor: Generating context...")
        df = df.copy()
        df["before"] = df["before"].fillna("").astype(str)
        # Sort by sentence_id, token_id to ensure order
        # Assuming token_id is numerical or parseable
        df["token_id_int"] = df["token_id"].astype(int)
        df = df.sort_values(["sentence_id", "token_id_int"])

        # Shift
        df["prev"] = df.groupby("sentence_id")["before"].shift(1).fillna("<SOS>")
        df["next"] = df.groupby("sentence_id")["before"].shift(-1).fillna("<EOS>")

        return df

    def _run_hfbb_vectorized(self, df):
        print("HybridPredictor: Running vectorized HFBB...")
        # Prepare keys
        df["key_tri"] = df["prev"] + self.sep + df["before"] + self.sep + df["next"]
        df["key_bp"] = df["prev"] + self.sep + df["before"]
        df["key_bn"] = df["before"] + self.sep + df["next"]

        results = pd.DataFrame(index=df.index)
        results["pred"] = None
        results["confidence"] = 0.0
        results["source"] = "NONE"

        # 1. Trigram
        if self.hfbb.trigram_map is not None:
            # Join
            merged = df.merge(
                self.hfbb.trigram_map,
                left_on="key_tri",
                right_index=True,
                how="left",
                suffixes=("", "_tri"),
            )
            mask = merged["after"].notna()
            results.loc[mask, "pred"] = merged.loc[mask, "after"]
            results.loc[mask, "confidence"] = 1.0
            results.loc[mask, "source"] = "TRIGRAM"

        # Remaining
        remaining_mask = results["pred"].isna()

        # 2. Bigram Prev
        if self.hfbb.bigram_prev_map is not None and remaining_mask.any():
            subset = df.loc[remaining_mask]
            merged = subset.merge(
                self.hfbb.bigram_prev_map,
                left_on="key_bp",
                right_index=True,
                how="left",
                suffixes=("", "_bp"),
            )
            mask = merged["after"].notna()
            # Update results
            idx_to_update = merged.loc[mask].index
            results.loc[idx_to_update, "pred"] = merged.loc[mask, "after"]
            results.loc[idx_to_update, "confidence"] = 1.0
            results.loc[idx_to_update, "source"] = "BIGRAM_PREV"

        remaining_mask = results["pred"].isna()

        # 3. Bigram Next
        if self.hfbb.bigram_next_map is not None and remaining_mask.any():
            subset = df.loc[remaining_mask]
            merged = subset.merge(
                self.hfbb.bigram_next_map,
                left_on="key_bn",
                right_index=True,
                how="left",
                suffixes=("", "_bn"),
            )
            mask = merged["after"].notna()
            idx_to_update = merged.loc[mask].index
            results.loc[idx_to_update, "pred"] = merged.loc[mask, "after"]
            results.loc[idx_to_update, "confidence"] = 1.0
            results.loc[idx_to_update, "source"] = "BIGRAM_NEXT"

        remaining_mask = results["pred"].isna()

        # 4. Unigram
        if self.hfbb.unigram_map is not None and remaining_mask.any():
            subset = df.loc[remaining_mask]
            merged = subset.merge(
                self.hfbb.unigram_map,
                left_on="before",
                right_index=True,
                how="left",
                suffixes=("", "_uni"),
            )
            mask = merged["after"].notna()
            idx_to_update = merged.loc[mask].index
            results.loc[idx_to_update, "pred"] = merged.loc[mask, "after"]
            results.loc[idx_to_update, "confidence"] = merged.loc[mask, "confidence"]
            results.loc[idx_to_update, "source"] = "UNIGRAM"

        return results

    def _greedy_decode(self, src, max_len=128):
        """
        Greedy decoding for a batch of source sequences.
        """
        batch_size = src.size(0)
        src = src.to(self.device)

        # Create masks
        # src padding mask: assuming 0 is pad
        src_padding_mask = src == 0
        # src_mask: usually zeros for encoder
        src_mask = torch.zeros((src.size(1), src.size(1)), device=self.device).type(
            torch.bool
        )

        # Encode
        memory = self.transformer.encode(src, src_mask)

        # Initialize decoder input with BOS
        bos_id = self.target_tokenizer.bos_id
        eos_id = self.target_tokenizer.eos_id

        ys = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=self.device)

        # Track finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(max_len):
            tgt_mask = generate_square_subsequent_mask(ys.size(1), self.device)

            # Decode
            out = self.transformer.decode(ys, memory, tgt_mask)

            # Generator
            prob = self.transformer.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)

            # Update finished status
            finished |= next_word == eos_id

            # Append
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            if finished.all():
                break

        return ys

    def predict(self, test_df):
        # 1. Context
        df = self._prepare_context(test_df)

        # 2. HFBB
        hfbb_results = self._run_hfbb_vectorized(df)

        # 3. Routing Logic
        # Identify Semiotics
        df["is_semiotic"] = df["before"].apply(is_semiotic)

        # Conditions for Transformer:
        # A: Unigram Match BUT Low Confidence AND Semiotic
        cond_ambiguous = (
            (hfbb_results["source"] == "UNIGRAM")
            & (hfbb_results["confidence"] <= Config.HFBB_CONFIDENCE_THRESHOLD)
            & (df["is_semiotic"])
        )

        # B: No Match (OOV) AND Semiotic
        cond_oov_semiotic = (hfbb_results["source"] == "NONE") & (df["is_semiotic"])

        mask_transformer = cond_ambiguous | cond_oov_semiotic

        # 4. Run Transformer
        if mask_transformer.any() and self.transformer is not None:
            print(
                f"HybridPredictor: Routing {mask_transformer.sum()} tokens to Transformer..."
            )

            transformer_subset = df[mask_transformer].copy()
            dataset = InferenceDataset(
                transformer_subset, self.char_tokenizer, max_len=Config.MAX_SEQ_LEN
            )
            dataloader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                collate_fn=inference_collate_fn,
                num_workers=Config.NUM_WORKERS,
            )

            transformer_preds = {}  # index -> prediction string

            with torch.no_grad():
                for src, indices in dataloader:
                    generated_ids = self._greedy_decode(src, max_len=Config.MAX_SEQ_LEN)

                    # Decode to string
                    generated_ids = generated_ids.cpu().tolist()
                    decoded_strs = self.target_tokenizer.sp.DecodeIds(generated_ids)

                    for idx, txt in zip(indices, decoded_strs):
                        transformer_preds[idx] = txt

            # Assign back to results
            # Note: indices in dataset correspond to iloc in transformer_subset
            # We need to map back to original df index
            # inference_collate_fn returns the index from __getitem__ which is iloc relative to subset?
            # No, __getitem__ uses self.df.iloc[idx]. self.df is transformer_subset reset_index.
            # So indices are 0..N of the subset.
            # We need to map these back to original indices.

            subset_indices = transformer_subset.index  # Original indices

            # Update hfbb_results with transformer predictions
            # We iterate through the subset and update
            # Since dataloader preserves order (shuffle=False), we can just assign list

            # Flatten predictions list ordered by iteration
            all_preds = []
            # We need to be careful with batching. The loop above fills transformer_preds dict with key=batch_index
            # But batch_index is 0..batch_size.
            # Let's collect all decoded strings in a list

            collected_preds = []
            with torch.no_grad():
                for src, _ in dataloader:
                    generated_ids = self._greedy_decode(src, max_len=Config.MAX_SEQ_LEN)
                    decoded_strs = self.target_tokenizer.sp.DecodeIds(
                        generated_ids.cpu().tolist()
                    )
                    collected_preds.extend(decoded_strs)

            hfbb_results.loc[mask_transformer, "pred"] = collected_preds
            hfbb_results.loc[mask_transformer, "source"] = "TRANSFORMER"

        # 5. Fallback / Finalize
        # If pred is still None, use Identity (fallback for OOV non-semiotic or if Transformer failed)
        mask_none = hfbb_results["pred"].isna()
        hfbb_results.loc[mask_none, "pred"] = df.loc[mask_none, "before"]
        hfbb_results.loc[mask_none, "source"] = "IDENTITY"

        return hfbb_results["pred"]


def generate_submission(load_cached_data=True):
    """
    Main function to generate submission.
    """
    set_seed()
    Config.setup_dirs()

    # 1. Load Data
    print("Inference: Loading test data...")
    test_df = pd.read_csv(Config.TEST_FILE)

    # 2. Initialize Predictor
    predictor = HybridPredictor()
    predictor.load_resources()

    # 3. Predict
    predictions = predictor.predict(test_df)

    # 4. Format Submission
    print("Inference: Formatting submission...")
    submission = pd.DataFrame()
    submission["id"] = (
        test_df["sentence_id"].astype(str) + "_" + test_df["token_id"].astype(str)
    )
    submission["after"] = predictions.values

    # 5. Save
    print(f"Inference: Saving to {Config.SUBMISSION_PATH}...")
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Inference: Done.")

    return submission
