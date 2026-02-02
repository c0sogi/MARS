import os
import torch
import pandas as pd
import numpy as np
import csv
from typing import List, Dict, Optional
from torch.utils.data import DataLoader

from library.config import Config, DEVICE, SOS_TOKEN, EOS_TOKEN
from library.utils import (
    is_digit_token,
    safe_load_model,
    ensure_dir,
    load_metadata,
    PAD_TOKEN,
)
from library.symbolic_model import HierarchicalNgram
from library.neural_arch import DualGranularityTransformer
from library.tokenizers import HybridTokenizer
from library.neural_dataset import NormalizationDataset, NormalizationCollator


class HybridRouter:
    """
    The core inference engine that routes tokens between the Symbolic Solver (N-gram)
    and the Neural Solver (Transformer) based on Contextual Precedence Logic.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = DEVICE

        # --- 1. Load Tokenizers ---
        print("Loading Tokenizers...")
        self.tokenizer = HybridTokenizer(config)
        # We assume tokenizers are already trained and exist in the working dir
        # If not, we would need to train them, but inference assumes a trained state.
        self.tokenizer.load(config.bpe_tokenizer_path, config.char_tokenizer_path)

        # --- 2. Load Symbolic Model ---
        print("Loading Symbolic Model...")
        self.symbolic_model = HierarchicalNgram(config)
        # Load stats. If missing, we might need to build them from train data.
        # We try to load cached first.
        try:
            self.symbolic_model.build_stats(load_cached_data=True)
        except FileNotFoundError:
            print("Cached symbolic stats not found. Building from training data...")
            train_df = load_metadata("train", config)
            self.symbolic_model.build_stats(train_df=train_df, load_cached_data=False)

        # --- 3. Load Neural Model ---
        print("Loading Neural Model...")
        self.neural_model = DualGranularityTransformer(
            config,
            bpe_pad_id=self.tokenizer.pad_token_id,
            char_pad_id=self.tokenizer.char_tokenizer.pad_token_id,
        ).to(self.device)

        # Load weights
        if os.path.exists(config.model_checkpoint_path):
            state_dict = safe_load_model(config.model_checkpoint_path, self.device)
            self.neural_model.load_state_dict(state_dict)
            print("Neural model weights loaded successfully.")
        else:
            print(
                f"Warning: Neural model checkpoint not found at {config.model_checkpoint_path}. Using random weights (expect poor performance)."
            )

        self.neural_model.eval()

    def _greedy_decode_batch(
        self, src_left: torch.Tensor, src_target: torch.Tensor, src_right: torch.Tensor
    ) -> List[str]:
        """
        Performs batched greedy decoding for the neural model.
        """
        batch_size = src_left.size(0)
        max_len = self.config.max_seq_len

        # 1. Encode
        with torch.no_grad():
            memory, memory_mask = self.neural_model.encode(
                src_left, src_target, src_right
            )

        # 2. Initialize Decoder Input with SOS
        sos_id = self.tokenizer.bpe_tokenizer.token_to_id(SOS_TOKEN)
        eos_id = self.tokenizer.bpe_tokenizer.token_to_id(EOS_TOKEN)

        # [Batch, 1]
        ys = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=self.device)

        # Track finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(max_len):
            with torch.no_grad():
                out = self.neural_model.decode(
                    ys, memory, memory_key_padding_mask=memory_mask
                )
                # Get last token logits
                prob = out[:, -1, :]
                _, next_word = torch.max(prob, dim=1)

            # Append
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            # Update finished
            finished |= next_word == eos_id
            if finished.all():
                break

        # 3. Decode to Strings
        predictions = []
        ys_list = ys.tolist()
        for row in ys_list:
            tokens = []
            # Skip SOS (index 0)
            for token_id in row[1:]:
                if token_id == eos_id:
                    break
                tokens.append(token_id)
            decoded_str = self.tokenizer.decode(tokens, skip_special_tokens=True)
            predictions.append(decoded_str)

        return predictions

    def generate_submission(self):
        """
        Generates predictions for the test set and saves the submission file.
        Implements the routing logic:
        1. Trigram (Exact Context)
        2. Neural (If Digits present)
        3. Bigram -> Unigram -> Identity
        """
        print("Generating submission...")

        # 1. Load Test Data
        # We load the raw test metadata.
        df_test = load_metadata("test", self.config)

        # Ensure correct sorting
        df_test = df_test.sort_values(["sentence_id", "token_id"]).reset_index(
            drop=True
        )

        # Create ID column for submission
        df_test["id"] = (
            df_test["sentence_id"].astype(str) + "_" + df_test["token_id"].astype(str)
        )

        # Initialize prediction column with None
        df_test["after"] = None

        # --- 2. Prepare Context for Symbolic Lookup ---
        print("Preparing symbolic context...")
        tokens = df_test["before"].astype(str).values
        sent_ids = df_test["sentence_id"].values

        # Vectorized shift for prev/next
        # Prev
        prev_tokens = np.roll(tokens, 1)
        prev_sent = np.roll(sent_ids, 1)
        prev_tokens[sent_ids != prev_sent] = PAD_TOKEN
        prev_tokens[0] = PAD_TOKEN  # Handle first element

        # Next
        next_tokens = np.roll(tokens, -1)
        next_sent = np.roll(sent_ids, -1)
        next_tokens[sent_ids != next_sent] = PAD_TOKEN
        next_tokens[-1] = PAD_TOKEN  # Handle last element

        # --- 3. Priority 1: Trigram Lookup ---
        print("Applying Trigram Lookup...")
        # Create tuples for lookup
        trigram_keys = zip(prev_tokens, tokens, next_tokens)

        # Apply lookup
        trigram_preds = []
        for key in trigram_keys:
            trigram_preds.append(self.symbolic_model.trigram_stats.get(key))

        df_test["trigram_pred"] = trigram_preds

        # Assign Trigram hits to 'after'
        mask_trigram = df_test["trigram_pred"].notna()
        df_test.loc[mask_trigram, "after"] = df_test.loc[mask_trigram, "trigram_pred"]

        print(f"Trigram Coverage: {mask_trigram.mean():.4f}")

        # --- 4. Priority 2: Neural Model (Digits) ---
        # Condition: Not solved by Trigram AND Contains Digits
        mask_unsolved = df_test["after"].isna()
        mask_digits = df_test["before"].astype(str).str.contains(r"\d", regex=True)
        mask_neural = mask_unsolved & mask_digits

        if mask_neural.sum() > 0:
            print(f"Routing {mask_neural.sum()} tokens to Neural Model...")

            # Extract subset for neural processing
            df_neural = df_test[mask_neural].copy()

            # We need to construct context lists for the neural tokenizer
            # We reuse the shifted arrays but need context window +/- 2
            # Re-calculating shifts for the specific indices is tricky,
            # easier to calculate +/- 2 for the whole array and then slice.

            # L2
            l2_tokens = np.roll(tokens, 2)
            l2_sent = np.roll(sent_ids, 2)
            l2_tokens[sent_ids != l2_sent] = PAD_TOKEN
            l2_tokens[0] = PAD_TOKEN
            l2_tokens[1] = PAD_TOKEN

            # R2
            r2_tokens = np.roll(tokens, -2)
            r2_sent = np.roll(sent_ids, -2)
            r2_tokens[sent_ids != r2_sent] = PAD_TOKEN
            r2_tokens[-1] = PAD_TOKEN
            r2_tokens[-2] = PAD_TOKEN

            # Create lists
            # We only need to do this for the neural rows, but slicing numpy arrays is fast
            # Zip into lists
            # Note: This list creation can be slow for 1M rows.
            # Optimization: Only create for mask_neural indices.
            indices = np.where(mask_neural)[0]

            context_left_list = []
            context_right_list = []

            for idx in indices:
                context_left_list.append([l2_tokens[idx], prev_tokens[idx]])
                context_right_list.append([next_tokens[idx], r2_tokens[idx]])

            df_neural["context_left"] = context_left_list
            df_neural["context_right"] = context_right_list

            # Create Dataset and Loader
            neural_dataset = NormalizationDataset(
                df_neural, self.tokenizer, self.config
            )
            collator = NormalizationCollator(
                bpe_pad_id=self.tokenizer.pad_token_id,
                char_pad_id=self.tokenizer.char_tokenizer.pad_token_id,
            )
            neural_loader = DataLoader(
                neural_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=self.config.num_workers,
            )

            # Inference Loop
            neural_preds = []
            neural_ids = []

            for batch in neural_loader:
                src_left = batch["src_left"].to(self.device)
                src_target = batch["src_target"].to(self.device)
                src_right = batch["src_right"].to(self.device)
                ids = batch["id"]

                preds = self._greedy_decode_batch(src_left, src_target, src_right)

                neural_preds.extend(preds)
                neural_ids.extend(ids)

            # Map back to dataframe
            # Create a mapping dict
            neural_map = dict(zip(neural_ids, neural_preds))

            # Update main dataframe
            df_test.loc[mask_neural, "after"] = df_test.loc[mask_neural, "id"].map(
                neural_map
            )

        # --- 5. Priority 3: Backoff (Bigram -> Unigram -> Identity) ---
        # Identify remaining unsolved
        mask_remaining = df_test["after"].isna()
        print(
            f"Processing remaining {mask_remaining.sum()} tokens with Symbolic Backoff..."
        )

        if mask_remaining.sum() > 0:
            df_rem = df_test[mask_remaining].copy()

            # Bigram Lookup
            bigram_keys = zip(prev_tokens[mask_remaining], tokens[mask_remaining])
            bigram_preds = [
                self.symbolic_model.bigram_stats.get(k) for k in bigram_keys
            ]

            # Unigram Lookup
            unigram_preds = [
                self.symbolic_model.unigram_stats.get(k) for k in tokens[mask_remaining]
            ]

            # Identity
            identity_preds = tokens[mask_remaining]

            # Combine logic
            final_preds = []
            for bi, uni, ident in zip(bigram_preds, unigram_preds, identity_preds):
                if bi is not None:
                    final_preds.append(bi)
                elif uni is not None:
                    final_preds.append(uni)
                else:
                    final_preds.append(ident)

            df_test.loc[mask_remaining, "after"] = final_preds

        # --- 6. Save Submission ---
        submission_path = self.config.submission_path
        print(f"Saving submission to {submission_path}...")

        # Format: id, after
        # Ensure quoting for text
        df_submission = df_test[["id", "after"]]
        df_submission.to_csv(
            submission_path,
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,  # Quotes non-numeric fields (id and after are strings)
        )
        print("Submission generated successfully.")
