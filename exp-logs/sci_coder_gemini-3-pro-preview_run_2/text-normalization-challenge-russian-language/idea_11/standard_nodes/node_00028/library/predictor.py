import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import get_device, load_raw_data, cleanup
from library.tokenization import build_tokenizers
from library.hfbb_layer import HFBBModel
from library.transformer_arch import CharToSubwordTransformer
from library.data_manager import _add_context_columns
from library.trainer import greedy_decode


class InferenceEngine:
    """
    Orchestrates the inference pipeline using the Strict-Priority Routing Logic.
    Combines the statistical HFBB model (Tier 1) and the Neural Transformer (Tier 2).
    """

    def __init__(self, load_cached_data: bool = True):
        """
        Initialize the inference engine by loading models and tokenizers.

        Args:
            load_cached_data (bool): Whether to use cached artifacts.
        """
        self.device = get_device()
        print(f"Initializing InferenceEngine on {self.device}...")

        # 1. Load Tokenizers
        print("Loading tokenizers...")
        self.char_tokenizer, self.bpe_tokenizer = build_tokenizers(
            load_cached_data=load_cached_data
        )

        # 2. Load Tier 1: HFBB Model
        print("Loading HFBB Model (Tier 1)...")
        self.hfbb = HFBBModel(load_cached_data=load_cached_data)

        # 3. Load Tier 2: Transformer Model
        print("Loading Transformer Model (Tier 2)...")
        self.src_vocab_size = len(self.char_tokenizer)
        self.tgt_vocab_size = len(self.bpe_tokenizer)

        self.transformer = CharToSubwordTransformer(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            src_pad_idx=self.char_tokenizer.pad_token_id,
            tgt_pad_idx=self.bpe_tokenizer.pad_token_id,
            d_model=Config.D_MODEL,
            nhead=Config.NHEAD,
            num_encoder_layers=Config.NUM_ENCODER_LAYERS,
            num_decoder_layers=Config.NUM_DECODER_LAYERS,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=Config.DROPOUT,
            max_src_len=Config.MAX_SRC_LEN,
            max_tgt_len=Config.MAX_TGT_LEN,
        ).to(self.device)

        # Load Weights
        if os.path.exists(Config.BEST_MODEL_PATH):
            print(f"Loading checkpoint from {Config.BEST_MODEL_PATH}")
            state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
            self.transformer.load_state_dict(state_dict)
        else:
            print("WARNING: No checkpoint found. Transformer will use random weights.")

        self.transformer.eval()

    def _prepare_transformer_batch(self, batch_df):
        """
        Constructs source tensors for a batch of dataframe rows using extended context.
        Format: Prev_2 Prev_1 <SEP> Target <SEP> Next_1 Next_2
        """
        src_batch = []
        sep_id = self.char_tokenizer.sep_token_id

        # Helper to encode text to char IDs
        def encode_text(text):
            return self.char_tokenizer.encode(str(text), add_special_tokens=False)

        space_ids = encode_text(" ")

        for _, row in batch_df.iterrows():
            p2 = row.get("prev_2", "")
            p1 = row.get("prev_1", "")
            curr = row["before"]
            n1 = row.get("next_1", "")
            n2 = row.get("next_2", "")

            src_ids = []
            # Context Left
            if p2:
                src_ids.extend(encode_text(p2) + space_ids)
            if p1:
                src_ids.extend(encode_text(p1))

            src_ids.append(sep_id)

            # Target
            src_ids.extend(encode_text(curr))

            src_ids.append(sep_id)

            # Context Right
            if n1:
                src_ids.extend(encode_text(n1) + space_ids)
            if n2:
                src_ids.extend(encode_text(n2))

            # Truncate
            if len(src_ids) > Config.MAX_SRC_LEN:
                src_ids = src_ids[: Config.MAX_SRC_LEN]

            src_batch.append(torch.tensor(src_ids, dtype=torch.long))

        # Pad sequence
        src_padded = torch.nn.utils.rnn.pad_sequence(
            src_batch, batch_first=True, padding_value=self.char_tokenizer.pad_token_id
        )
        return src_padded

    def generate_submission(self):
        """
        Executes the full inference pipeline on the test set and saves the submission file.
        """
        print("Starting submission generation...")

        # 1. Load and Preprocess Data
        df_test = load_raw_data("test")
        print(f"Test data loaded: {len(df_test)} rows.")

        # Add context columns (prev_1, prev_2, etc.)
        df_test = _add_context_columns(df_test)

        # 2. Initialize Results Array
        results = [None] * len(df_test)
        transformer_indices = []

        # Pre-calculate semiotic mask for fallback logic
        semiotic_mask = (
            df_test["before"]
            .astype(str)
            .str.contains(Config.SEMIOTIC_REGEX, regex=True)
        )

        # Convert columns to lists for faster iteration
        befores = df_test["before"].astype(str).tolist()
        prev1s = df_test["prev_1"].fillna("").astype(str).tolist()
        next1s = df_test["next_1"].fillna("").astype(str).tolist()

        print("Running Tier 1: HFBB Inference...")

        # 3. Tier 1: HFBB Routing
        for i in range(len(df_test)):
            # Attempt to resolve using statistical model
            pred = self.hfbb.query(befores[i], prev1s[i], next1s[i])

            if pred is not None:
                results[i] = pred
            else:
                # Tier 1 failed. Check routing logic for Tier 2.
                if semiotic_mask[i]:
                    # Complex token -> Send to Transformer
                    transformer_indices.append(i)
                else:
                    # Simple/Unknown token -> Identity Fallback
                    results[i] = befores[i]

        resolved_count = len(df_test) - len(transformer_indices)
        print(f"HFBB/Identity resolved {resolved_count} tokens.")
        print(f"Sending {len(transformer_indices)} tokens to Tier 2: Transformer.")

        # 4. Tier 2: Transformer Inference
        if transformer_indices:
            batch_size = Config.BATCH_SIZE
            num_batches = (len(transformer_indices) + batch_size - 1) // batch_size

            for b in range(num_batches):
                batch_idxs = transformer_indices[b * batch_size : (b + 1) * batch_size]

                # Extract sub-dataframe for this batch
                batch_df = df_test.iloc[batch_idxs]

                # Prepare tensors
                src_padded = self._prepare_transformer_batch(batch_df)

                # Run Inference
                with torch.no_grad():
                    generated_ids = greedy_decode(
                        self.transformer,
                        src_padded,
                        max_len=Config.MAX_TGT_LEN,
                        start_symbol=self.bpe_tokenizer.sos_token_id,
                        end_symbol=self.bpe_tokenizer.eos_token_id,
                        device=self.device,
                    )

                # Decode outputs
                for k, g_ids in enumerate(generated_ids):
                    ids_list = g_ids.cpu().tolist()
                    decoded_text = self.bpe_tokenizer.decode(
                        ids_list, skip_special_tokens=True
                    )

                    # Store result
                    original_idx = batch_idxs[k]
                    results[original_idx] = decoded_text

                if (b + 1) % 100 == 0:
                    print(f"Processed {b + 1}/{num_batches} transformer batches.")
                    cleanup()

        # 5. Save Submission
        print("Formatting submission...")
        df_test["after"] = results

        # Create 'id' column as sentence_id + "_" + token_id
        df_test["id"] = (
            df_test["sentence_id"].astype(str) + "_" + df_test["token_id"].astype(str)
        )

        submission_df = df_test[["id", "after"]]

        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        return submission_df
