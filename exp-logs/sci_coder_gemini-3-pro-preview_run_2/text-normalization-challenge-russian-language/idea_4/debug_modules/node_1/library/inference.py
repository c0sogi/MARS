import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, is_semiotic
from library.hfbb_engine import HFBBModel
from library.transformer_model import Seq2SeqTransformer
from library.transformer_data import CharTokenizer, NormalizationDataset
from library.trainer import fit_transformer


class HybridSystem:
    """
    Orchestrates the Text Normalization pipeline using a Strict-Priority Cascade.
    Priority: HFBB (Memory) -> Transformer (Generative, Gated) -> Identity (Fallback).
    """

    def __init__(self):
        Config.setup()
        self.device = torch.device(Config.DEVICE)
        self.hfbb = HFBBModel()
        self.tokenizer = CharTokenizer()
        self.transformer = None

    def prepare_models(self):
        """
        Ensures all models are loaded and ready for inference.
        Triggers training if checkpoints are missing.
        """
        # 1. Fit/Load HFBB (Tier 1)
        print("Initializing HFBB (Tier 1)...")
        self.hfbb.fit(load_cached_data=True)

        # 2. Check/Train Transformer (Tier 2)
        if not os.path.exists(Config.MODEL_CHECKPOINT) or not os.path.exists(
            Config.VOCAB_PATH
        ):
            print("Transformer checkpoint not found. Initiating training...")
            fit_transformer(load_cached_data=True)

        # 3. Load Transformer
        print("Loading Transformer (Tier 2)...")
        self.tokenizer.load_vocab(Config.VOCAB_PATH)
        vocab_size = len(self.tokenizer)

        self.transformer = Seq2SeqTransformer(
            vocab_size=vocab_size,
            d_model=Config.D_MODEL,
            nhead=Config.NHEAD,
            num_encoder_layers=Config.NUM_ENCODER_LAYERS,
            num_decoder_layers=Config.NUM_DECODER_LAYERS,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=Config.DROPOUT,
            pad_token_id=self.tokenizer.pad_token_id,
        ).to(self.device)

        state_dict = torch.load(Config.MODEL_CHECKPOINT, map_location=self.device)
        self.transformer.load_state_dict(state_dict)
        self.transformer.eval()

    def generate_test_context(self, df):
        """
        Generates 'prev' and 'next' context columns for the test set,
        respecting sentence boundaries.
        """
        df["before"] = df["before"].fillna("").astype(str)

        # Ensure sentence_id exists (should be present in metadata/test.csv)
        if "sentence_id" not in df.columns and "id" in df.columns:
            df["sentence_id"] = df["id"].apply(lambda x: x.split("_")[0])

        s_ids = df["sentence_id"].values
        tokens = df["before"].values

        # Shift tokens
        prev_tokens = np.roll(tokens, 1)
        next_tokens = np.roll(tokens, -1)

        # Shift sentence IDs to detect boundaries
        prev_s_ids = np.roll(s_ids, 1)
        next_s_ids = np.roll(s_ids, -1)

        # Handle boundaries
        prev_tokens[0] = "<START>"
        next_tokens[-1] = "<END>"

        # If sentence ID changed, prev is start
        start_mask = s_ids != prev_s_ids
        prev_tokens[start_mask] = "<START>"

        # If sentence ID changed, next is end
        end_mask = s_ids != next_s_ids
        next_tokens[end_mask] = "<END>"

        df["prev"] = prev_tokens
        df["next"] = next_tokens

        return df

    def batch_greedy_decode(self, src, max_len=128):
        """
        Performs greedy decoding for a batch of source sequences.
        """
        batch_size = src.size(0)
        src = src.to(self.device)

        # Encode
        memory, src_padding_mask = self.transformer.encode(src)

        # Initialize decoder input with SOS
        ys = (
            torch.ones(batch_size, 1)
            .fill_(self.tokenizer.sos_token_id)
            .type(torch.long)
            .to(self.device)
        )

        # Keep track of finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)

        for _ in range(max_len - 1):
            # Decode
            out = self.transformer.decode(ys, memory, src_padding_mask)

            # Get last token probabilities
            prob = out[:, -1]
            _, next_word = torch.max(prob, dim=1)

            # Update ys
            next_word = next_word.unsqueeze(1)
            ys = torch.cat([ys, next_word], dim=1)

            # Check for EOS
            is_eos = next_word.squeeze(1) == self.tokenizer.eos_token_id
            finished = finished | is_eos

            if finished.all():
                break

        return ys

    def generate_submission(self):
        """
        Main inference loop:
        1. Load Test Data.
        2. Apply HFBB (Tier 1).
        3. Identify Semiotic tokens for Transformer (Tier 2).
        4. Run Transformer in batches.
        5. Apply Identity Fallback (Tier 3).
        6. Save submission.
        """
        set_seed(Config.SEED)
        self.prepare_models()

        print(f"Loading test data from {Config.TEST_DATA}...")
        df_test = pd.read_csv(Config.TEST_DATA)

        # Generate context (prev/next)
        df_test = self.generate_test_context(df_test)

        # Initialize predictions array
        predictions = [None] * len(df_test)

        # Lists for Transformer batching
        transformer_indices = []

        print("Running Tier 1 (HFBB) and Gating...")
        tokens = df_test["before"].values
        prevs = df_test["prev"].values
        nexts = df_test["next"].values

        # Pass 1: HFBB & Gating
        for i in range(len(df_test)):
            token = str(tokens[i])
            prev_tok = str(prevs[i])
            next_tok = str(nexts[i])

            # 1. Tier 1: HFBB
            norm = self.hfbb.get_normalization(token, prev_tok, next_tok)
            if norm is not None:
                predictions[i] = norm
                continue

            # 2. Gate
            if is_semiotic(token):
                # Queue for Transformer
                transformer_indices.append(i)
            else:
                # 3. Fallback: Identity
                predictions[i] = token

        # Pass 2: Transformer Batch Processing
        if transformer_indices:
            print(
                f"Running Tier 2 (Transformer) on {len(transformer_indices)} tokens..."
            )

            # Create subset DataFrame
            df_subset = df_test.iloc[transformer_indices].copy()

            # Create Dataset & Loader (reuse NormalizationDataset with is_test=True)
            dataset = NormalizationDataset(
                df_subset, self.tokenizer, max_len=Config.MAX_SEQ_LEN, is_test=True
            )
            loader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE * 2,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
            )

            generated_texts = []

            with torch.no_grad():
                for src in loader:
                    # Run greedy decode
                    out_seqs = self.batch_greedy_decode(src, max_len=Config.MAX_SEQ_LEN)

                    # Convert to strings
                    out_seqs = out_seqs.cpu().tolist()
                    for seq in out_seqs:
                        text = self.tokenizer.decode(seq, skip_special_tokens=True)
                        generated_texts.append(text)

            # Assign back to predictions
            for idx, text in zip(transformer_indices, generated_texts):
                predictions[idx] = text

        # Final safety check (fill any remaining Nones with identity)
        for i in range(len(predictions)):
            if predictions[i] is None:
                predictions[i] = str(tokens[i])

        # Create Submission DataFrame
        print("Formatting submission...")
        # Construct ID: sentence_id + "_" + token_id
        ids = df_test["sentence_id"].astype(str) + "_" + df_test["token_id"].astype(str)

        submission = pd.DataFrame({"id": ids, "after": predictions})

        # Save
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Submission generated successfully.")


def main():
    system = HybridSystem()
    system.generate_submission()
