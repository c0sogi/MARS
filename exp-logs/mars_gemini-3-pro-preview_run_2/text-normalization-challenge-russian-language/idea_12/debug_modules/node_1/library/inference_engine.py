import os
import torch
import pandas as pd
import numpy as np
import re
from torch.utils.data import DataLoader, Dataset
from library.config import (
    ModelConfig,
    DEVICE,
    CHECKPOINT_DIR,
    TOKENIZER_DIR,
    SUBMISSION_PATH,
    SEMIOTIC_REGEX,
    TRAIN_META_PATH,
    CACHE_DIR,
)
from library.text_utils import (
    CharTokenizer,
    train_bpe_tokenizer,
    format_context_window,
    SEP_TOKEN,
)
from library.hfbb_engine import HFBBModel
from library.transformer_model import CharToBPESeq2Seq

# =============================================================================
# INFERENCE DATASET
# =============================================================================


class InferenceDataset(Dataset):
    """
    Dataset for Transformer inference.
    Returns encoder inputs for a list of context-formatted strings.
    """

    def __init__(self, input_texts, char_tokenizer, max_enc_len=128):
        self.input_texts = input_texts
        self.char_tokenizer = char_tokenizer
        self.max_enc_len = max_enc_len

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        text = self.input_texts[idx]
        enc_ids = self.char_tokenizer.encode(text, max_len=self.max_enc_len)
        return torch.tensor(enc_ids, dtype=torch.long)


# =============================================================================
# HYBRID NORMALIZER ENGINE
# =============================================================================


class HybridNormalizer:
    """
    Orchestrates the Density-Maximized Confidence-Gated Hybrid Cascade.

    Pipeline:
    1. Preprocessing: Generate context windows for the test set.
    2. Tier 1 (HFBB): Apply statistical lookup with confidence gating.
    3. Routing: Identify ambiguous semiotic tokens.
    4. Tier 2 (Transformer): Run neural inference on routed tokens.
    5. Assembly: Merge results and handle fallbacks.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.hfbb = None
        self.transformer = None
        self.char_tokenizer = None
        self.bpe_tokenizer = None

    def load_resources(self):
        """
        Loads tokenizers, HFBB model, and Transformer checkpoint.
        """
        print("Loading resources for inference...")

        # 1. Load Tokenizers
        # We assume these were created during training/setup
        self.char_tokenizer = CharTokenizer()
        char_vocab_path = os.path.join(TOKENIZER_DIR, "char_vocab.json")
        if os.path.exists(char_vocab_path):
            self.char_tokenizer.load(char_vocab_path)
        else:
            raise FileNotFoundError(
                f"Character vocabulary not found at {char_vocab_path}"
            )

        # Load BPE (SentencePiece)
        # train_bpe_tokenizer will load existing model if available
        # We pass an empty DF because we expect the model to exist
        self.bpe_tokenizer = train_bpe_tokenizer(
            pd.DataFrame({"after": []}),
            vocab_size=self.config.bpe_vocab_size,
            load_cached_data=True,
        )

        # 2. Load HFBB Model
        self.hfbb = HFBBModel(self.config)
        # We pass an empty DF; HFBBModel.build will load from parquet cache
        # If cache is missing, this will fail, which is expected for inference-only run
        # strictly following the provided architecture.
        self.hfbb.build(pd.DataFrame(), load_cached_data=True)

        # 3. Load Transformer Model
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "transformer_best.pth")

        self.transformer = CharToBPESeq2Seq(
            self.config,
            char_vocab_size=self.char_tokenizer.vocab_size,
            bpe_vocab_size=len(self.bpe_tokenizer),  # SP vocab size
            src_pad_idx=0,
            tgt_pad_idx=0,
        ).to(DEVICE)

        if os.path.exists(checkpoint_path):
            print(f"Loading Transformer checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=DEVICE)
            self.transformer.load_state_dict(state_dict)
            self.transformer.eval()
        else:
            print(
                f"Warning: Checkpoint {checkpoint_path} not found. Transformer will use random weights."
            )

    def _greedy_decode(self, src_batch):
        """
        Performs greedy decoding for a batch of source sequences.
        """
        batch_size = src_batch.size(0)
        max_len = self.config.max_dec_len

        # Special Tokens for SentencePiece (assumed from config/training)
        # BOS=2, EOS=3
        bos_idx = 2
        eos_idx = 3

        src_batch = src_batch.to(DEVICE)

        # Encode
        memory, src_mask = self.transformer.encode(src_batch)

        # Initialize decoder input with BOS
        ys = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=DEVICE)

        # Track finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)

        for _ in range(max_len):
            # Decode
            out = self.transformer.decode(ys, memory, src_mask)

            # Get next token (argmax)
            # out is [batch, seq_len, vocab]
            prob = out[:, -1, :]
            _, next_word = torch.max(prob, dim=1)

            # Check for EOS
            is_eos = next_word == eos_idx
            finished = finished | is_eos

            # Append to sequence
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            if finished.all():
                break

        return ys

    def predict(self, df_test: pd.DataFrame):
        """
        Runs the full prediction pipeline on the test dataframe.
        Generates submission.csv.
        """
        print(f"Starting inference on {len(df_test)} samples...")

        # Ensure ID column exists
        if "id" not in df_test.columns:
            df_test["id"] = (
                df_test["sentence_id"].astype(str)
                + "_"
                + df_test["token_id"].astype(str)
            )

        # Initialize predictions with None
        # We will fill this array progressively
        predictions = np.array([None] * len(df_test), dtype=object)

        # Convert columns to string for processing
        tokens = df_test["before"].fillna("").astype(str).values
        sentence_ids = df_test["sentence_id"].values

        # =========================================================================
        # TIER 1: HFBB (Statistical Lookup)
        # =========================================================================
        print("Running Tier 1: HFBB...")

        # Pre-calculate context (Prev/Next)
        # Using pandas shift is fast
        df_context = pd.DataFrame({"token": tokens, "sent": sentence_ids})

        df_context["prev"] = df_context["token"].shift(1).fillna("")
        df_context.loc[df_context["sent"] != df_context["sent"].shift(1), "prev"] = ""

        df_context["next"] = df_context["token"].shift(-1).fillna("")
        df_context.loc[df_context["sent"] != df_context["sent"].shift(-1), "next"] = ""

        prev_tokens = df_context["prev"].values
        next_tokens = df_context["next"].values

        # Vectorized lookup is hard with dicts, so we iterate
        # Optimization: Use list comprehension
        hfbb_preds = []
        for t, p, n in zip(tokens, prev_tokens, next_tokens):
            res = self.hfbb.query(t, p, n)
            hfbb_preds.append(res)

        predictions = np.array(hfbb_preds, dtype=object)

        # Calculate coverage
        filled_mask = predictions != None
        print(f"Tier 1 Coverage: {np.mean(filled_mask):.2%}")

        # =========================================================================
        # ROUTING LOGIC
        # =========================================================================
        # Identify tokens that need Transformer
        # Criteria: Prediction is None AND Token is Semiotic (contains digit or latin)

        is_semiotic = pd.Series(tokens).str.contains(SEMIOTIC_REGEX, regex=True).values
        needs_transformer_mask = (~filled_mask) & is_semiotic

        transformer_indices = np.where(needs_transformer_mask)[0]
        print(f"Routing {len(transformer_indices)} tokens to Tier 2 (Transformer)...")

        # =========================================================================
        # TIER 2: TRANSFORMER
        # =========================================================================

        if len(transformer_indices) > 0:
            # 1. Prepare Context Windows for these indices
            # We need to run format_context_window on the whole DF to get correct context,
            # then slice. Or slice the DF and ensure format_context_window handles it.
            # format_context_window requires contiguous sentence structure.
            # Efficient approach: Generate context strings for ALL, then subset.
            # Since format_context_window is vectorized string op, it's fast enough for 1M rows.

            print("Formatting context windows...")
            all_input_texts = format_context_window(
                df_test, context_window=self.config.context_window
            )
            target_input_texts = all_input_texts[transformer_indices]

            # 2. Dataset & DataLoader
            inf_ds = InferenceDataset(
                target_input_texts,
                self.char_tokenizer,
                max_enc_len=self.config.max_enc_len,
            )
            inf_loader = DataLoader(
                inf_ds,
                batch_size=self.config.batch_size * 2,
                shuffle=False,
                num_workers=self.config.num_workers,
            )

            # 3. Inference Loop
            generated_texts = []

            with torch.no_grad():
                for batch_src in inf_loader:
                    # Greedy Decode
                    # output: [batch, seq_len]
                    batch_out = self._greedy_decode(batch_src)

                    # Decode BPE to String
                    batch_out_list = batch_out.cpu().tolist()
                    for seq in batch_out_list:
                        # Remove BOS (index 0 in sequence usually, but we appended)
                        # SentencePiece DecodeIds ignores special tokens usually, but let's be safe
                        # Filter out BOS=2, EOS=3, PAD=0
                        filtered_seq = [idx for idx in seq if idx not in [0, 1, 2, 3]]
                        text = self.bpe_tokenizer.DecodeIds(filtered_seq)
                        generated_texts.append(text)

            # 4. Update Predictions
            predictions[transformer_indices] = generated_texts

        # =========================================================================
        # FALLBACK & FINALIZATION
        # =========================================================================

        # Fill remaining Nones with original text (Identity)
        # These are non-semiotic tokens that HFBB didn't explicitly cache (e.g. rare proper nouns, punctuation)
        final_mask = predictions == None
        predictions[final_mask] = tokens[final_mask]

        # Create Submission DataFrame
        df_submission = pd.DataFrame({"id": df_test["id"], "after": predictions})

        # Post-processing: Ensure "after" is string
        df_submission["after"] = df_submission["after"].fillna("").astype(str)

        # Save
        print(f"Saving submission to {SUBMISSION_PATH}")
        df_submission.to_csv(
            SUBMISSION_PATH, index=False, quoting=1
        )  # quote all non-numeric

        return df_submission


def run_inference(debug=False, subset_size=None):
    """
    Main entry point for the inference module.
    """
    # 1. Config
    config = ModelConfig(debug=debug)
    if subset_size:
        config.subset_size = subset_size

    # 2. Load Test Data
    print(
        f"Loading test metadata from {os.path.join(os.path.dirname(TRAIN_META_PATH), 'test.csv')}"
    )
    # Note: Using TEST_META_PATH from config if available, else constructing
    test_path = os.path.join(os.path.dirname(TRAIN_META_PATH), "test.csv")
    df_test = pd.read_csv(test_path)

    if debug and subset_size:
        print(f"Debug: Subsetting test data to {subset_size} rows.")
        df_test = df_test.head(subset_size)

    # 3. Initialize Engine
    engine = HybridNormalizer(config)
    engine.load_resources()

    # 4. Run Prediction
    engine.predict(df_test)
    print("Inference complete.")
