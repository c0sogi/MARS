import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import seed_everything
from library.data_manager import (
    load_parquet_data,
    get_tokenizer,
    _add_context_columns,
    NormalizationDataset,
    collate_fn,
)
from library.symbolic_solver import SymbolicModel
from library.neural_solver import TransformerSeq2Seq
from library.trainer import train_neural_model


class HybridNormalizer:
    """
    Orchestrates the Hybrid Neuro-Symbolic inference pipeline.
    Combines fast symbolic lookup, heuristic gating, and deep neural transduction.
    """

    def __init__(self, device=Config.DEVICE):
        self.device = device
        self.symbolic_model = None
        self.neural_model = None
        self.tokenizer = None

    def initialize_models(self):
        """
        Loads the symbolic model and the neural model.
        If resources (stats or checkpoints) are missing, triggers generation/training.
        """
        print("Initializing Hybrid Models...")

        # 1. Symbolic Model (Fast, Deterministic)
        # Check if symbolic stats exist to decide if we need to load raw train data
        stats_exist = os.path.exists(
            os.path.join(Config.STATS_CACHE_DIR, "stats_trigram.parquet")
        )

        df_train_for_stats = None
        if not stats_exist:
            print("Symbolic stats not found. Loading training data to build them...")
            df_train_for_stats = load_parquet_data("train")

        # Initialize SymbolicModel (builds or loads stats)
        self.symbolic_model = SymbolicModel(
            df_train=df_train_for_stats, load_cached_data=True
        )

        # 2. Neural Model (Deep Learning)
        # Check if model checkpoint and tokenizer exist
        model_path = Config.MODEL_PATH
        tokenizer_path = os.path.join(Config.WORKING_DIR, "tokenizer.json")

        needs_training = not os.path.exists(model_path) or not os.path.exists(
            tokenizer_path
        )

        if needs_training:
            print(
                "Neural model or tokenizer not found. Triggering training pipeline..."
            )
            self.neural_model, self.tokenizer = train_neural_model(
                load_cached_data=True
            )
        else:
            print(f"Loading neural model from {model_path}...")
            # Load tokenizer first to get vocab size
            self.tokenizer = get_tokenizer(load_cached_data=True)

            # Initialize Architecture
            self.neural_model = TransformerSeq2Seq(
                vocab_size=len(self.tokenizer),
                pad_token_id=self.tokenizer.pad_token_id,
                d_model=Config.D_MODEL,
                nhead=Config.NHEAD,
                num_encoder_layers=Config.NUM_ENCODER_LAYERS,
                num_decoder_layers=Config.NUM_DECODER_LAYERS,
                dim_feedforward=Config.DIM_FEEDFORWARD,
                dropout=Config.DROPOUT,
            )

            # Load Weights
            state_dict = torch.load(model_path, map_location=self.device)
            self.neural_model.load_state_dict(state_dict)

        self.neural_model.to(self.device)
        self.neural_model.eval()
        print("Models initialized successfully.")

    def predict_batch(self, df_test):
        """
        Runs the hybrid inference logic on the test dataframe.

        Strategy:
        1. Vectorized Symbolic Lookup (Trigram -> Bigrams -> Unigram).
        2. Vectorized Heuristic Check (IsAlpha -> Identity).
        3. Neural Inference for remaining 'hard' tokens.
        """
        # Ensure context columns exist
        if "prev_before" not in df_test.columns:
            print("Generating context columns for test set...")
            df_test = _add_context_columns(df_test)

        # Initialize prediction column
        df_test["predicted"] = np.nan

        # ==========================================
        # Stage 1: Symbolic Memory (Vectorized)
        # ==========================================
        print("Running Symbolic Inference...")

        # Prepare keys for lookup
        # Fill NaNs in context with empty string for lookup consistency
        p = df_test["prev_before"].fillna("").astype(str)
        c = df_test["before"].fillna("").astype(str)
        n = df_test["next_before"].fillna("").astype(str)

        # 1. Trigram
        print("  Checking Trigrams...")
        trigram_keys = list(zip(p, c, n))
        df_test["predicted"] = pd.Series(trigram_keys).map(self.symbolic_model.trigram)

        # 2. Bigram Left (fill where NaN)
        mask = df_test["predicted"].isna()
        if mask.any():
            print("  Checking Left Bigrams...")
            bigram_left_keys = list(zip(p[mask], c[mask]))
            df_test.loc[mask, "predicted"] = pd.Series(
                bigram_left_keys, index=df_test.index[mask]
            ).map(self.symbolic_model.bigram_left)

        # 3. Bigram Right
        mask = df_test["predicted"].isna()
        if mask.any():
            print("  Checking Right Bigrams...")
            bigram_right_keys = list(zip(c[mask], n[mask]))
            df_test.loc[mask, "predicted"] = pd.Series(
                bigram_right_keys, index=df_test.index[mask]
            ).map(self.symbolic_model.bigram_right)

        # 4. Unigram
        mask = df_test["predicted"].isna()
        if mask.any():
            print("  Checking Unigrams...")
            df_test.loc[mask, "predicted"] = c[mask].map(self.symbolic_model.unigram)

        # ==========================================
        # Stage 2: Heuristic Gate
        # ==========================================
        print("Running Heuristic Gate...")
        mask = df_test["predicted"].isna()

        # Heuristic: If purely alphabetic, assume identity (regular words/names)
        is_alpha = c.str.isalpha()

        heuristic_mask = mask & is_alpha
        df_test.loc[heuristic_mask, "predicted"] = df_test.loc[heuristic_mask, "before"]

        # ==========================================
        # Stage 3: Neural Inference
        # ==========================================
        neural_mask = df_test["predicted"].isna()
        num_neural = neural_mask.sum()

        if num_neural > 0:
            print(f"Running Neural Inference on {num_neural} tokens...")

            # Extract the subset for neural processing
            df_neural = df_test[neural_mask].copy()

            # Create Dataset and Loader
            # mode='test' implies we only get source tensors
            dataset = NormalizationDataset(
                df_neural, self.tokenizer, max_len=Config.MAX_CHAR_LEN, mode="test"
            )

            loader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                collate_fn=collate_fn,
                pin_memory=(self.device == "cuda"),
            )

            all_preds = []

            with torch.no_grad():
                for batch in loader:
                    # Move to device
                    src = batch.to(self.device)

                    # Predict (Greedy Decode)
                    # Returns tensor of shape (batch, seq_len)
                    generated_ids = self.neural_model.predict(
                        src, self.tokenizer, max_len=Config.MAX_CHAR_LEN
                    )

                    # Decode to strings
                    for i in range(generated_ids.size(0)):
                        seq = generated_ids[i]
                        # Decode
                        text = self.tokenizer.decode(seq, remove_special_tokens=True)
                        all_preds.append(text)

            # Assign back to dataframe
            df_test.loc[neural_mask, "predicted"] = all_preds

        else:
            print("No tokens required neural inference.")

        return df_test


def generate_submission(output_path=None):
    """
    Main entry point for the inference pipeline.
    1. Loads test data.
    2. Initializes HybridNormalizer (training if needed).
    3. Generates predictions.
    4. Saves submission file.
    """
    if output_path is None:
        output_path = Config.SUBMISSION_PATH

    seed_everything(Config.SEED)

    # 1. Load Test Data
    print("Loading test data...")
    df_test = load_parquet_data("test")

    # Ensure sorting for context window logic
    # We sort by sentence_id then token_id to ensure prev/next logic holds
    df_test = df_test.sort_values(["sentence_id", "token_id"]).reset_index(drop=True)

    # 2. Run Inference
    normalizer = HybridNormalizer()
    normalizer.initialize_models()

    df_result = normalizer.predict_batch(df_test)

    # 3. Format Submission
    # Required columns: id, after
    submission = df_result[["id", "predicted"]].rename(columns={"predicted": "after"})

    # Fill any remaining NaNs (safety net) with original text
    num_na = submission["after"].isna().sum()
    if num_na > 0:
        print(f"Warning: {num_na} predictions are NaN. Filling with original text.")
        original_text = df_test.loc[
            submission[submission["after"].isna()].index, "before"
        ]
        submission.loc[submission["after"].isna(), "after"] = original_text

    # 4. Save
    print(f"Saving submission to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission.to_csv(output_path, index=False)

    print("Submission generation complete.")
