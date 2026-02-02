import os
import shutil
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from library.config import Config
from library.utils import set_seed, is_semiotic, save_parquet_cache, load_parquet_cache
from library.hfbb_engine import HFBB
from library.neural_net import (
    ResidualGenerator,
    CharToSubwordTransformer,
    CharTokenizer,
    TargetBPETokenizer,
)
from library.data_utils import CollateFn
from library.training_engine import Trainer


class CascadeManager:
    """
    Orchestrates the Residual-Optimized Hybrid Cascade system.
    Manages data preparation, model training, and hierarchical inference.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        set_seed(Config.SEED)

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

    def generate_residual_dataset(
        self, df: pd.DataFrame, k_folds: int = 5, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Generates a dataset of 'residuals' (cache misses) by performing K-Fold
        Jackknifing on the provided dataframe. This simulates inference conditions
        where the HFBB model encounters unseen data.

        Args:
            df: Input dataframe containing 'sentence_id', 'before', 'after'.
            k_folds: Number of folds for cross-validation.
            load_cached_data: Whether to load from cache if available.

        Returns:
            DataFrame containing only the tokens where HFBB failed or missed.
        """
        cache_path = os.path.join(Config.WORKING_DIR, "manual_residual_dataset.parquet")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached manual residuals from {cache_path}")
            return load_parquet_cache(cache_path)

        print(f"Generating residuals with {k_folds}-Fold Jackknifing...")

        # Ensure we have necessary columns
        if "sentence_id" not in df.columns:
            raise ValueError(
                "Dataframe must contain 'sentence_id' for group splitting."
            )

        sentences = df["sentence_id"].unique()
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=Config.SEED)

        residuals_list = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(sentences)):
            # print(f"  Processing Fold {fold + 1}/{k_folds}...") # Suppressed as per instructions

            train_sents = sentences[train_idx]
            val_sents = sentences[val_idx]

            fold_train = df[df["sentence_id"].isin(train_sents)].copy()
            fold_val = df[df["sentence_id"].isin(val_sents)].copy()

            # Train temporary HFBB on this fold's training set
            temp_hfbb = HFBB()
            # Use a temp directory for this fold's cache to avoid collisions
            temp_dir = os.path.join(Config.WORKING_DIR, f"temp_manual_fold_{fold}")
            os.makedirs(temp_dir, exist_ok=True)
            temp_hfbb.cache_paths = {
                k: os.path.join(temp_dir, os.path.basename(v))
                for k, v in temp_hfbb.cache_paths.items()
            }

            # Fit without loading main cache
            temp_hfbb.fit(fold_train, load_cached_data=False)

            # Predict on validation set (simulating unseen data)
            # Use vectorized prediction for speed
            preds = ResidualGenerator._vectorized_predict(temp_hfbb, fold_val)

            fold_val["hfbb_pred"] = preds.fillna("<MISSING>")

            # Identify residuals: Prediction doesn't match Target
            is_mismatch = fold_val["hfbb_pred"] != fold_val["after"]

            # Filter for semiotic tokens (we only care about normalizing numbers/dates etc.)
            is_semiotic_mask = fold_val["before"].astype(str).apply(is_semiotic)

            fold_residuals = fold_val[is_mismatch & is_semiotic_mask].copy()
            residuals_list.append(fold_residuals)

            # Cleanup temp directory
            shutil.rmtree(temp_dir)

        if residuals_list:
            all_residuals = pd.concat(residuals_list, ignore_index=True)
        else:
            all_residuals = pd.DataFrame(columns=df.columns)

        print(f"Generated {len(all_residuals)} residuals.")
        save_parquet_cache(all_residuals, cache_path)
        return all_residuals

    def train_hfbb(self):
        """
        Trains the Tier 1 HFBB (Memory) model on the full training set.
        """
        print("Training Tier 1: HFBB Memory Model...")
        df_train = pd.read_csv(Config.TRAIN_FILE)
        hfbb = HFBB()
        hfbb.fit(df_train, load_cached_data=False)
        print("HFBB Training Complete.")

    def train_neural(self):
        """
        Trains the Tier 2 Neural Network (Transformer) on residuals.
        Uses the library's Trainer which handles data loading and training loop.
        """
        print("Training Tier 2: Residual Transformer...")
        trainer = Trainer()
        trainer.fit()
        print("Neural Network Training Complete.")

    def train(self):
        """
        Executes the full training pipeline.
        """
        self.train_hfbb()
        self.train_neural()

    def run_inference(
        self,
        test_df: pd.DataFrame,
        hfbb_model: HFBB,
        neural_model: torch.nn.Module,
        char_tokenizer: CharTokenizer,
        bpe_tokenizer: TargetBPETokenizer,
    ) -> pd.Series:
        """
        Executes the Hybrid Cascade Inference logic.

        Strategy:
        1. Tier 1: HFBB Lookup (Fast, High Precision).
        2. Tier 2: Neural Network (Slow, High Generalization) for semiotic cache misses.
        3. Tier 3: Identity Fallback for everything else.

        Args:
            test_df: DataFrame with 'sentence_id', 'token_id', 'before'.
            hfbb_model: Trained HFBB instance.
            neural_model: Trained Transformer instance.
            char_tokenizer: Tokenizer for input characters.
            bpe_tokenizer: Tokenizer for output subwords.

        Returns:
            pd.Series containing the normalized text for each row in test_df.
        """
        print("Running Hybrid Cascade Inference...")

        # Ensure data types
        test_df = test_df.copy()
        test_df["before"] = test_df["before"].astype(str)

        # ==========================================
        # Tier 1: HFBB Memory Lookup
        # ==========================================
        # This handles ~87% of tokens (Plain, Punct) and frequent semiotic tokens
        print("Tier 1: Querying Memory Model...")
        preds = ResidualGenerator._vectorized_predict(hfbb_model, test_df)

        # ==========================================
        # Tier 2: Neural Network for Residuals
        # ==========================================
        # Identify candidates: HFBB yielded NaN AND token is semiotic
        is_miss = preds.isna()
        is_semiotic_mask = test_df["before"].apply(is_semiotic)
        candidate_mask = is_miss & is_semiotic_mask

        candidates = test_df[candidate_mask].copy()
        num_candidates = len(candidates)
        print(f"Tier 2: Found {num_candidates} candidates for Neural Inference.")

        if num_candidates > 0:
            # 1. Prepare Context (Prev/Next) for Candidates
            # We need to reconstruct context respecting sentence boundaries
            # Note: We do this on the full test_df to get correct neighbors, then slice
            test_df["prev"] = test_df["before"].shift(1).fillna("<START>")
            test_df["next"] = test_df["before"].shift(-1).fillna("<END>")

            if "sentence_id" in test_df.columns:
                is_start = test_df["sentence_id"] != test_df["sentence_id"].shift(1)
                test_df.loc[is_start, "prev"] = "<START>"
                is_end = test_df["sentence_id"] != test_df["sentence_id"].shift(-1)
                test_df.loc[is_end, "next"] = "<END>"

            # Extract updated candidates with context
            candidates = test_df.loc[candidate_mask].copy()

            # 2. Tokenize Inputs
            # Construct input string: prev <SEP> curr <SEP> next
            context_window = Config.CONTEXT_WINDOW_CHARS

            input_texts = []
            for _, row in candidates.iterrows():
                prev_ctx = str(row["prev"])[-context_window:]
                curr_tok = str(row["before"])
                next_ctx = str(row["next"])[:context_window]
                input_texts.append(prev_ctx + "<SEP>" + curr_tok + "<SEP>" + next_ctx)

            # Encode
            encoded_inputs = [char_tokenizer.encode(txt) for txt in input_texts]

            # 3. Batch Inference
            batch_size = Config.BATCH_SIZE
            neural_preds = []

            neural_model.eval()

            with torch.no_grad():
                for i in range(0, num_candidates, batch_size):
                    batch_indices = encoded_inputs[i : i + batch_size]

                    # Pad batch
                    src_tensors = [
                        torch.tensor(x, dtype=torch.long) for x in batch_indices
                    ]
                    src_padded = pad_sequence(
                        src_tensors,
                        batch_first=True,
                        padding_value=char_tokenizer.pad_token_id,
                    ).to(self.device)

                    # Generate
                    # predict returns tensor of shape (batch, seq_len)
                    generated_ids = neural_model.predict(
                        src_padded,
                        max_len=Config.MAX_OUTPUT_LEN,
                        bos_id=bpe_tokenizer.bos_id,
                        eos_id=bpe_tokenizer.eos_id,
                    )

                    # Decode
                    for seq in generated_ids:
                        seq_list = seq.tolist()
                        decoded_text = bpe_tokenizer.decode(seq_list)
                        neural_preds.append(decoded_text)

            # 4. Merge Results
            preds.loc[candidate_mask] = neural_preds

        # ==========================================
        # Tier 3: Identity Fallback
        # ==========================================
        # Fill any remaining NaNs (non-semiotic misses) with original text
        preds = preds.fillna(test_df["before"])

        return preds

    def predict_and_submit(self):
        """
        Loads models, runs inference on the test set, and saves the submission file.
        """
        print("Starting Prediction Pipeline...")

        # 1. Load Data
        df_test = pd.read_csv(Config.TEST_FILE)

        # 2. Load HFBB (Tier 1)
        hfbb = HFBB()
        hfbb.fit(load_cached_data=True)

        # 3. Load Neural Model (Tier 2)
        if not os.path.exists(Config.TRANSFORMER_CHECKPOINT):
            raise FileNotFoundError(
                f"Checkpoint not found: {Config.TRANSFORMER_CHECKPOINT}"
            )

        print(f"Loading Neural Model from {Config.TRANSFORMER_CHECKPOINT}")
        checkpoint = torch.load(Config.TRANSFORMER_CHECKPOINT, map_location=self.device)

        # Restore Tokenizers
        char_tokenizer = CharTokenizer()
        char_tokenizer.char2idx = checkpoint["char_vocab"]
        char_tokenizer.idx2char = {v: k for k, v in char_tokenizer.char2idx.items()}
        char_tokenizer.vocab_size = len(char_tokenizer.char2idx)

        bpe_tokenizer = TargetBPETokenizer()
        bpe_tokenizer.load()

        # Restore Model
        model_config = checkpoint["config"]
        model = CharToSubwordTransformer(
            src_vocab_size=model_config["src_vocab_size"],
            tgt_vocab_size=model_config["tgt_vocab_size"],
            pad_idx=char_tokenizer.pad_token_id,
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # 4. Run Inference
        predictions = self.run_inference(
            df_test, hfbb, model, char_tokenizer, bpe_tokenizer
        )

        # 5. Format and Save Submission
        print("Formatting submission...")
        submission = pd.DataFrame(
            {
                "id": df_test["sentence_id"].astype(str)
                + "_"
                + df_test["token_id"].astype(str),
                "after": predictions,
            }
        )

        submission.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
