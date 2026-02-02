import os
import pandas as pd
import torch
import re
from library.config import Config, set_seed
from library.utils import ensure_dir, save_data
from library.hfbb import HFBBModel
from library.vocab import get_tokenizer
from library.model import load_model, predict_beam
from library.trainer import Trainer


class HybridNormalizer:
    """
    Hybrid Text Normalization System.
    Routes tokens to either HFBB (memory-based) or Seq2Seq (neural) models
    based on the presence of digits.
    """

    def __init__(self, load_cached_data=True):
        self.device = Config.DEVICE

        # 1. Initialize and Load Tokenizer
        # This will load from cache or compute from training data
        self.tokenizer = get_tokenizer(load_cached_data=load_cached_data)

        # 2. Initialize and Load HFBB Model
        self.hfbb = HFBBModel()

        # Check if HFBB cache exists
        hfbb_cache_exists = all(
            os.path.exists(p) for p in self.hfbb.cache_files.values()
        )

        # If cache is missing or we force reload, we need training data
        train_df = None
        if not load_cached_data or not hfbb_cache_exists:
            print("HFBB cache missing or reload requested. Loading training data...")
            if os.path.exists(Config.TRAIN_DATA_PATH):
                train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
            else:
                # If training data is missing, we can't fit.
                # We assume cache exists in that case.
                print(
                    "Warning: Training data not found. Relying on existing cache if available."
                )

        self.hfbb.fit(train_df=train_df, load_cached_data=load_cached_data)

        # 3. Initialize and Load Seq2Seq Model
        # Check if model checkpoint exists
        if not os.path.exists(Config.MODEL_CHECKPOINT):
            print(
                f"Model checkpoint not found at {Config.MODEL_CHECKPOINT}. Initiating training..."
            )
            trainer = Trainer(tokenizer=self.tokenizer)
            # Train the model
            trainer.fit(load_cached_data=load_cached_data)
            # Ensure memory is cleared
            del trainer
            torch.cuda.empty_cache()

        # Load the trained model
        self.model = load_model(Config.MODEL_CHECKPOINT, self.tokenizer)
        self.model.to(self.device)
        self.model.eval()

    def _preprocess_context(self, df):
        """
        Generates prev_before and next_before context columns.
        Assumes dataframe is sorted by sentence_id and token_id.
        """
        # Shift to get previous and next tokens
        df["prev_before"] = df["before"].shift(1).fillna("<start>")
        df["next_before"] = df["before"].shift(-1).fillna("<end>")

        # Handle Sentence Boundaries if sentence_id is present
        if "sentence_id" in df.columns:
            # If prev sentence != current sentence, prev_token is <start>
            mask_start = df["sentence_id"] != df["sentence_id"].shift(1)
            df.loc[mask_start, "prev_before"] = "<start>"

            # If next sentence != current sentence, next_token is <end>
            mask_end = df["sentence_id"] != df["sentence_id"].shift(-1)
            df.loc[mask_end, "next_before"] = "<end>"

        return df

    def predict(self, test_df):
        """
        Generates predictions for the test dataframe.
        """
        # Work on a copy
        df = test_df.copy()

        # Ensure sorting for context generation
        if "sentence_id" in df.columns and "token_id" in df.columns:
            df.sort_values(["sentence_id", "token_id"], inplace=True)

        # Generate context
        df = self._preprocess_context(df)

        # Initialize results
        df["after"] = ""

        # Identify Digit Tokens
        # Regex: contains any digit 0-9
        digit_mask = df["before"].astype(str).str.contains(r"\d", regex=True)

        # ---------------------------------------------------------
        # 1. HFBB Inference (Non-Digit Tokens)
        # ---------------------------------------------------------
        non_digit_count = (~digit_mask).sum()
        if non_digit_count > 0:
            print(f"Routing {non_digit_count} tokens to HFBB model...")
            # Extract subset for processing
            non_digit_subset = df[~digit_mask]

            # Apply HFBB prediction
            # Using list comprehension for efficiency
            hfbb_preds = [
                self.hfbb.predict(row.before, row.prev_before, row.next_before)
                for row in non_digit_subset.itertuples(index=False)
            ]

            df.loc[~digit_mask, "after"] = hfbb_preds

        # ---------------------------------------------------------
        # 2. Seq2Seq Inference (Digit Tokens)
        # ---------------------------------------------------------
        digit_count = digit_mask.sum()
        if digit_count > 0:
            print(f"Routing {digit_count} tokens to Seq2Seq model...")
            digit_subset = df[digit_mask]

            seq2seq_preds = []
            sep = Config.SEP_TOKEN

            # Iterate through digit tokens
            for row in digit_subset.itertuples(index=False):
                prev_token = str(row.prev_before)
                curr_token = str(row.before)
                next_token = str(row.next_before)

                # Construct source sequence
                source_text = f"{prev_token}{sep}{curr_token}{sep}{next_token}"

                # Encode
                src_tensor = self.tokenizer.encode(
                    source_text, max_len=Config.MAX_SEQ_LEN, add_special_tokens=True
                )

                # Predict using Beam Search
                try:
                    pred_str = predict_beam(
                        self.model,
                        self.tokenizer,
                        src_tensor,
                        beam_width=Config.BEAM_WIDTH,
                        max_len=Config.MAX_SEQ_LEN,
                    )
                except Exception:
                    # Fallback to identity in case of model error
                    pred_str = curr_token

                seq2seq_preds.append(pred_str)

            df.loc[digit_mask, "after"] = seq2seq_preds

        return df


def generate_submission(load_cached_data=True, debug=False):
    """
    Main function to generate the submission file.
    """
    set_seed(Config.SEED)
    ensure_dir(Config.SUBMISSION_DIR)

    # Load Test Data
    print(f"Loading test data from {Config.TEST_DATA_PATH}...")
    test_df = pd.read_csv(Config.TEST_DATA_PATH)

    # Ensure string types
    test_df["before"] = test_df["before"].fillna("").astype(str)

    # Debug Mode
    if debug:
        print(f"Debug mode enabled. Processing first {Config.DEBUG_SIZE} rows.")
        test_df = test_df.iloc[: Config.DEBUG_SIZE].copy()

    # Initialize Normalizer
    # This handles model training/loading and tokenizer setup
    normalizer = HybridNormalizer(load_cached_data=load_cached_data)

    # Run Inference
    print("Running inference pipeline...")
    result_df = normalizer.predict(test_df)

    # Format for Submission
    # Construct ID if not present (though metadata usually has sentence_id, token_id)
    if "id" not in result_df.columns:
        if "sentence_id" in result_df.columns and "token_id" in result_df.columns:
            result_df["id"] = (
                result_df["sentence_id"].astype(str)
                + "_"
                + result_df["token_id"].astype(str)
            )
        else:
            # Fallback
            result_df["id"] = result_df.index.astype(str)

    submission = result_df[["id", "after"]]

    # Save
    save_path = Config.SUBMISSION_PATH
    print(f"Saving submission to {save_path}...")
    submission.to_csv(save_path, index=False)
    print("Submission generation completed.")
