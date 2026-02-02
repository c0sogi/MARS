import os
import pandas as pd
import numpy as np
import torch
import sentencepiece as spm
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.config import Config
from library.utils import setup_logger, is_semiotic, set_seed
from library.hfbb import HFBB
from library.transformer_model import CharToSubwordTransformer
from library.dataset import NormalizationDataset, CharTokenizer, collate_fn


class CascadeInference:
    """
    Implements the Curriculum-Enriched Residual Hybrid Cascade inference pipeline.
    1. HFBB (Memory) Lookup
    2. Semiotic Gate
    3. Transformer (Generative) Fallback
    4. Identity Fallback
    """

    def __init__(self):
        self.logger = setup_logger("CascadeInference")
        self.device = Config.DEVICE
        set_seed(Config.SEED)

        # 1. Initialize HFBB (Tier 1)
        self.logger.info("Initializing HFBB model...")
        self.hfbb = HFBB()
        # We need to call fit to load the maps from cache.
        # We pass the training metadata path so it can load/compute if needed.
        if os.path.exists(Config.TRAIN_META):
            df_train = pd.read_csv(Config.TRAIN_META)
            self.hfbb.fit(df_train, load_cached_data=True)
        else:
            self.logger.warning(
                "Training metadata not found. HFBB might fail if cache is missing."
            )
            # Attempt to load cache without fitting (relies on cache existence)
            try:
                self.hfbb.fit(pd.DataFrame(), load_cached_data=True)
            except Exception as e:
                self.logger.error(f"Failed to initialize HFBB: {e}")

        # 2. Initialize Transformer (Tier 2)
        self.logger.info("Initializing Transformer model...")
        self.char_vocab_path = os.path.join(Config.WORKING_DIR, "char_vocab.json")
        self.bpe_model_path = Config.BPE_MODEL_PREFIX + ".model"

        # Load Tokenizers to get dimensions
        self.char_tokenizer = CharTokenizer()
        if os.path.exists(self.char_vocab_path):
            self.char_tokenizer.load(self.char_vocab_path)
        else:
            raise FileNotFoundError(f"Char vocab not found at {self.char_vocab_path}")

        if os.path.exists(self.bpe_model_path):
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(self.bpe_model_path)
        else:
            raise FileNotFoundError(f"BPE model not found at {self.bpe_model_path}")

        # Define Dimensions
        src_vocab_size = self.char_tokenizer.vocab_size
        # Must match the buffer used in training
        tgt_vocab_size = self.sp.get_piece_size() + 10

        pad_idx_src = self.char_tokenizer.PAD_ID
        pad_idx_tgt = self.sp.pad_id()
        bos_idx = self.sp.bos_id()
        eos_idx = self.sp.eos_id()

        # Instantiate Model
        self.model = CharToSubwordTransformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            pad_idx_src=pad_idx_src,
            pad_idx_tgt=pad_idx_tgt,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
        )

        # Load Weights
        if os.path.exists(Config.BEST_MODEL_PATH):
            self.logger.info(f"Loading model weights from {Config.BEST_MODEL_PATH}")
            state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            self.logger.warning(
                f"Model checkpoint not found at {Config.BEST_MODEL_PATH}. Using random weights (Expect poor performance)."
            )

        self.model.to(self.device)
        self.model.eval()

    def run_transformer_inference(
        self, df_subset: pd.DataFrame, context_source_path: str
    ) -> pd.Series:
        """
        Runs the transformer on a subset of data.
        """
        if df_subset.empty:
            return pd.Series(dtype=object)

        self.logger.info(f"Running Transformer on {len(df_subset)} tokens...")

        # Create Dataset
        # context_source_path is crucial here to recover prev/next for non-contiguous rows
        dataset = NormalizationDataset(
            data=df_subset,
            bpe_model_path=Config.BPE_MODEL_PREFIX,
            context_source_path=context_source_path,
            char_vocab_path=self.char_vocab_path,
            mode="inference",
        )

        loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        predictions = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Transformer Inference"):
                # batch is src_tensor in inference mode
                src = batch.to(self.device)

                # Predict (Greedy Decode)
                # Output: (batch, seq_len)
                generated_ids = self.model.predict(src)

                # Decode to text
                generated_ids_list = generated_ids.cpu().tolist()
                decoded_texts = self.sp.decode(generated_ids_list)

                predictions.extend(decoded_texts)

        # Return as Series indexed by the subset index
        return pd.Series(data=predictions, index=df_subset.index)

    def predict_batch(self, df: pd.DataFrame, context_source_path: str) -> pd.Series:
        """
        Main inference pipeline.
        """
        # Ensure input is string
        df["before"] = df["before"].fillna("").astype(str)

        # 1. HFBB Lookup
        self.logger.info("Step 1: HFBB Lookup")
        results = self.hfbb.predict_batch(df)

        # 2. Identify candidates for Transformer
        # Condition: HFBB failed (NaN) AND Token is Semiotic (Digits/Latin)
        missing_mask = results.isna()
        semiotic_mask = df["before"].apply(is_semiotic)
        transformer_mask = missing_mask & semiotic_mask

        df_transformer = df[transformer_mask].copy()

        self.logger.info(
            f"Step 2: Routing. HFBB Found: {(~missing_mask).sum()}. Transformer Queue: {len(df_transformer)}."
        )

        # 3. Transformer Inference
        if not df_transformer.empty:
            trans_preds = self.run_transformer_inference(
                df_transformer, context_source_path
            )
            # Update results
            results.loc[transformer_mask] = trans_preds

        # 4. Fallback (Identity)
        # Fill remaining NaNs with original 'before' text
        remaining_mask = results.isna()
        if remaining_mask.any():
            self.logger.info(
                f"Step 3: Identity Fallback for {remaining_mask.sum()} tokens."
            )
            results.loc[remaining_mask] = df.loc[remaining_mask, "before"]

        return results

    def generate_submission(self):
        """
        Generates the submission file for the test set.
        """
        self.logger.info("Loading test data...")
        df_test = pd.read_csv(Config.TEST_META)

        self.logger.info(f"Test Data Shape: {df_test.shape}")

        # Run Prediction
        # We pass Config.TEST_META as context source so the dataset can recover context
        predictions = self.predict_batch(df_test, context_source_path=Config.TEST_META)

        # Format Submission
        self.logger.info("Formatting submission...")
        submission = pd.DataFrame()

        # Construct ID: sentence_id + "_" + token_id
        submission["id"] = (
            df_test["sentence_id"].astype(str) + "_" + df_test["token_id"].astype(str)
        )
        submission["after"] = predictions.values

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

        # Print head for verification
        print(submission.head())
