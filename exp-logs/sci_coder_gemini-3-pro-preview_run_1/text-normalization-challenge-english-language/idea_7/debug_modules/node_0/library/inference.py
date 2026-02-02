import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.data import (
    Vocabulary,
    KnowledgeBase,
    SubmissionDataset,
    process_tagger_data,
    collate_submission,
)
from library.models import AttentionBiLSTMTagger, TransformerSeq2Seq
from library.engine import TaggerEngine, Seq2SeqEngine, generate_submission


class InferencePipeline:
    """
    Orchestrates the inference process for the Text Normalization task.

    Steps:
    1. Load Vocabulary and Knowledge Base.
    2. Load and process Test Data (with caching).
    3. Initialize Models (Tagger and Seq2Seq) and load checkpoints.
    4. Run the hybrid prediction pipeline to generate the submission file.
    """

    def __init__(self):
        self.device = Config.DEVICE
        seed_everything()

    def load_data_resources(self, load_cached_data=True):
        """
        Loads Vocabulary, Knowledge Base, and prepares the Test DataLoader.
        Implements caching for the processed test data.
        """
        # 1. Load Vocabulary
        vocab = Vocabulary()
        if not vocab.load():
            raise FileNotFoundError(
                "Vocabulary files not found. Please ensure the model has been trained."
            )

        # 2. Load Knowledge Base
        kb = KnowledgeBase()
        if not kb.load():
            print(
                "Warning: Knowledge Base not found. Inference will rely solely on models."
            )
            # Initialize empty KB to prevent errors, though performance will degrade
            kb.lookup = {}

        # 3. Load/Process Test Data
        if load_cached_data and os.path.exists(Config.TEST_GROUPED_PATH):
            print(f"Loading Test Grouped Data from cache: {Config.TEST_GROUPED_PATH}")
            df_test_grouped = pd.read_parquet(
                Config.TEST_GROUPED_PATH, engine="pyarrow"
            )
        else:
            print("Processing Test Data...")
            if not os.path.exists(Config.TEST_FILE):
                raise FileNotFoundError(f"Test file not found at {Config.TEST_FILE}")

            # Load raw test CSV
            df_test = pd.read_csv(Config.TEST_FILE, dtype=str, keep_default_na=False)

            # Process using library function
            df_test_grouped = process_tagger_data(df_test, vocab)

            # Save to cache
            os.makedirs(Config.WORKING_DIR, exist_ok=True)
            df_test_grouped.to_parquet(
                Config.TEST_GROUPED_PATH, index=False, engine="pyarrow"
            )
            print(f"Test data processed and saved to {Config.TEST_GROUPED_PATH}")

        # 4. Create Dataset and DataLoader
        test_dataset = SubmissionDataset(df_test_grouped, vocab)

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_submission,
            pin_memory=True if self.device == "cuda" else False,
        )

        return vocab, kb, test_loader

    def run(self):
        print("Starting Inference Pipeline...")

        # 1. Load Data
        vocab, kb, test_loader = self.load_data_resources(load_cached_data=True)

        # 2. Initialize Models
        print("Initializing Models...")
        # Dimensions required for model instantiation
        num_tokens = len(vocab.token2id)
        num_chars = len(vocab.char2id)
        num_classes = len(vocab.class2id)

        tagger_model = AttentionBiLSTMTagger(num_tokens, num_chars, num_classes)
        seq2seq_model = TransformerSeq2Seq(num_chars, num_classes)

        # 3. Initialize Engines and Load Checkpoints
        # We pass None for train/val loaders as they are not needed for inference
        tagger_engine = TaggerEngine(tagger_model, self.device, None, None)
        seq2seq_engine = Seq2SeqEngine(seq2seq_model, self.device, None, None)

        # Load Tagger
        if os.path.exists(Config.TAGGER_MODEL_PATH):
            print(f"Loading Tagger checkpoint from {Config.TAGGER_MODEL_PATH}...")
            tagger_engine.load_checkpoint(Config.TAGGER_MODEL_PATH)
        else:
            raise FileNotFoundError(
                f"Tagger model checkpoint not found at {Config.TAGGER_MODEL_PATH}"
            )

        # Load Seq2Seq
        if os.path.exists(Config.SEQ2SEQ_MODEL_PATH):
            print(f"Loading Seq2Seq checkpoint from {Config.SEQ2SEQ_MODEL_PATH}...")
            seq2seq_engine.load_checkpoint(Config.SEQ2SEQ_MODEL_PATH)
        else:
            print(
                f"Warning: Seq2Seq checkpoint not found at {Config.SEQ2SEQ_MODEL_PATH}. Fallback generation may fail."
            )

        # 4. Generate Submission
        # This function handles the hybrid logic: Tagger -> KB -> Seq2Seq Fallback
        generate_submission(tagger_engine, seq2seq_engine, test_loader, kb, vocab)

        print("Inference Pipeline Completed.")


def run_inference():
    """
    Entry point function to run the inference pipeline.
    """
    pipeline = InferencePipeline()
    pipeline.run()
