import sys
import os
import pandas as pd
import torch
import logging

# Ensure library is in path
sys.path.append(".")

# Import library modules
from library.config import Config, set_seed
from library.utils import get_logger
from library.vocab_manager import build_vocabs
from library.feature_engineering import FeatureEngineer
from library.knowledge_base import KnowledgeBase
from library.datasets import TaggerDataset, Seq2SeqDataset
from library.models import GatedBiLSTMTagger, Seq2SeqFallback
from library.trainer import ModelTrainer
from library.inference import Predictor

# Setup Logger
logger = get_logger("demo_script")


def run_demo():
    # 1. Configuration Setup
    logger.info("--- 1. Configuring Environment ---")

    # Modify Config for speed/demo purposes
    # We use class attribute modification to propagate settings globally
    Config.DEBUG = True
    Config.DEBUG_SIZE = 5000  # Small subset for speed
    Config.TAGGER_EPOCHS = 1
    Config.SEQ2SEQ_EPOCHS = 1
    Config.BATCH_SIZE = 64
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    Config.BPE_VOCAB_SIZE = 2000  # Reduce vocab size for small debug dataset

    # Setup directories
    Config.setup()
    set_seed(Config.SEED)

    # 2. Vocabulary Generation
    logger.info("\n--- 2. Building Vocabularies ---")
    # Force build from scratch to demonstrate logic (load_cached_data=False)
    # This will train BPE on the debug subset of the training data
    word_vocab, char_vocab, class_vocab, bpe_tokenizer = build_vocabs(
        load_cached_data=False
    )

    logger.info(f"Word Vocab Size: {len(word_vocab)}")
    logger.info(f"Char Vocab Size: {len(char_vocab)}")
    logger.info(f"Class Vocab Size: {len(class_vocab)}")
    logger.info(f"BPE Vocab Size: {len(bpe_tokenizer)}")

    # Validation
    assert len(word_vocab) > 0
    assert len(class_vocab) > 0

    # 3. Feature Engineering & Knowledge Base
    logger.info("\n--- 3. Feature Engineering & Knowledge Base ---")
    fe = FeatureEngineer()

    # Test Regex Features
    sample_tokens = ["123", "hello", "$5.00", "2020-01-01"]
    regex_feats = fe.extract_regex_features(sample_tokens)
    logger.info(f"Regex Features Shape: {regex_feats.shape}")
    assert regex_feats.shape == (4, Config.NUM_REGEX_FEATURES)

    # Priors (Compute from scratch)
    priors_df = fe.build_or_load_priors(class_vocab, load_cached_data=False)
    logger.info(f"Priors Shape: {priors_df.shape}")

    # Knowledge Base (Build from scratch)
    kb = KnowledgeBase()
    kb.build(load_cached_data=False)
    logger.info(f"KB Size: {len(kb.lookup_table)}")

    # 4. Dataset Preparation
    logger.info("\n--- 4. Dataset Preparation ---")

    # Tagger Dataset
    train_tagger_ds = TaggerDataset(
        data_path=Config.TRAIN_FILE,
        word_vocab=word_vocab,
        char_vocab=char_vocab,
        class_vocab=class_vocab,
        bpe_tokenizer=bpe_tokenizer,
        feature_engineer=fe,
        priors_df=priors_df,
        split="train",
        load_cached_data=False,
        debug=True,
    )

    val_tagger_ds = TaggerDataset(
        data_path=Config.VAL_FILE,
        word_vocab=word_vocab,
        char_vocab=char_vocab,
        class_vocab=class_vocab,
        bpe_tokenizer=bpe_tokenizer,
        feature_engineer=fe,
        priors_df=priors_df,
        split="val",
        load_cached_data=False,
        debug=True,
    )

    logger.info(f"Tagger Train Size: {len(train_tagger_ds)}")

    # Seq2Seq Dataset
    train_seq2seq_ds = Seq2SeqDataset(
        data_path=Config.TRAIN_FILE,
        char_vocab=char_vocab,
        class_vocab=class_vocab,
        split="train",
        load_cached_data=False,
        debug=True,
    )

    val_seq2seq_ds = Seq2SeqDataset(
        data_path=Config.VAL_FILE,
        char_vocab=char_vocab,
        class_vocab=class_vocab,
        split="val",
        load_cached_data=False,
        debug=True,
    )

    logger.info(f"Seq2Seq Train Size: {len(train_seq2seq_ds)}")

    # 5. Model Training
    logger.info("\n--- 5. Model Training ---")
    trainer = ModelTrainer(device=Config.DEVICE)

    # Train Tagger
    logger.info("Training Tagger...")
    tagger_model = trainer.train_tagger(
        train_dataset=train_tagger_ds,
        val_dataset=val_tagger_ds,
        word_vocab=word_vocab,
        bpe_tokenizer=bpe_tokenizer,
        char_vocab=char_vocab,
        class_vocab=class_vocab,
    )

    # Train Seq2Seq
    # Only train if we have data (might be empty if no changes in debug subset)
    if len(train_seq2seq_ds) > 0:
        logger.info("Training Seq2Seq...")
        seq2seq_model = trainer.train_seq2seq(
            train_dataset=train_seq2seq_ds,
            val_dataset=val_seq2seq_ds,
            char_vocab=char_vocab,
            class_vocab=class_vocab,
        )
    else:
        logger.warning(
            "Skipping Seq2Seq training due to empty dataset (no changes in debug subset)."
        )
        # Create a dummy model and save it so inference doesn't fail
        seq2seq_model = Seq2SeqFallback(
            char_vocab_size=len(char_vocab),
            class_vocab_size=len(class_vocab),
            sos_idx=char_vocab["<sos>"],
            eos_idx=char_vocab["<eos>"],
            max_seq_len=Config.MAX_SEQ_LEN,
        ).to(Config.DEVICE)
        torch.save(seq2seq_model.state_dict(), Config.SEQ2SEQ_MODEL_PATH)

    # 6. Inference Pipeline
    logger.info("\n--- 6. Inference Pipeline ---")

    # Initialize Predictor (this loads the models we just saved/trained)
    predictor = Predictor()

    # Run Generation on Test Set (Debug mode uses subset)
    predictor.generate_submission(debug=True)

    # Verify Submission
    if os.path.exists(Config.SUBMISSION_FILE):
        df_sub = pd.read_csv(Config.SUBMISSION_FILE)
        logger.info(f"Submission generated with {len(df_sub)} rows.")
        logger.info(f"Head:\n{df_sub.head()}")

        # Basic assertions
        assert len(df_sub) > 0
        assert "id" in df_sub.columns
        assert "after" in df_sub.columns
    else:
        raise FileNotFoundError("Submission file was not created.")

    logger.info("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
