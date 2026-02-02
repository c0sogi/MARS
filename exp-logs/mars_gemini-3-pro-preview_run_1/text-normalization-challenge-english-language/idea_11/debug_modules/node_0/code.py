import os
import sys
import pandas as pd
import torch
import numpy as np
import shutil
import warnings
import logging


# 1. Suppress Progress Bars and Warnings
# --------------------------------------
# Monkey-patch tqdm to be silent before importing library modules that use it
def silent_tqdm(iterable, *args, **kwargs):
    return iterable


import tqdm
import tqdm.auto

tqdm.tqdm = silent_tqdm
tqdm.auto.tqdm = silent_tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

# 2. Import Library Modules
# -------------------------
# We import after patching tqdm to ensure the library uses the silent version
from library.config import Config
from library.utils import set_seed, setup_logger
from library.vocab import VocabManager
from library.features import RegexFeatureExtractor, GlobalPriorManager
from library.data_loader import get_dataloaders, KnowledgeBase
from library.models import PriorAugmentedBiLSTMTagger, TransformerSeq2Seq
from library.engine import train_tagger, train_seq2seq
from library.inference import run_inference


def main():
    # Setup Logger
    logger = setup_logger("demo_script", level=logging.INFO)
    logger.info("Starting Text Normalization Demo Script")

    # Set Seed
    set_seed(42)

    # 3. Create Data Subsets for Speed
    # --------------------------------
    # We use the provided metadata but sample it down significantly to run quickly.
    logger.info("Creating data subsets...")

    subset_dir = os.path.join("./working", "demo_data")
    os.makedirs(subset_dir, exist_ok=True)

    # Paths to original metadata
    orig_train = "./metadata/train.csv"
    orig_val = "./metadata/val.csv"
    orig_test = "./metadata/test.csv"

    # Define subset paths
    subset_train_path = os.path.join(subset_dir, "train_subset.csv")
    subset_val_path = os.path.join(subset_dir, "val_subset.csv")
    subset_test_path = os.path.join(subset_dir, "test_subset.csv")

    # Create subsets (Top 2000 rows for train, 500 for val, 200 for test)
    # This ensures we have enough data to form a vocab but small enough to be fast.
    pd.read_csv(orig_train, nrows=2000).to_csv(subset_train_path, index=False)
    pd.read_csv(orig_val, nrows=500).to_csv(subset_val_path, index=False)
    pd.read_csv(orig_test, nrows=200).to_csv(subset_test_path, index=False)

    logger.info(f"Subsets created in {subset_dir}")

    # 4. Override Configuration
    # -------------------------
    # We modify the Config class attributes directly to use our subsets and speed settings.
    Config.TRAIN_FILE = subset_train_path
    Config.VAL_FILE = subset_val_path
    Config.TEST_FILE = subset_test_path

    # Use a separate working directory for this demo to avoid overwriting real experiments
    Config.WORK_DIR = os.path.join("./working", "demo_execution")
    Config.CACHE_DIR = os.path.join(Config.WORK_DIR, "cache")
    Config.VOCAB_DIR = os.path.join(Config.WORK_DIR, "vocabs")
    Config.TAGGER_MODEL_PATH = os.path.join(Config.WORK_DIR, "tagger_demo.pth")
    Config.SEQ2SEQ_MODEL_PATH = os.path.join(Config.WORK_DIR, "seq2seq_demo.pth")
    Config.KNOWLEDGE_BASE_PATH = os.path.join(Config.WORK_DIR, "knowledge_base.parquet")
    Config.PRIORS_PATH = os.path.join(Config.WORK_DIR, "priors.parquet")
    Config.SUBMISSION_PATH = os.path.join(
        Config.WORK_DIR, "submission", "submission.csv"
    )
    Config.BPE_MODEL_PREFIX = os.path.join(Config.WORK_DIR, "bpe_tokenizer")

    # Reduce Hyperparameters for Speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.MAX_VOCAB_SIZE_BPE = 1000  # Small BPE vocab for speed
    Config.MAX_VOCAB_SIZE_WORD = 2000
    Config.LSTM_HIDDEN_SIZE = 64  # Smaller model
    Config.LSTM_LAYERS = 1
    Config.SEQ2SEQ_EMBED_DIM = 64
    Config.SEQ2SEQ_HIDDEN_DIM = 128
    Config.SEQ2SEQ_LAYERS = 1
    Config.SEQ2SEQ_HEADS = 2

    # Re-run setup to create new directories
    Config.setup()

    # 5. Build Vocabularies & Managers
    # --------------------------------
    logger.info("Building vocabularies and features...")

    # Force rebuild by setting load_cached_data=False
    data_artifacts = get_dataloaders(load_cached_data=False)

    vocab_manager = data_artifacts["vocab_manager"]
    prior_manager = data_artifacts["prior_manager"]
    kb = data_artifacts["kb"]
    dataloaders = {
        "tagger": data_artifacts["tagger"],
        "seq2seq": data_artifacts["seq2seq"],
    }

    # Validation: Check Vocab Sizes
    word_vocab = vocab_manager.get_word_vocab()
    logger.info(f"Word Vocab Size: {len(word_vocab)}")
    assert len(word_vocab) > 0, "Word vocabulary should not be empty"

    # Validation: Check Regex Extractor
    regex = RegexFeatureExtractor()
    sample_text = "123.45"
    feats = regex.extract(sample_text)
    assert isinstance(feats, np.ndarray), "Regex features must be numpy array"
    assert feats.shape[0] == regex.get_feature_dim(), "Regex feature dimension mismatch"
    # Check if 'is_decimal' or 'has_digit' triggered (indices depend on sorted keys)
    # We just verify it runs without error and returns correct shape.

    # 6. Model Initialization & Forward Pass Check
    # --------------------------------------------
    logger.info("Initializing models...")

    num_classes = len(vocab_manager.get_class_vocab())
    regex_dim = regex.get_feature_dim()

    tagger_model = PriorAugmentedBiLSTMTagger(vocab_manager, regex_dim, num_classes).to(
        Config.DEVICE
    )
    seq2seq_model = TransformerSeq2Seq(vocab_manager, num_classes).to(Config.DEVICE)

    # Fetch a batch to verify forward pass
    tagger_batch = next(iter(dataloaders["tagger"]["train"]))
    with torch.no_grad():
        word_ids = tagger_batch["word_ids"].to(Config.DEVICE)
        bpe_ids = tagger_batch["bpe_ids"].to(Config.DEVICE)
        char_ids = tagger_batch["char_ids"].to(Config.DEVICE)
        regex_feats = tagger_batch["regex_feats"].to(Config.DEVICE)
        prior_feats = tagger_batch["prior_feats"].to(Config.DEVICE)

        logits = tagger_model(word_ids, bpe_ids, char_ids, regex_feats, prior_feats)

        # Expected shape: (Batch, Seq, NumClasses)
        assert logits.dim() == 3, f"Expected 3D logits, got {logits.dim()}"
        assert (
            logits.size(2) == num_classes
        ), f"Expected {num_classes} classes, got {logits.size(2)}"
        logger.info("Tagger forward pass successful.")

    # 7. Training Loop (Demo)
    # -----------------------
    logger.info("Running Tagger Training (1 Epoch)...")
    trained_tagger = train_tagger(
        dataloaders, vocab_manager, prior_manager, load_cached_data=False
    )

    logger.info("Running Seq2Seq Training (1 Epoch)...")
    # Note: Seq2Seq dataset might be empty if no changes in subset. Check length.
    if len(dataloaders["seq2seq"]["train"].dataset) > 0:
        trained_seq2seq = train_seq2seq(dataloaders, vocab_manager)
    else:
        logger.warning(
            "Seq2Seq dataset is empty (no changed tokens in subset). Skipping Seq2Seq training."
        )
        # Save dummy model for inference to work
        torch.save(seq2seq_model.state_dict(), Config.SEQ2SEQ_MODEL_PATH)

    # 8. Inference Pipeline
    # ---------------------
    logger.info("Running Inference on Test Subset...")

    # We use the run_inference helper which orchestrates everything
    run_inference(test_file=Config.TEST_FILE, output_path=Config.SUBMISSION_PATH)

    # 9. Final Validation
    # -------------------
    logger.info("Validating output...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    logger.info(f"Submission generated with {len(df_sub)} rows.")

    # Check columns
    assert (
        "id" in df_sub.columns and "after" in df_sub.columns
    ), "Submission missing required columns"

    # Check row count matches test subset
    df_test = pd.read_csv(Config.TEST_FILE)
    # Note: TaggerDataset groups by sentence, but submission is per token.
    # The inference pipeline handles flattening.
    # However, if the test subset has incomplete sentences (cut by nrows),
    # the logic might differ slightly, but nrows=200 on CSV usually cuts in middle of sentence or token list.
    # Let's just check it's not empty.
    assert len(df_sub) > 0, "Submission file is empty"

    # Check if 'id' format is correct (e.g., "12_3")
    sample_id = df_sub.iloc[0]["id"]
    assert "_" in str(sample_id), f"Invalid ID format: {sample_id}"

    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    main()
