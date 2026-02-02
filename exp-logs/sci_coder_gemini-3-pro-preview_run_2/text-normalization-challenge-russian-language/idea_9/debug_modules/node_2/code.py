import os
import torch
import pandas as pd
import numpy as np
import shutil
from library.config import Config
from library.utils import set_seed, is_semiotic, clean_text
from library.tokenizer import CharTokenizer, TargetBPETokenizer, train_tokenizers
from library.hfbb import HFBBModel
from library.model import Seq2SeqTransformer
from library.dataset import ContextWindowDataset
from library.train import train_model
from library.inference import CascadePredictor


def main():
    print("=== Starting Demonstration of Text Normalization Library ===")

    # ------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo
    # ------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid execution...")

    # Set paths to use a clean working directory for this run
    demo_working_dir = "./working/demo_execution"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir)

    # Override Config class attributes directly
    Config.WORKING_DIR = demo_working_dir
    Config.CACHE_DIR = os.path.join(demo_working_dir, "data_cache")
    Config.HFBB_CACHE_DIR = os.path.join(demo_working_dir, "hfbb_cache")
    Config.CHECKPOINT_DIR = os.path.join(demo_working_dir, "checkpoints")
    Config.TOKENIZER_DIR = os.path.join(demo_working_dir, "tokenizers")
    Config.SUBMISSION_DIR = os.path.join(demo_working_dir, "submission")

    Config.BEST_MODEL_PATH = os.path.join(
        Config.CHECKPOINT_DIR, "transformer_demo_best.pth"
    )
    Config.TARGET_TOKENIZER_MODEL = os.path.join(Config.TOKENIZER_DIR, "bpe_demo.model")
    Config.TARGET_TOKENIZER_VOCAB = os.path.join(Config.TOKENIZER_DIR, "bpe_demo.vocab")
    Config.CHAR_VOCAB_PATH = os.path.join(Config.TOKENIZER_DIR, "char_vocab.json")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Override Hyperparameters for a tiny model and quick training
    Config.DEBUG = True
    Config.DEBUG_SIZE = 2000  # Small subset of data
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.D_MODEL = 64
    Config.NHEAD = 2
    Config.NUM_ENCODER_LAYERS = 2
    Config.NUM_DECODER_LAYERS = 2
    Config.DIM_FEEDFORWARD = 128
    Config.TARGET_VOCAB_SIZE = (
        4000  # Increased to accommodate Russian char set (Cite debug_lesson_4)
    )
    Config.PATIENCE = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo script

    # Ensure directories exist
    Config.setup()
    set_seed(Config.SEED)
    print("Configuration updated for demo mode.")

    # ------------------------------------------------------------------------
    # 2. Tokenizer Demonstration
    # ------------------------------------------------------------------------
    print("\n[2] Demonstrating Tokenizers...")

    # Create dummy data for tokenizer training
    dummy_texts = ["hello world", "testing 123", "normalization is fun"]

    # A. CharTokenizer
    char_tok = CharTokenizer()
    char_tok.fit_on_texts(dummy_texts)
    encoded = char_tok.encode("hello", add_special_tokens=False)
    decoded = char_tok.decode(encoded)

    print(f"  CharTokenizer: 'hello' -> {encoded} -> '{decoded}'")
    assert decoded == "hello", "CharTokenizer round-trip failed"
    assert char_tok.get_vocab_size() > 0, "CharTokenizer vocab is empty"

    # B. TargetBPETokenizer
    # We need to train it using a file or list. The class method train takes a list.
    target_tok = TargetBPETokenizer()
    model_prefix = os.path.join(Config.TOKENIZER_DIR, "test_bpe")
    target_tok.train(
        dummy_texts, model_prefix, vocab_size=50
    )  # Tiny vocab for tiny data

    t_encoded = target_tok.encode("hello", add_special_tokens=False)
    t_decoded = target_tok.decode(t_encoded)

    print(f"  TargetBPETokenizer: 'hello' -> {t_encoded} -> '{t_decoded}'")
    assert t_decoded == "hello", "TargetBPETokenizer round-trip failed"

    # ------------------------------------------------------------------------
    # 3. HFBB (Hierarchical Frequency Back-off) Demonstration
    # ------------------------------------------------------------------------
    print("\n[3] Demonstrating HFBB Model...")

    # Create a synthetic dataframe
    hfbb_data = pd.DataFrame(
        {
            "sentence_id": [0, 0, 0, 1, 1],
            "token_id": [0, 1, 2, 0, 1],
            "before": ["The", "cat", "sat", "The", "dog"],
            "after": ["the", "cat", "sat", "the", "dog"],
        }
    )

    hfbb = HFBBModel()
    hfbb.fit(hfbb_data, load_cached_data=False)

    # Test Unigram: "cat" -> "cat"
    pred, level, conf = hfbb.predict("cat")
    print(f"  HFBB Prediction for 'cat': {pred} (Level: {level}, Conf: {conf})")
    assert pred == "cat", "HFBB failed to learn unigram 'cat'"
    assert level == "unigram", "HFBB level should be unigram"

    # Test Bigram Context: "The" -> "the" (appears twice)
    # Context: prev="<start>", curr="The"
    pred_bi, level_bi, _ = hfbb.predict("The", prev_token="<start>")
    print(f"  HFBB Prediction for 'The' (start of sent): {pred_bi} (Level: {level_bi})")
    assert pred_bi == "the", "HFBB failed to learn bigram context"

    # ------------------------------------------------------------------------
    # 4. Model Architecture Demonstration
    # ------------------------------------------------------------------------
    print("\n[4] Demonstrating Seq2SeqTransformer Architecture...")

    src_vocab = 100
    tgt_vocab = 100
    model = Seq2SeqTransformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        src_pad_idx=0,
        tgt_pad_idx=0,
        d_model=Config.D_MODEL,
        nhead=Config.NHEAD,
        num_encoder_layers=Config.NUM_ENCODER_LAYERS,
        num_decoder_layers=Config.NUM_DECODER_LAYERS,
        dim_feedforward=Config.DIM_FEEDFORWARD,
    )

    # Dummy inputs: (batch_size=2, seq_len=10)
    src = torch.randint(0, src_vocab, (2, 10))
    tgt = torch.randint(0, tgt_vocab, (2, 8))

    # Forward pass
    logits = model(src, tgt)
    print(f"  Model Output Shape: {logits.shape}")

    # Expected: (batch, tgt_seq_len, tgt_vocab)
    assert logits.shape == (2, 8, tgt_vocab), "Model output shape mismatch"

    # ------------------------------------------------------------------------
    # 5. Full Pipeline: Training
    # ------------------------------------------------------------------------
    print("\n[5] Running Full Training Pipeline (Debug Mode)...")

    # This will:
    # 1. Load data (truncated by DEBUG flag)
    # 2. Train tokenizers
    # 3. Generate curriculum indices
    # 4. Train HFBB
    # 5. Train Transformer for 1 epoch
    trained_model = train_model(load_cached_data=False)

    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not saved."
    print("  Training completed successfully.")

    # ------------------------------------------------------------------------
    # 6. Full Pipeline: Inference
    # ------------------------------------------------------------------------
    print("\n[6] Running Inference Pipeline...")

    # Create a dummy test file to ensure we have something compatible with the trained tokenizers
    # We use the validation data from metadata as a proxy for test data
    df_val = pd.read_csv(Config.VAL_DATA_PATH).iloc[:100].copy()
    # Test file format only has sentence_id, token_id, before
    df_test_proxy = df_val[["sentence_id", "token_id", "before"]].copy()

    # Save to a temporary location and update config path
    mini_test_path = os.path.join(demo_working_dir, "mini_test.csv")
    df_test_proxy.to_csv(mini_test_path, index=False)
    Config.TEST_DATA_PATH = mini_test_path

    predictor = CascadePredictor()
    predictor.generate_submission()

    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not generated."

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"  Submission generated with {len(sub_df)} rows.")
    assert len(sub_df) == len(df_test_proxy), "Submission row count mismatch."
    assert (
        "id" in sub_df.columns and "after" in sub_df.columns
    ), "Submission columns missing."

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
