import os
import shutil
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, CharTokenizer, encode_context_window
from library.data_manager import (
    load_parquet_data,
    _add_context_columns,
    prepare_neural_dataframe,
    NormalizationDataset,
    collate_fn,
    get_tokenizer,
)
from library.symbolic_solver import SymbolicModel
from library.neural_solver import TransformerSeq2Seq
from library.trainer import train_neural_model
from library.inference_pipeline import HybridNormalizer, generate_submission


def run_demo():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print(">>> SECTION 1: CONFIGURATION & SETUP")

    # Override Config for speed and isolation
    Config.WORKING_DIR = "./working/demo_execution"
    Config.STATS_CACHE_DIR = os.path.join(Config.WORKING_DIR, "stats")
    Config.PROCESSED_DATA_DIR = os.path.join(Config.WORKING_DIR, "processed")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "model_demo.pt")

    # Enable Debug Mode to use small data subsets
    Config.DEBUG = True
    Config.DEBUG_SIZE = 2000  # Small subset for fast execution

    # Reduce Model Complexity for Demo
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.D_MODEL = 64
    Config.NHEAD = 2
    Config.NUM_ENCODER_LAYERS = 2
    Config.NUM_DECODER_LAYERS = 2
    Config.DIM_FEEDFORWARD = 128

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    # Initialize environment
    Config.setup()
    seed_everything(Config.SEED)
    print("Configuration updated for rapid demonstration.")

    # ==========================================
    # 2. Tokenizer Verification
    # ==========================================
    print("\n>>> SECTION 2: TOKENIZER UTILITIES")

    # Create a tokenizer and fit on dummy text
    tokenizer = CharTokenizer()
    dummy_texts = ["hello", "world", "123", "$"]
    tokenizer.fit_on_texts(dummy_texts)

    print(f"Vocab size: {len(tokenizer)}")

    # Test Encoding
    sample_text = "h$1"
    encoded_ids = tokenizer.encode(sample_text, add_special_tokens=True)
    decoded_text = tokenizer.decode(encoded_ids, remove_special_tokens=True)

    print(f"Original: {sample_text}")
    print(f"Encoded IDs: {encoded_ids}")
    print(f"Decoded: {decoded_text}")

    # Verification
    assert (
        decoded_text == sample_text
    ), "Tokenizer decode failed to reconstruct original text."
    assert tokenizer.sos_token_id in encoded_ids, "SOS token missing."
    assert tokenizer.eos_token_id in encoded_ids, "EOS token missing."

    # Test Context Window Encoding
    ctx_ids = encode_context_window(tokenizer, "prev", "curr", "next")
    # Structure: [prev] <SEP> [curr] <SEP> [next]
    # Check for 2 separators
    sep_count = ctx_ids.count(tokenizer.sep_token_id)
    assert sep_count == 2, f"Context window should have 2 separators, found {sep_count}"
    print("Tokenizer logic verified.")

    # ==========================================
    # 3. Data Manager & Processing
    # ==========================================
    print("\n>>> SECTION 3: DATA PROCESSING")

    # Load raw training data (subsampled due to DEBUG=True)
    df_train = load_parquet_data("train")
    print(f"Loaded {len(df_train)} training samples (DEBUG mode).")

    # Verify Context Columns Logic
    # Create a synthetic dataframe to test strict boundary logic
    df_synth = pd.DataFrame(
        {
            "sentence_id": [1, 1, 2, 2],
            "token_id": [0, 1, 0, 1],
            "before": ["s1_t1", "s1_t2", "s2_t1", "s2_t2"],
            "class": ["PLAIN", "PLAIN", "PLAIN", "PLAIN"],
            "after": ["norm1", "norm2", "norm3", "norm4"],
        }
    )

    df_synth = _add_context_columns(df_synth)

    # Check Sentence 1
    # Token 0: prev should be empty
    assert (
        df_synth.loc[0, "prev_before"] == ""
    ), "Sentence start should have empty prev."
    # Token 1: prev should be s1_t1
    assert df_synth.loc[1, "prev_before"] == "s1_t1", "Context shift incorrect."

    # Check Sentence Boundary (Sentence 2 Token 0)
    # Prev should be empty, NOT s1_t2
    assert df_synth.loc[2, "prev_before"] == "", "Context leaked across sentences."
    print("Context window generation logic verified.")

    # ==========================================
    # 4. Symbolic Model Logic
    # ==========================================
    print("\n>>> SECTION 4: SYMBOLIC MODEL")

    # Create a specific dataset to test symbolic lookup
    df_sym = pd.DataFrame(
        {
            "sentence_id": [10, 10, 10],
            "token_id": [0, 1, 2],
            "before": ["A", "B", "C"],
            "after": ["a_norm", "b_norm", "c_norm"],
            "class": ["PLAIN", "PLAIN", "PLAIN"],
        }
    )
    # Add context
    df_sym = _add_context_columns(df_sym)

    # Initialize SymbolicModel with this specific data
    # It will build stats from df_sym
    symbolic_model = SymbolicModel(df_train=df_sym, load_cached_data=False)

    # Test Trigram Resolution: A -> B -> C (Target B)
    # Context for B is prev="A", next="C"
    res = symbolic_model.resolve("A", "B", "C")
    print(f"Symbolic Lookup ('A', 'B', 'C') -> {res}")
    assert (
        res == "b_norm"
    ), f"Symbolic trigram lookup failed. Expected 'b_norm', got {res}"

    # Test Unigram Resolution
    res_uni = symbolic_model.resolve(None, "A", None)  # Context missing
    print(f"Symbolic Lookup (None, 'A', None) -> {res_uni}")
    assert res_uni == "a_norm", "Symbolic unigram lookup failed."

    print("Symbolic model logic verified.")

    # ==========================================
    # 5. Neural Model & Training
    # ==========================================
    print("\n>>> SECTION 5: NEURAL MODEL TRAINING")

    # Train the model using the library function
    # This handles data loading, tokenization, and the training loop
    # With DEBUG=True and EPOCHS=1, this should be fast.
    model, trained_tokenizer = train_neural_model(load_cached_data=False)

    assert isinstance(
        model, TransformerSeq2Seq
    ), "Trainer did not return a Transformer model."
    assert len(trained_tokenizer) > 0, "Tokenizer is empty."

    # Validate Forward Pass Dimensions manually
    model.eval()
    dummy_src = torch.randint(0, len(trained_tokenizer), (4, 32)).to(
        Config.DEVICE
    )  # (batch, seq)
    dummy_tgt = torch.randint(0, len(trained_tokenizer), (4, 32)).to(Config.DEVICE)

    with torch.no_grad():
        output = model(dummy_src, dummy_tgt)

    # Output shape: (batch, tgt_seq_len, vocab_size)
    expected_shape = (4, 32, len(trained_tokenizer))
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"
    print("Neural model training and forward pass verified.")

    # ==========================================
    # 6. Inference Pipeline
    # ==========================================
    print("\n>>> SECTION 6: INFERENCE PIPELINE")

    # We will run the full submission generation pipeline.
    # It loads the test set (subsampled), initializes the HybridNormalizer,
    # and produces a CSV.

    generate_submission()

    # Verify Submission File
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not generated.")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(df_sub)} rows.")

    # Check format
    assert (
        "id" in df_sub.columns and "after" in df_sub.columns
    ), "Submission columns missing."

    # Sanitize text column to prevent null propagation from empty strings. Cite debug_lesson_1
    df_sub["after"] = df_sub["after"].fillna("").astype(str)

    assert len(df_sub) > 0, "Submission file is empty."

    # Check for NaNs
    nan_count = df_sub["after"].isna().sum()
    assert nan_count == 0, f"Submission contains {nan_count} NaN values."

    print("Inference pipeline verified successfully.")
    print("\n>>> DEMO COMPLETE")


if __name__ == "__main__":
    run_demo()
