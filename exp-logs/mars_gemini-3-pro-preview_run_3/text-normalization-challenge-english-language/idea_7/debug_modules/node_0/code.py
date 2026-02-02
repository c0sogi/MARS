import os
import sys
import shutil
import random
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.data_utils import (
    Tokenizer,
    TextNormalizerDataset,
    process_context,
    get_dataloader,
    load_data,
)
from library.symbolic_model import SymbolicMemory
from library.neural_model import Seq2SeqModel
from library.trainer import Trainer
from library.inference import CascadePredictor, generate_submission


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_demo_environment():
    print("=== Setting up Demo Environment ===")

    # 1. Define paths
    base_dir = "./working/demo_execution"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    # 2. Monkey-patch Config to use the demo directory and reduced settings
    print("Overriding Config parameters for speed...")
    Config.WORK_DIR = base_dir
    Config.STATS_CACHE_DIR = os.path.join(base_dir, "stats")
    Config.PROCESSED_DATA_DIR = os.path.join(base_dir, "processed")
    Config.MODEL_CHECKPOINT_PATH = os.path.join(base_dir, "model_demo.pt")
    Config.SUBMISSION_PATH = os.path.join(base_dir, "submission/submission_demo.csv")

    # Create directories
    os.makedirs(Config.STATS_CACHE_DIR, exist_ok=True)
    os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Reduce Model/Training Hyperparameters for Demo Speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.ENC_HIDDEN_DIM = 32
    Config.DEC_HIDDEN_DIM = 32
    Config.ATTN_DIM = 16
    Config.CHAR_EMB_DIM = 16
    Config.NUM_LAYERS = 1
    Config.SOFT_FILTER_RATIO = 1.0  # Use all data in our mini set

    # 3. Create Mini Datasets
    print("Creating mini datasets from metadata...")

    # Load source metadata
    orig_train_path = "./metadata/train.parquet"
    orig_test_path = "./metadata/test.parquet"

    if not os.path.exists(orig_train_path):
        raise FileNotFoundError(f"Original metadata not found at {orig_train_path}")

    # Read a small sample
    df_train_full = pd.read_parquet(orig_train_path)
    df_test_full = pd.read_parquet(orig_test_path)

    # Select top 500 samples for train, 100 for val, 100 for test
    # Ensuring we keep sentences intact is good practice, but for this speed demo
    # simple head slicing is sufficient as long as we process context correctly.
    df_mini_train = df_train_full.head(500).copy()
    df_mini_val = df_train_full.iloc[500:600].copy()
    df_mini_test = df_test_full.head(100).copy()

    # Save mini datasets
    mini_train_path = os.path.join(base_dir, "mini_train.parquet")
    mini_val_path = os.path.join(base_dir, "mini_val.parquet")
    mini_test_path = os.path.join(base_dir, "mini_test.parquet")

    df_mini_train.to_parquet(mini_train_path, index=False)
    df_mini_val.to_parquet(mini_val_path, index=False)
    df_mini_test.to_parquet(mini_test_path, index=False)

    # Point Config to these new files
    Config.TRAIN_META_PATH = mini_train_path
    Config.VAL_META_PATH = mini_val_path
    Config.TEST_META_PATH = mini_test_path

    print(f"Mini datasets created at {base_dir}")


def demo_tokenizer_logic():
    print("\n=== Demo: Tokenizer & Data Utils ===")
    tokenizer = Tokenizer()

    text = "Hello $3.50"

    # 1. Test Tokenize/Detokenize
    ids = tokenizer.tokenize(text)
    reconstructed = tokenizer.detokenize(ids)
    print(f"Original: '{text}'")
    print(f"Token IDs: {ids}")
    print(f"Reconstructed: '{reconstructed}'")

    assert reconstructed == text, "Detokenization failed to reconstruct original text"

    # 2. Test Factored Features
    char_ids, case_ids, type_ids = tokenizer.get_factored_features(text)
    print(f"Factored Features for '{text}':")
    print(f"  Char: {char_ids}")
    print(f"  Case: {case_ids}")
    print(f"  Type: {type_ids}")

    assert (
        len(char_ids) == len(case_ids) == len(type_ids) == len(text)
    ), "Factored feature lengths mismatch"

    # 3. Test Context Processing
    df_sample = pd.DataFrame(
        {
            "sentence_id": [0, 0, 0, 1],
            "token_id": [0, 1, 2, 0],
            "before": ["A", "B", "C", "D"],
        }
    )
    df_proc = process_context(df_sample)

    # Check logic:
    # Token 0_1 ("B") should have prev="A", next="C"
    row_b = df_proc[(df_proc.sentence_id == 0) & (df_proc.token_id == 1)].iloc[0]
    assert (
        row_b["prev"] == "A"
    ), f"Context prev incorrect: expected 'A', got {row_b['prev']}"
    assert (
        row_b["next"] == "C"
    ), f"Context next incorrect: expected 'C', got {row_b['next']}"

    # Token 0_2 ("C") should have next="" (end of sentence)
    row_c = df_proc[(df_proc.sentence_id == 0) & (df_proc.token_id == 2)].iloc[0]
    assert (
        row_c["next"] == ""
    ), f"Context boundary incorrect: expected '', got {row_c['next']}"

    print("Tokenizer and Context Processing verified.")


def demo_symbolic_model():
    print("\n=== Demo: Symbolic Memory ===")

    # Initialize
    sym_model = SymbolicMemory()

    # Fit on the mini training data
    # We force load_cached_data=False to ensure it computes from our mini dataset
    sym_model.fit(load_cached_data=False)

    # Verify it learned something
    # We pick a token from the mini_train dataset to test
    df_train = pd.read_parquet(Config.TRAIN_META_PATH)
    sample_row = df_train.iloc[0]

    token = str(sample_row["before"])
    # Context might be empty if it's the start of sentence, so we process to be sure
    df_train_proc = process_context(df_train)
    sample_row = df_train_proc.iloc[0]

    pred = sym_model.predict(
        sample_row["before"], sample_row["prev"], sample_row["next"]
    )

    print(f"Symbolic Prediction for '{sample_row['before']}': {pred}")
    print(f"Actual Target: {sample_row['after']}")

    # It should match because we just trained on it (unless there's ambiguity, but for top 1 it's likely correct)
    if pred is not None:
        assert pred == str(
            sample_row["after"]
        ), "Symbolic memory failed to recall training sample"
    else:
        print(
            "Warning: Symbolic memory returned None (possibly due to ambiguity logic or empty stats)."
        )


def demo_neural_model_architecture():
    print("\n=== Demo: Neural Model Architecture ===")

    model = Seq2SeqModel().to(Config.DEVICE)

    # Create dummy batch
    batch_size = 2
    seq_len = 10

    # Random inputs within vocab range
    src_char = torch.randint(0, Config.VOCAB_SIZE, (batch_size, seq_len)).to(
        Config.DEVICE
    )
    src_case = torch.randint(0, Config.CASE_VOCAB_SIZE, (batch_size, seq_len)).to(
        Config.DEVICE
    )
    src_type = torch.randint(0, Config.TYPE_VOCAB_SIZE, (batch_size, seq_len)).to(
        Config.DEVICE
    )

    # Forward pass (Inference mode, no target)
    outputs, aux_logits = model(src_char, src_case, src_type, tgt=None)

    print(f"Input shape: {src_char.shape}")
    print(f"Output shape: {outputs.shape}")  # Should be [batch, max_len, vocab_size]
    print(f"Aux logits shape: {aux_logits.shape}")  # Should be [batch, num_classes]

    assert outputs.shape[0] == batch_size
    assert outputs.shape[2] == Config.VOCAB_SIZE
    assert aux_logits.shape[1] == Config.NUM_CLASSES

    print("Neural model forward pass successful.")


def demo_training_pipeline():
    print("\n=== Demo: Training Pipeline ===")

    trainer = Trainer()

    # Run training
    # This uses the mini datasets defined in setup
    # load_cached_data=False forces regeneration of processed files
    trainer.fit(load_cached_data=False)

    # Check if model was saved
    assert os.path.exists(
        Config.MODEL_CHECKPOINT_PATH
    ), "Model checkpoint was not created."
    print(f"Model successfully saved to {Config.MODEL_CHECKPOINT_PATH}")


def demo_inference_pipeline():
    print("\n=== Demo: Inference Pipeline ===")

    # 1. Direct Cascade Prediction
    predictor = CascadePredictor(model_path=Config.MODEL_CHECKPOINT_PATH)

    # Load mini test set
    df_test = pd.read_parquet(Config.TEST_META_PATH)

    # Run prediction on first 10 rows
    df_sample = df_test.head(10).copy()
    preds = predictor.predict(df_sample)

    print(f"Predictions for first 5 samples:")
    for i in range(5):
        print(f"  In: '{df_sample.iloc[i]['before']}' -> Out: '{preds[i]}'")

    assert len(preds) == len(df_sample), "Prediction count mismatch"
    assert all(isinstance(p, str) for p in preds), "Some predictions are not strings"

    # 2. Full Submission Generation
    # This function reads from Config.TEST_META_PATH and writes to Config.SUBMISSION_PATH
    generate_submission(load_cached_data=True, limit=20)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    # Verify submission format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")
    print(f"Submission columns: {list(df_sub.columns)}")

    assert (
        "id" in df_sub.columns and "after" in df_sub.columns
    ), "Submission columns incorrect"
    assert len(df_sub) > 0, "Submission file is empty"

    print("Inference pipeline verified.")


if __name__ == "__main__":
    set_seed(42)

    try:
        setup_demo_environment()

        demo_tokenizer_logic()

        demo_symbolic_model()

        demo_neural_model_architecture()

        demo_training_pipeline()

        demo_inference_pipeline()

        print("\n=== All Demonstrations Completed Successfully ===")

    except Exception as e:
        print(f"\n!!! Demo Failed with Error: {e} !!!")
        import traceback

        traceback.print_exc()
        sys.exit(1)
