import os
import shutil
import random
import numpy as np
import pandas as pd
import torch
import warnings

# Import library components
from library.config import Config
from library.text_utils import CharTokenizer, is_hard_token, get_context_window
from library.symbolic_layer import SymbolicMemory
from library.retrieval_system import SimilarityIndex
from library.dataset_factory import create_dataloaders, RAGDataset
from library.neural_architecture import RAGTransformer
from library.trainer import ModelTrainer
from library.inference_pipeline import CascadeSolver

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_demo():
    print("=== Starting Text Normalization Library Demo ===\n")

    # ==========================================
    # 0. Configuration & Setup
    # ==========================================
    print("--- Step 0: Configuring for Speed ---")

    # Override Config for fast execution
    Config.WORKING_DIR = "./working/demo_execution"

    # Cite debug_lesson_3: Purge Stale Artifacts
    if os.path.exists(Config.WORKING_DIR):
        print(f"Cleaning up stale artifacts in {Config.WORKING_DIR}...")
        shutil.rmtree(Config.WORKING_DIR)

    Config.STATS_DIR = os.path.join(Config.WORKING_DIR, "stats")
    Config.PROCESSED_DIR = os.path.join(Config.WORKING_DIR, "processed")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Update derived paths in Config
    Config.TRIGRAM_STATS_PATH = os.path.join(Config.STATS_DIR, "trigram_stats.parquet")
    Config.BIGRAM_LEFT_STATS_PATH = os.path.join(
        Config.STATS_DIR, "bigram_left_stats.parquet"
    )
    Config.BIGRAM_RIGHT_STATS_PATH = os.path.join(
        Config.STATS_DIR, "bigram_right_stats.parquet"
    )
    Config.UNIGRAM_STATS_PATH = os.path.join(Config.STATS_DIR, "unigram_stats.parquet")

    Config.TFIDF_MODEL_PATH = os.path.join(
        Config.WORKING_DIR, "tfidf_vectorizer.joblib"
    )
    Config.KNN_INDEX_PATH = os.path.join(Config.WORKING_DIR, "knn_index.bin")
    Config.HARD_SAMPLES_PATH = os.path.join(
        Config.PROCESSED_DIR, "hard_samples.parquet"
    )

    Config.TOKENIZER_PATH = os.path.join(Config.WORKING_DIR, "tokenizer.json")
    Config.MODEL_CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "model_demo.pt")

    # Limit data and model size
    Config.MAX_TRAIN_SAMPLES = 5000  # Use only 5k samples for stats/training
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.EMBED_DIM = 32
    Config.HIDDEN_DIM = 64
    Config.N_HEADS = 2
    Config.N_ENCODER_LAYERS = 1
    Config.N_DECODER_LAYERS = 1
    Config.MAX_SEQ_LEN = 64

    # Re-run setup to create new directories
    Config.setup()
    set_seed(Config.SEED)
    print("Configuration updated for demo run.")

    # ==========================================
    # 1. Text Utilities & Tokenizer
    # ==========================================
    print("\n--- Step 1: Validating Text Utilities ---")

    # Test is_hard_token
    assert is_hard_token("DATE", "2023") is True
    assert is_hard_token("PLAIN", "hello") is False
    assert is_hard_token("VERBATIM", "hello") is False  # Alpha is handled by heuristic
    assert is_hard_token("MONEY", "$5") is True
    print("is_hard_token logic verified.")

    # Test Tokenizer
    tokenizer = CharTokenizer()
    sample_texts = ["hello", "world", "$123", "cat"]
    tokenizer.train(sample_texts)

    encoded = tokenizer.encode("cat", add_special_tokens=True)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)

    print(f"Vocab Size: {tokenizer.vocab_size}")
    print(f"Encoded 'cat': {encoded}")
    print(f"Decoded: '{decoded}'")

    assert decoded == "cat"
    assert tokenizer.sos_token_id in encoded
    assert tokenizer.eos_token_id in encoded

    # Save/Load check - USE TEMP PATH TO AVOID POLLUTING PIPELINE
    temp_tokenizer_path = Config.TOKENIZER_PATH + ".test"
    tokenizer.save(temp_tokenizer_path)
    tokenizer_loaded = CharTokenizer()
    tokenizer_loaded.load(temp_tokenizer_path)
    assert tokenizer_loaded.char2id == tokenizer.char2id
    print("Tokenizer save/load verified.")

    # ==========================================
    # 2. Symbolic Memory Layer
    # ==========================================
    print("\n--- Step 2: Validating Symbolic Memory ---")

    symbolic_mem = SymbolicMemory()
    # Force build from scratch using the small MAX_TRAIN_SAMPLES
    symbolic_mem.build_stats(load_cached_data=False)

    # Verify internal state
    print(f"Stats built. Unigrams: {len(symbolic_mem.unigram_stats)}")

    # Test Query (Mocking a known entry if possible, or just checking return type)
    # Since we used a random subset of 5000, we can't guarantee specific tokens exist,
    # but we can check the API handles queries without crashing.
    res = symbolic_mem.query("unknown_token_xyz", None, None)
    assert res is None
    print("Symbolic memory query API verified.")

    # ==========================================
    # 3. Retrieval System
    # ==========================================
    print("\n--- Step 3: Validating Retrieval System ---")

    sim_index = SimilarityIndex()
    # Build index from scratch
    sim_index.build_index(load_cached_data=False)

    # Test Retrieval
    query = "$100"
    results = sim_index.retrieve(query, k=2)
    print(f"Query: '{query}'")
    for r in results:
        print(f"  -> Found: '{r['source']}' (Dist: {r['distance']:.4f})")

    assert isinstance(results, list)
    if len(results) > 0:
        assert "source" in results[0]
        assert "target" in results[0]
    print("Retrieval system verified.")

    # ==========================================
    # 4. Data Loading Pipeline
    # ==========================================
    print("\n--- Step 4: Validating Data Loaders ---")

    # Create dataloaders (this will use the tokenizer and index we just built/saved)
    train_loader, val_loader, test_loader, tokenizer = create_dataloaders(
        load_cached_data=True
    )

    # Inspect one batch
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    target_ids = batch["target_ids"]

    print(f"Batch Input Shape: {input_ids.shape}")
    print(f"Batch Target Shape: {target_ids.shape}")

    assert input_ids.shape[0] == Config.BATCH_SIZE
    assert target_ids.shape[0] == Config.BATCH_SIZE
    # Check for separator token in input (RAG structure)
    assert (input_ids == tokenizer.sep_token_id).any()
    print("Data loading pipeline verified.")

    # ==========================================
    # 5. Neural Model & Training
    # ==========================================
    print("\n--- Step 5: Validating Neural Training ---")

    trainer = ModelTrainer(tokenizer)

    # Run training for the configured 1 epoch
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # Check if model file was created
    assert os.path.exists(Config.MODEL_CHECKPOINT_PATH)
    print("Model training and checkpointing verified.")

    # ==========================================
    # 6. Inference Pipeline (Cascade Solver)
    # ==========================================
    print("\n--- Step 6: Validating Cascade Inference ---")

    # Create a dummy test set
    # We include:
    # 1. A simple alpha word (handled by Gate)
    # 2. A complex token (handled by Neural/Retrieval)
    # 3. A token likely in symbolic memory (if we were using full stats, here just for flow)
    df_test_mock = pd.DataFrame(
        {
            "sentence_id": [0, 0, 1],
            "token_id": [0, 1, 0],
            "before": ["Hello", "$5.00", "world"],
            "id": ["0_0", "0_1", "1_0"],
        }
    )

    solver = CascadeSolver()

    # Run inference
    submission = solver.solve(df_test_mock)

    print("\nInference Results:")
    print(submission)

    # Validations
    assert len(submission) == 3
    assert "id" in submission.columns
    assert "after" in submission.columns

    # Check Heuristic Gate (Hello -> Hello)
    hello_row = submission[submission["id"] == "0_0"].iloc[0]
    assert hello_row["after"] == "Hello", "Heuristic gate failed for 'Hello'"

    # Check Neural/Retrieval flow (should produce *something* string-like for $5.00)
    money_row = submission[submission["id"] == "0_1"].iloc[0]
    assert isinstance(money_row["after"], str)
    assert (
        len(money_row["after"]) > 0
    ), "Neural model returned empty string (likely tokenizer issue)"

    print("Cascade inference pipeline verified.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
