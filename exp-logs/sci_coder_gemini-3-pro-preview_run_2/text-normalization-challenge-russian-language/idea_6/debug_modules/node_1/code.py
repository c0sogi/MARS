import os
import sys
import shutil
import pandas as pd
import torch
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, is_semiotic
from library.hfbb_engine import HFBB
from library.neural_net import (
    CharTokenizer,
    TargetBPETokenizer,
    ResidualGenerator,
    CharToSubwordTransformer,
)
from library.training_engine import Trainer
from library.cascade_manager import CascadeManager
from library.data_utils import get_enriched_residuals


def main():
    print("=== Starting Text Normalization Pipeline Demo ===")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[1] Configuring environment for fast demonstration...")

    # Define a demo working directory
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config parameters for speed and isolation
    Config.WORKING_DIR = DEMO_DIR
    Config.HFBB_CACHE_DIR = os.path.join(DEMO_DIR, "hfbb_cache")
    Config.RESIDUAL_TRAIN_CACHE = os.path.join(DEMO_DIR, "processed_train.parquet")
    Config.RESIDUAL_VAL_CACHE = os.path.join(DEMO_DIR, "processed_val.parquet")
    Config.TOKENIZER_PREFIX = os.path.join(DEMO_DIR, "bpe_demo")
    Config.TRANSFORMER_CHECKPOINT = os.path.join(DEMO_DIR, "seq2seq_demo.pth")
    Config.SUBMISSION_FILE = os.path.join(DEMO_DIR, "submission_demo.csv")

    # Hyperparameters for speed
    Config.DEBUG = True
    Config.DEBUG_SIZE = 200  # Only use 200 samples
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.K_FOLDS = 2
    Config.TARGET_VOCAB_SIZE = 100  # Small vocab for small dataset
    Config.EARLY_STOPPING_PATIENCE = 1

    # Re-run setup to ensure directories exist
    Config.setup()
    set_seed(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")

    # ==========================================
    # 2. Data Preparation (Mini Datasets)
    # ==========================================
    print("\n[2] Creating mini datasets from metadata...")

    # Load original metadata
    orig_train_path = "./metadata/train.csv"
    orig_val_path = "./metadata/val.csv"
    orig_test_path = "./metadata/test.csv"

    # Create subsets
    df_train_mini = pd.read_csv(orig_train_path).head(Config.DEBUG_SIZE)
    df_val_mini = pd.read_csv(orig_val_path).head(Config.DEBUG_SIZE)
    df_test_mini = pd.read_csv(orig_test_path).head(Config.DEBUG_SIZE)

    # Save mini datasets
    mini_train_path = os.path.join(DEMO_DIR, "mini_train.csv")
    mini_val_path = os.path.join(DEMO_DIR, "mini_val.csv")
    mini_test_path = os.path.join(DEMO_DIR, "mini_test.csv")

    df_train_mini.to_csv(mini_train_path, index=False)
    df_val_mini.to_csv(mini_val_path, index=False)
    df_test_mini.to_csv(mini_test_path, index=False)

    # Point Config to mini datasets
    Config.TRAIN_FILE = mini_train_path
    Config.VAL_FILE = mini_val_path
    Config.TEST_FILE = mini_test_path

    print(f"Mini Train Size: {len(df_train_mini)}")
    print(f"Mini Val Size: {len(df_val_mini)}")

    # ==========================================
    # 3. Tier 1: HFBB (Memory Model)
    # ==========================================
    print("\n[3] Demonstrating HFBB (Tier 1)...")

    hfbb = HFBB()
    # Fit on mini train data (ignoring cache to force computation)
    hfbb.fit(df_train_mini, load_cached_data=False)

    # Verify HFBB populated
    print(f"Unigram Map Size: {len(hfbb.unigram_map)}")
    assert len(hfbb.unigram_map) > 0, "HFBB Unigram map should not be empty."

    # Test a simple prediction (Self-test on training data)
    sample_token = df_train_mini.iloc[0]["before"]
    sample_target = df_train_mini.iloc[0]["after"]
    pred = hfbb.predict_token(str(sample_token))

    print(
        f"Sample Prediction: '{sample_token}' -> '{pred}' (Expected: '{sample_target}')"
    )
    # Note: It might not match exactly if context matters, but it should return something.

    # ==========================================
    # 4. Residual Generation
    # ==========================================
    print("\n[4] Generating Residuals (Hard Examples)...")

    # This uses K-Fold Jackknifing to find what HFBB cannot predict
    res_train = ResidualGenerator.get_train_residuals(load_cached_data=False)

    print(f"Generated {len(res_train)} training residuals.")

    # If dataset is too simple, residuals might be empty.
    # For the sake of the demo, if empty, we force some data into the residual cache
    # so the neural net training doesn't crash.
    if len(res_train) < 10:
        print(
            "Warning: Too few residuals generated naturally. Injecting dummy residuals for demo."
        )
        # Create dummy residuals from semiotic tokens in training
        mask = df_train_mini["before"].astype(str).apply(is_semiotic)
        dummy_res = df_train_mini[mask].copy()
        # Ensure we have enough
        if len(dummy_res) < 10:
            dummy_res = df_train_mini.head(20).copy()

        # Add context columns required by residual processor
        dummy_res["prev"] = dummy_res["before"].shift(1).fillna("<START>")
        dummy_res["next"] = dummy_res["before"].shift(-1).fillna("<END>")

        # Save to the cache path expected by the system
        dummy_res.to_parquet(Config.RESIDUAL_TRAIN_CACHE, index=False)

        # Also ensure val residuals exist
        dummy_res.to_parquet(Config.RESIDUAL_VAL_CACHE, index=False)

        # Reload to verify
        res_train = pd.read_parquet(Config.RESIDUAL_TRAIN_CACHE)
        print(f"Injected {len(res_train)} dummy residuals.")

    # ==========================================
    # 5. Tier 2: Neural Network Training
    # ==========================================
    print("\n[5] Training Neural Network (Tier 2)...")

    # Trainer handles Tokenizer fitting, Dataset creation, and Training Loop
    trainer = Trainer(batch_size=Config.BATCH_SIZE, debug=True)

    # Verify Tokenizers
    print(f"Char Vocab Size: {trainer.tokenizer.char.vocab_size}")
    print(f"BPE Vocab Size: {trainer.tokenizer.bpe.vocab_size()}")
    assert trainer.tokenizer.char.vocab_size > 0

    # Run Training
    trainer.fit()

    # Verify Checkpoint
    assert os.path.exists(
        Config.TRANSFORMER_CHECKPOINT
    ), "Model checkpoint was not saved."
    print("Checkpoint saved successfully.")

    # ==========================================
    # 6. Hybrid Cascade Inference
    # ==========================================
    print("\n[6] Running Hybrid Cascade Inference...")

    manager = CascadeManager()

    # Load the trained model components
    checkpoint = torch.load(Config.TRANSFORMER_CHECKPOINT, map_location=Config.DEVICE)

    # Reconstruct Tokenizers
    char_tokenizer = CharTokenizer()
    char_tokenizer.char2idx = checkpoint["char_vocab"]
    char_tokenizer.idx2char = {v: k for k, v in char_tokenizer.char2idx.items()}
    char_tokenizer.vocab_size = len(char_tokenizer.char2idx)

    bpe_tokenizer = TargetBPETokenizer()
    bpe_tokenizer.load()

    # Reconstruct Model
    model_config = checkpoint["config"]
    model = CharToSubwordTransformer(
        src_vocab_size=model_config["src_vocab_size"],
        tgt_vocab_size=model_config["tgt_vocab_size"],
        pad_idx=char_tokenizer.pad_token_id,
    ).to(Config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Run Inference
    # We use the loaded HFBB and Neural Model
    predictions = manager.run_inference(
        df_test_mini, hfbb, model, char_tokenizer, bpe_tokenizer
    )

    # ==========================================
    # 7. Validation & Submission
    # ==========================================
    print("\n[7] Validating Output...")

    # Check predictions length
    assert len(predictions) == len(
        df_test_mini
    ), f"Prediction count mismatch: {len(predictions)} vs {len(df_test_mini)}"

    # Create submission file
    submission = pd.DataFrame(
        {
            "id": df_test_mini["sentence_id"].astype(str)
            + "_"
            + df_test_mini["token_id"].astype(str),
            "after": predictions,
        }
    )

    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission generated at: {Config.SUBMISSION_FILE}")
    print("Head of submission:")
    print(submission.head())

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
