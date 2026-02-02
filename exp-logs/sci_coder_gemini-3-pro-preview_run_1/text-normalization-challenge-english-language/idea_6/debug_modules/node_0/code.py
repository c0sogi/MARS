import os
import sys
import torch
import pandas as pd
import shutil
import time

# Import from the provided library
from library.config import Config
from library.utils import set_seed, get_device
from library.data_loader import (
    get_data,
    get_tagger_loaders,
    get_seq2seq_loaders,
    Vocabulary,
)
from library.models_tagger import BiLSTM_CRF
from library.models_seq2seq import CharTransformer
from library.trainer import train_tagger, train_seq2seq
from library.inference import NormalizationPipeline


def run_demo():
    print("=== Setting up Demo Configuration ===")

    # 1. Override Config for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SIZE = 2000  # Small subset for speed
    Config.NUM_EPOCHS = 1  # Single epoch
    Config.BATCH_SIZE = 16

    # Use a specific working directory for this demo to avoid conflicts
    Config.WORK_DIR = "./working/demo_execution"
    if os.path.exists(Config.WORK_DIR):
        shutil.rmtree(Config.WORK_DIR)
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    # Update dependent paths in Config to point to the new working directory
    Config.VOCAB_TOKENS_PATH = os.path.join(Config.WORK_DIR, "vocab_tokens.parquet")
    Config.VOCAB_CLASSES_PATH = os.path.join(Config.WORK_DIR, "vocab_classes.parquet")
    Config.VOCAB_CHARS_PATH = os.path.join(Config.WORK_DIR, "vocab_chars.parquet")
    Config.KNOWLEDGE_BASE_PATH = os.path.join(Config.WORK_DIR, "knowledge_base.parquet")

    Config.TAGGER_MODEL_PATH = os.path.join(Config.WORK_DIR, "tagger_demo.pth")
    Config.SEQ2SEQ_MODEL_PATH = os.path.join(Config.WORK_DIR, "seq2seq_demo.pth")

    Config.SUBMISSION_DIR = os.path.join(Config.WORK_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Setup directories
    Config.setup()
    set_seed(42)
    device = get_device()
    print(f"Running on device: {device}")

    print("\n=== Step 1: Data Processing & Loading ===")
    # Force processing (load_cached=False) to generate artifacts in our new WORK_DIR
    vocab_tokens, vocab_chars, vocab_classes, train_g, val_g, test_g, s2s_df = get_data(
        load_cached=False
    )

    # Verification
    print(f"Vocab Tokens Size: {len(vocab_tokens)}")
    print(f"Vocab Chars Size: {len(vocab_chars)}")
    print(f"Vocab Classes Size: {len(vocab_classes)}")
    print(f"Train Grouped Shape: {train_g.shape}")

    assert len(vocab_tokens) > 0, "Token vocabulary is empty"
    assert len(vocab_chars) > 0, "Char vocabulary is empty"
    assert not train_g.empty, "Train grouped dataframe is empty"
    assert os.path.exists(Config.VOCAB_TOKENS_PATH), "Vocab file not saved"

    print("\n=== Step 2: Tagger Model Verification ===")
    # Get loaders
    train_loader, _, _, vt, vc, vcl = get_tagger_loaders(batch_size=4, load_cached=True)
    batch = next(iter(train_loader))

    token_ids = batch["token_ids"].to(device)
    char_ids = batch["char_ids"].to(device)
    mask = batch["mask"].to(device)

    print(f"Batch Token IDs Shape: {token_ids.shape}")
    print(f"Batch Char IDs Shape: {char_ids.shape}")

    # Instantiate Model
    tagger = BiLSTM_CRF(
        vocab_size=len(vt), char_vocab_size=len(vc), num_classes=len(vcl)
    ).to(device)

    # Forward Pass
    emissions = tagger(token_ids, char_ids, mask)
    print(f"Tagger Emissions Shape: {emissions.shape}")

    assert emissions.shape == (
        4,
        Config.MAX_SEQ_LEN,
        len(vcl),
    ), "Tagger output shape mismatch"

    # Decode
    tags = tagger.decode(token_ids, char_ids, mask)
    assert tags.shape == (4, Config.MAX_SEQ_LEN), "Tagger decode shape mismatch"

    print("\n=== Step 3: Running Tagger Training Loop ===")
    # Run the provided training function
    train_tagger(load_cached=True)
    assert os.path.exists(
        Config.TAGGER_MODEL_PATH
    ), "Tagger model checkpoint not found after training"

    print("\n=== Step 4: Seq2Seq Model Verification ===")
    # Get loaders
    s2s_train_loader, _, vc_s2s, vcl_s2s = get_seq2seq_loaders(
        batch_size=4, load_cached=True
    )

    # Check if we have data for Seq2Seq (might be empty if no changes in debug subset)
    if len(s2s_train_loader) > 0:
        batch_s2s = next(iter(s2s_train_loader))
        src_ids = batch_s2s["src_ids"].to(device)
        tgt_ids = batch_s2s["tgt_ids"].to(device)
        class_id = batch_s2s["class_id"].to(device)

        print(f"Seq2Seq Src Shape: {src_ids.shape}")
        print(f"Seq2Seq Tgt Shape: {tgt_ids.shape}")

        # Instantiate Model
        seq2seq = CharTransformer(num_chars=len(vc_s2s), num_classes=len(vcl_s2s)).to(
            device
        )

        # Forward Pass
        # Decoder input is tgt without last token
        dec_input = tgt_ids[:, :-1]
        logits = seq2seq(src_ids, dec_input, class_id)
        print(f"Seq2Seq Logits Shape: {logits.shape}")

        # Expected output shape: (Batch, Max_Len - 1, Vocab)
        assert logits.shape == (
            4,
            Config.SEQ2SEQ_MAX_OUTPUT_LEN - 1,
            len(vc_s2s),
        ), "Seq2Seq output shape mismatch"

        print("\n=== Step 5: Running Seq2Seq Training Loop ===")
        train_seq2seq(load_cached=True)
        assert os.path.exists(
            Config.SEQ2SEQ_MODEL_PATH
        ), "Seq2Seq model checkpoint not found after training"
    else:
        print("Skipping Seq2Seq verification: No changed tokens in debug subset.")
        # Create a dummy checkpoint so inference doesn't fail
        dummy_model = CharTransformer(len(vc_s2s), len(vcl_s2s)).to(device)
        state = {"model_state_dict": dummy_model.state_dict()}
        torch.save(state, Config.SEQ2SEQ_MODEL_PATH)

    print("\n=== Step 6: Inference Pipeline ===")
    pipeline = NormalizationPipeline(load_cached=True)
    pipeline.predict(batch_size=16)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    # Validate Submission Format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Rows: {len(df_sub)}")
    print(df_sub.head())

    assert (
        "id" in df_sub.columns and "after" in df_sub.columns
    ), "Submission columns missing"
    assert len(df_sub) > 0, "Submission is empty"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
