import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data_processing import load_data
from library.models import PriorInformedTagger, CharLSTMSeq2Seq
from library.trainers import TaggerTrainer, Seq2SeqTrainer
from library.inference import InferencePipeline


def run_demo():
    print("=== Starting Text Normalization Pipeline Demo ===\n")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SIZE = 200  # Small subset for demo
    Config.BPE_VOCAB_SIZE = 1000  # Reduce vocab for small debug data
    Config.TAGGER_EPOCHS = 1
    Config.SEQ2SEQ_EPOCHS = 1
    Config.TAGGER_BATCH_SIZE = 4
    Config.SEQ2SEQ_BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Use a specific demo directory to avoid overwriting existing work
    Config.WORKING_DIR = "./working/demo_run"
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update paths in Config to point to the demo directory
    Config.BPE_MODEL_PREFIX = os.path.join(Config.WORKING_DIR, "bpe_tokenizer")
    Config.VOCAB_WORDS_PATH = os.path.join(Config.WORKING_DIR, "vocab_words.json")
    Config.VOCAB_CHARS_PATH = os.path.join(Config.WORKING_DIR, "vocab_chars.json")
    Config.VOCAB_CLASSES_PATH = os.path.join(Config.WORKING_DIR, "vocab_classes.json")
    Config.KNOWLEDGE_BASE_PATH = os.path.join(
        Config.WORKING_DIR, "knowledge_base.parquet"
    )
    Config.PRIORS_PATH = os.path.join(Config.WORKING_DIR, "priors.parquet")
    Config.TAGGER_TRAIN_DATA = os.path.join(Config.WORKING_DIR, "tagger_train_data.pt")
    Config.TAGGER_VAL_DATA = os.path.join(Config.WORKING_DIR, "tagger_val_data.pt")
    Config.SEQ2SEQ_TRAIN_DATA = os.path.join(
        Config.WORKING_DIR, "seq2seq_train_data.pt"
    )
    Config.SEQ2SEQ_VAL_DATA = os.path.join(Config.WORKING_DIR, "seq2seq_val_data.pt")
    Config.TAGGER_MODEL_PATH = os.path.join(Config.WORKING_DIR, "tagger_best_model.pth")
    Config.SEQ2SEQ_MODEL_PATH = os.path.join(
        Config.WORKING_DIR, "seq2seq_best_model.pth"
    )
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    set_seed(Config.SEED)
    print(f"Working Directory: {Config.WORKING_DIR}")

    # --------------------------------------------------------------------------
    # 2. Data Loading & Processing
    # --------------------------------------------------------------------------
    print("\n[2] Loading and processing data...")
    # load_cached_data=False forces regeneration of data processing steps
    data_resources = load_data(load_cached_data=False, debug=True)

    # Verify resources
    vocab = data_resources["vocab"]
    bpe = data_resources["bpe"]
    kb = data_resources["kb"]
    tagger_train_ds = data_resources["tagger_train"]
    seq2seq_train_ds = data_resources["seq2seq_train"]

    print(
        f"Vocab sizes -> Words: {len(vocab.word2id)}, Chars: {len(vocab.char2id)}, Classes: {len(vocab.class2id)}"
    )
    print(
        f"Dataset sizes -> Tagger Train: {len(tagger_train_ds)}, Seq2Seq Train: {len(seq2seq_train_ds)}"
    )

    # Assertions
    if len(tagger_train_ds) == 0:
        raise AssertionError("Tagger training dataset is empty.")
    if not os.path.exists(Config.VOCAB_WORDS_PATH):
        raise AssertionError("Vocabulary files were not saved.")

    # --------------------------------------------------------------------------
    # 3. Model Initialization & Verification
    # --------------------------------------------------------------------------
    print("\n[3] Initializing models...")

    # --- Tagger ---
    tagger_model = PriorInformedTagger(
        vocab_words=vocab.word2id,
        vocab_classes=vocab.class2id,
        bpe_vocab_size=len(bpe),
        vocab_chars=vocab.char2id,
    ).to(Config.DEVICE)

    # Verify Tagger Forward Pass
    sample_batch = tagger_train_ds[0]
    # Add batch dimension
    word_ids = sample_batch["word_ids"].unsqueeze(0).to(Config.DEVICE)
    bpe_ids = sample_batch["bpe_ids"].unsqueeze(0).to(Config.DEVICE)
    char_ids = sample_batch["char_ids"].unsqueeze(0).to(Config.DEVICE)
    regex_features = sample_batch["regex_features"].unsqueeze(0).to(Config.DEVICE)
    prior_features = sample_batch["prior_features"].unsqueeze(0).to(Config.DEVICE)

    with torch.no_grad():
        logits = tagger_model(
            word_ids, bpe_ids, char_ids, regex_features, prior_features
        )

    print(f"Tagger Output Shape: {logits.shape}")
    # Shape should be (batch_size, seq_len, num_classes)
    expected_shape = (1, Config.MAX_SENT_LEN, len(vocab.class2id))
    if logits.shape != expected_shape:
        raise AssertionError(
            f"Tagger output shape mismatch. Expected {expected_shape}, got {logits.shape}"
        )

    # --- Seq2Seq ---
    seq2seq_model = CharLSTMSeq2Seq(
        vocab_chars=vocab.char2id, vocab_classes=vocab.class2id
    ).to(Config.DEVICE)

    # Verify Seq2Seq Forward Pass
    if len(seq2seq_train_ds) > 0:
        s2s_sample = seq2seq_train_ds[0]
        src_char_ids = s2s_sample["src_char_ids"].unsqueeze(0).to(Config.DEVICE)
        tgt_char_ids = s2s_sample["tgt_char_ids"].unsqueeze(0).to(Config.DEVICE)
        class_id = s2s_sample["class_id"].unsqueeze(0).to(Config.DEVICE)

        with torch.no_grad():
            s2s_out = seq2seq_model(
                src_char_ids, tgt_char_ids, class_id, teacher_forcing_ratio=0.0
            )

        print(f"Seq2Seq Output Shape: {s2s_out.shape}")
        # Shape: (batch, tgt_len-1, vocab_size)
        expected_s2s_shape = (1, Config.SEQ2SEQ_MAX_OUTPUT_LEN - 1, len(vocab.char2id))
        if s2s_out.shape != expected_s2s_shape:
            raise AssertionError(
                f"Seq2Seq output shape mismatch. Expected {expected_s2s_shape}, got {s2s_out.shape}"
            )
    else:
        print("Skipping Seq2Seq forward check (no changed tokens in debug subset).")

    # --------------------------------------------------------------------------
    # 4. Training Simulation
    # --------------------------------------------------------------------------
    print("\n[4] Running training simulation...")

    # Train Tagger
    print("Training Tagger...")
    tagger_trainer = TaggerTrainer(
        tagger_model, tagger_train_ds, data_resources["tagger_val"]
    )
    tagger_trainer.train()

    # Check if model saved
    if not os.path.exists(Config.TAGGER_MODEL_PATH):
        raise AssertionError("Tagger model checkpoint not found after training.")

    # Train Seq2Seq
    if len(seq2seq_train_ds) > 0:
        print("Training Seq2Seq...")
        s2s_trainer = Seq2SeqTrainer(
            seq2seq_model, seq2seq_train_ds, data_resources["seq2seq_val"]
        )
        s2s_trainer.train()

        if not os.path.exists(Config.SEQ2SEQ_MODEL_PATH):
            raise AssertionError("Seq2Seq model checkpoint not found after training.")
    else:
        print(
            "Skipping Seq2Seq training (insufficient data). Saving dummy model for inference."
        )
        torch.save(seq2seq_model.state_dict(), Config.SEQ2SEQ_MODEL_PATH)

    # --------------------------------------------------------------------------
    # 5. Inference Pipeline
    # --------------------------------------------------------------------------
    print("\n[5] Running inference pipeline...")

    # Create a temporary test CSV from the loaded test dataframe for prediction
    test_df = data_resources["test_df"]
    temp_test_csv = os.path.join(Config.WORKING_DIR, "temp_test.csv")
    test_df.to_csv(temp_test_csv, index=False)

    pipeline = InferencePipeline()
    # Load resources (will pick up the models we just trained and saved)
    pipeline.load_resources()

    # Run prediction
    df_submission = pipeline.predict(
        test_csv_path=temp_test_csv, batch_size=Config.TAGGER_BATCH_SIZE
    )

    # Save submission
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # --------------------------------------------------------------------------
    # 6. Final Validation
    # --------------------------------------------------------------------------
    print("\n[6] Validating results...")

    # Check submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError("Submission file was not created.")

    df_check = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Rows: {len(df_check)}")
    print("Head:")
    print(df_check.head())

    # Validate format
    if list(df_check.columns) != ["id", "after"]:
        raise AssertionError(f"Invalid submission columns: {df_check.columns}")

    # Check that we have predictions for the input test set
    if len(df_check) != len(test_df):
        raise AssertionError(
            f"Submission row count ({len(df_check)}) matches test input ({len(test_df)})"
        )

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
