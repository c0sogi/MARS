import os
import sys
import torch
import pandas as pd
import shutil
from torch.utils.data import DataLoader

# Import library modules
from library.config import ProjectConfig, TrainingConfig, DataConfig, set_seed
from library.data_utils import (
    load_dataset_raw,
    build_vocabularies,
    build_knowledge_base,
    TaggerDataset,
    Seq2SeqDataset,
    collate_fn_tagger,
    collate_fn_seq2seq,
)
from library.models import MultiGranularityTagger, Seq2SeqNormalizer
from library.engine import Trainer
from library.inference import CascadePredictor


def run_demo():
    print("=" * 50)
    print("Start of Library Usage Demo")
    print("=" * 50)

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("\n[Step 1] Configuring Environment for Demo...")

    # Override ProjectConfig to use a separate working directory for this demo
    ProjectConfig.BASE_DIR = "./working/demo_execution"
    os.makedirs(ProjectConfig.BASE_DIR, exist_ok=True)

    # Update artifact paths to be inside the new BASE_DIR
    ProjectConfig.VOCAB_WORDS_PATH = os.path.join(
        ProjectConfig.BASE_DIR, "vocab_words.parquet"
    )
    ProjectConfig.VOCAB_CHARS_PATH = os.path.join(
        ProjectConfig.BASE_DIR, "vocab_chars.parquet"
    )
    ProjectConfig.VOCAB_CLASSES_PATH = os.path.join(
        ProjectConfig.BASE_DIR, "vocab_classes.parquet"
    )
    ProjectConfig.KNOWLEDGE_BASE_PATH = os.path.join(
        ProjectConfig.BASE_DIR, "knowledge_base.parquet"
    )
    ProjectConfig.TAGGER_MODEL_PATH = os.path.join(
        ProjectConfig.BASE_DIR, "tagger_demo.pth"
    )
    ProjectConfig.SEQ2SEQ_MODEL_PATH = os.path.join(
        ProjectConfig.BASE_DIR, "seq2seq_demo.pth"
    )
    ProjectConfig.SUBMISSION_PATH = os.path.join(
        ProjectConfig.BASE_DIR, "submission_demo.csv"
    )

    # Enable DEBUG mode to load only a small subset of data for speed
    ProjectConfig.DEBUG = True
    ProjectConfig.DEBUG_SIZE = 2000  # Only load 2000 rows

    # Reduce Training parameters for speed
    TrainingConfig.TAGGER_EPOCHS = 1
    TrainingConfig.SEQ_EPOCHS = 1
    TrainingConfig.TAGGER_BATCH_SIZE = 16
    TrainingConfig.SEQ_BATCH_SIZE = 16

    # Set Seed
    set_seed(TrainingConfig.SEED)
    print("Configuration updated. DEBUG mode enabled.")

    # ---------------------------------------------------------
    # 2. Data Preparation (Vocab & Knowledge Base)
    # ---------------------------------------------------------
    print("\n[Step 2] Building Vocabularies and Knowledge Base...")

    # This will trigger loading the raw CSV (truncated by DEBUG_SIZE) and building artifacts
    vocab_words, vocab_chars, vocab_classes = build_vocabularies(load_cached_data=False)
    kb_dict = build_knowledge_base(load_cached_data=False)

    # Validation
    print(f"  - Word Vocab Size: {len(vocab_words)}")
    print(f"  - Char Vocab Size: {len(vocab_chars)}")
    print(f"  - Class Vocab Size: {len(vocab_classes)}")
    print(f"  - Knowledge Base Size: {len(kb_dict)}")

    if len(vocab_words) == 0 or len(vocab_chars) == 0:
        raise AssertionError("Vocabularies are empty!")
    if DataConfig.PAD_TOKEN not in vocab_words.stoi:
        raise AssertionError("PAD token missing from word vocab!")

    print("Vocabularies and Knowledge Base built successfully.")

    # ---------------------------------------------------------
    # 3. Dataset & DataLoader Instantiation
    # ---------------------------------------------------------
    print("\n[Step 3] Verifying Datasets and DataLoaders...")

    # Load the raw dataframe (cached by previous steps or reloaded)
    df_train = load_dataset_raw("train")
    print(f"  - Loaded Train DataFrame with {len(df_train)} rows (Debug Subset).")

    # Tagger Dataset
    tagger_dataset = TaggerDataset(df_train, vocab_words, vocab_chars, vocab_classes)
    tagger_loader = DataLoader(
        tagger_dataset, batch_size=4, collate_fn=collate_fn_tagger, shuffle=True
    )

    # Verify Tagger Batch
    tagger_batch = next(iter(tagger_loader))
    print("  - Tagger Batch Keys:", tagger_batch.keys())
    if tagger_batch["word_ids"].shape[0] != 4:
        raise AssertionError("Tagger batch size incorrect.")
    if "raw_texts" not in tagger_batch:
        raise AssertionError("Tagger batch missing raw_texts.")

    # Seq2Seq Dataset
    # Seq2Seq dataset filters for changed tokens only.
    # In a small random subset, there might be few. We ensure we handle empty or small datasets gracefully.
    seq2seq_dataset = Seq2SeqDataset(df_train, vocab_chars, vocab_classes)
    print(f"  - Seq2Seq Dataset Size (Changed Tokens): {len(seq2seq_dataset)}")

    if len(seq2seq_dataset) > 0:
        seq_loader = DataLoader(
            seq2seq_dataset, batch_size=4, collate_fn=collate_fn_seq2seq, shuffle=True
        )
        seq_batch = next(iter(seq_loader))
        print("  - Seq2Seq Batch Keys:", seq_batch.keys())
        if seq_batch["src_char_ids"].dim() != 2:
            raise AssertionError("Seq2Seq src tensor has wrong dimensions.")
    else:
        print(
            "  - Warning: No changed tokens in debug subset for Seq2Seq. Skipping batch check."
        )

    # ---------------------------------------------------------
    # 4. Model Training
    # ---------------------------------------------------------
    print("\n[Step 4] Demonstrating Model Training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(device=device)

    # --- Train Tagger ---
    print("  -> Initializing Tagger Model...")
    tagger_model = MultiGranularityTagger(
        len(vocab_words), len(vocab_chars), len(vocab_classes)
    )

    # Create valid loader for training loop (using same subset for demo)
    # We use a slightly larger batch size for the loop
    train_loader_tagger = DataLoader(
        tagger_dataset,
        batch_size=TrainingConfig.TAGGER_BATCH_SIZE,
        collate_fn=collate_fn_tagger,
    )

    print("  -> Running Tagger Training Loop...")
    trained_tagger = trainer.train_tagger(
        tagger_model,
        train_loader_tagger,
        train_loader_tagger,  # Use train as val for demo
        epochs=1,
    )

    # Check if model saved
    if not os.path.exists(ProjectConfig.TAGGER_MODEL_PATH):
        raise AssertionError("Tagger model file was not saved!")
    print("  -> Tagger training complete and model saved.")

    # --- Train Seq2Seq ---
    if len(seq2seq_dataset) > 0:
        print("  -> Initializing Seq2Seq Model...")
        seq2seq_model = Seq2SeqNormalizer(len(vocab_chars), len(vocab_classes))

        train_loader_seq = DataLoader(
            seq2seq_dataset,
            batch_size=TrainingConfig.SEQ_BATCH_SIZE,
            collate_fn=collate_fn_seq2seq,
        )

        print("  -> Running Seq2Seq Training Loop...")
        trained_seq2seq = trainer.train_seq2seq(
            seq2seq_model,
            train_loader_seq,
            train_loader_seq,  # Use train as val for demo
            epochs=1,
        )

        if not os.path.exists(ProjectConfig.SEQ2SEQ_MODEL_PATH):
            raise AssertionError("Seq2Seq model file was not saved!")
        print("  -> Seq2Seq training complete and model saved.")
    else:
        print(
            "  -> Skipping Seq2Seq training (no data). Creating dummy file for Inference step."
        )
        # Create a dummy model file so Inference step doesn't crash
        dummy_model = Seq2SeqNormalizer(len(vocab_chars), len(vocab_classes))
        torch.save(dummy_model.state_dict(), ProjectConfig.SEQ2SEQ_MODEL_PATH)

    # ---------------------------------------------------------
    # 5. Inference & Prediction
    # ---------------------------------------------------------
    print("\n[Step 5] Demonstrating Inference with CascadePredictor...")

    # Initialize Predictor (loads models from disk)
    predictor = CascadePredictor(device=device)

    # Create a dummy test batch
    # We manually create a mock batch similar to what DataLoader produces
    # Let's use the first 5 items from the tagger dataset as "test" samples
    test_samples = [tagger_dataset[i] for i in range(min(5, len(tagger_dataset)))]
    test_batch = collate_fn_tagger(test_samples)

    print("  -> Running Prediction on batch...")
    predictions = predictor.predict_batch(test_batch)

    print("  -> Prediction Results:")
    for i, res in enumerate(predictions):
        print(
            f"     ID: {res['id']}, Input: {test_batch['raw_texts'][i]}, Output: {res['after']}"
        )

        # Validation
        if "id" not in res or "after" not in res:
            raise AssertionError("Prediction output format is incorrect.")
        if not isinstance(res["after"], str):
            raise AssertionError("Prediction 'after' is not a string.")

    print("\n[Success] All demo steps completed successfully.")
    print(f"Artifacts stored in: {ProjectConfig.BASE_DIR}")


if __name__ == "__main__":
    run_demo()
