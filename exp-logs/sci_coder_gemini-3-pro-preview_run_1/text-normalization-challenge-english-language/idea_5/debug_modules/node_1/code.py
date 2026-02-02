import os
import shutil
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.data_utils import (
    build_vocabularies,
    build_knowledge_base,
    load_and_group_data,
)
from library.datasets import (
    TaggerDataset,
    tagger_collate_fn,
    Seq2SeqDataset,
    seq2seq_collate_fn,
)
from library.models import BiLSTMTagger, Seq2SeqModel
from library.engine import Trainer
from library.inference import NormalizationPipeline


def setup_demo_environment():
    """
    Sets up a temporary environment for the demo run.
    Overrides Config paths and hyperparameters for speed.
    """
    print(">>> Setting up demo environment...")

    # Define demo directories
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config paths
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    Config.VOCAB_WORDS_PATH = os.path.join(demo_dir, "vocab_words.parquet")
    Config.VOCAB_CHARS_PATH = os.path.join(demo_dir, "vocab_chars.parquet")
    Config.VOCAB_CLASSES_PATH = os.path.join(demo_dir, "vocab_classes.parquet")
    Config.KNOWLEDGE_BASE_PATH = os.path.join(demo_dir, "knowledge_base.parquet")

    Config.TRAIN_GROUPED_PATH = os.path.join(demo_dir, "train_grouped.parquet")
    Config.VAL_GROUPED_PATH = os.path.join(demo_dir, "val_grouped.parquet")
    Config.TEST_GROUPED_PATH = os.path.join(demo_dir, "test_grouped.parquet")

    Config.TAGGER_MODEL_PATH = os.path.join(demo_dir, "tagger_demo.pth")
    Config.SEQ2SEQ_MODEL_PATH = os.path.join(demo_dir, "seq2seq_demo.pth")
    Config.SUBMISSION_OUTPUT_PATH = os.path.join(
        Config.SUBMISSION_DIR, "submission.csv"
    )

    # Override Data Params for Speed
    Config.MAX_VOCAB_SIZE = 1000
    Config.MAX_SEQ_LEN = 32
    Config.MAX_CHAR_LEN = 10
    Config.SEQ2SEQ_MAX_LEN = 20

    # Override Model Params for Speed (Tiny Models)
    Config.TAGGER_EMBED_DIM = 32
    Config.TAGGER_CHAR_EMBED_DIM = 16
    Config.TAGGER_CHAR_CNN_FILTERS = 16
    Config.TAGGER_HIDDEN_DIM = 32
    Config.TAGGER_NUM_LAYERS = 1
    Config.TAGGER_BIDIRECTIONAL = True  # Keep bidirectional to test logic

    Config.SEQ2SEQ_EMBED_DIM = 32
    Config.SEQ2SEQ_HIDDEN_DIM = 64

    # Override Training Params
    Config.BATCH_SIZE = 8
    Config.SEQ2SEQ_BATCH_SIZE = 8
    Config.NUM_EPOCHS = 1
    Config.SEQ2SEQ_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Create a subset of training data for the demo
    # We read the original metadata but only keep top 2000 rows
    print(">>> Creating data subset...")
    original_train_path = "./metadata/train.csv"
    demo_train_path = os.path.join(demo_dir, "demo_train.csv")

    df = pd.read_csv(original_train_path, nrows=2000, dtype=str, keep_default_na=False)
    df.to_csv(demo_train_path, index=False)

    # Point Config to this new subset file for 'train'
    Config.TRAIN_DATA_PATH = demo_train_path
    # Use the same subset for validation to ensure it exists and runs
    Config.VAL_DATA_PATH = demo_train_path

    # Create a dummy test set
    demo_test_path = os.path.join(demo_dir, "demo_test.csv")
    # Take first 100 rows, drop 'after' and 'class'
    df_test = df.head(100)[["sentence_id", "token_id", "before", "id"]].copy()
    df_test.to_csv(demo_test_path, index=False)
    Config.TEST_DATA_PATH = demo_test_path

    return df


def demonstrate_data_utils(df_train):
    print("\n>>> Demonstrating Data Utils...")

    # 1. Build Vocabularies
    # Force rebuild (load_cached_data=False)
    vocab_words, vocab_chars, vocab_classes = build_vocabularies(
        df_train, load_cached_data=False
    )

    print(f"Word Vocab Size: {len(vocab_words)}")
    print(f"Char Vocab Size: {len(vocab_chars)}")
    print(f"Class Vocab Size: {len(vocab_classes)}")

    assert len(vocab_words) > 0
    assert len(vocab_chars) > 0
    assert len(vocab_classes) > 0

    # 2. Build Knowledge Base
    kb = build_knowledge_base(df_train, load_cached_data=False)
    print(f"Knowledge Base Size: {len(kb)}")

    # Verify a known entry if possible, or just type
    assert isinstance(kb, dict)

    # 3. Load and Group Data
    # This uses Config.TRAIN_DATA_PATH which we pointed to our demo subset
    df_grouped = load_and_group_data("train", load_cached_data=False)
    print(f"Grouped Data Rows (Sentences): {len(df_grouped)}")

    assert "before" in df_grouped.columns
    assert isinstance(df_grouped.iloc[0]["before"], (list, np.ndarray))

    return vocab_words, vocab_chars, vocab_classes, df_grouped


def demonstrate_tagger_components(vocab_words, vocab_chars, vocab_classes, df_grouped):
    print("\n>>> Demonstrating Tagger Components...")

    # 1. Dataset
    dataset = TaggerDataset(
        df_grouped, vocab_words, vocab_chars, vocab_classes, split="train"
    )
    sample = dataset[0]

    print("Sample Tagger Dataset Item Keys:", sample.keys())
    assert "word_ids" in sample
    assert "char_ids" in sample
    assert "class_ids" in sample

    # 2. DataLoader & Collate
    loader = DataLoader(
        dataset, batch_size=Config.BATCH_SIZE, collate_fn=tagger_collate_fn
    )
    batch = next(iter(loader))

    print("Tagger Batch Shapes:")
    print(f"  Word IDs: {batch['word_ids'].shape}")  # (Batch, Seq)
    print(f"  Char IDs: {batch['char_ids'].shape}")  # (Batch, Seq, Char)
    print(f"  Class IDs: {batch['class_ids'].shape}")  # (Batch, Seq)

    assert batch["word_ids"].dim() == 2
    assert batch["char_ids"].dim() == 3

    # 3. Model
    model = BiLSTMTagger(
        vocab_size=len(vocab_words),
        num_classes=len(vocab_classes),
        char_vocab_size=len(vocab_chars),
    )

    # Forward Pass
    logits = model(batch["word_ids"], batch["char_ids"])
    print(f"Tagger Output Logits Shape: {logits.shape}")  # (Batch, Seq, Num_Classes)

    assert logits.shape[0] == Config.BATCH_SIZE
    assert logits.shape[2] == len(vocab_classes)

    return model, loader


def demonstrate_seq2seq_components(df_raw, vocab_chars):
    print("\n>>> Demonstrating Seq2Seq Components...")

    # 1. Dataset
    dataset = Seq2SeqDataset(df_raw, vocab_chars, split="train")
    if len(dataset) == 0:
        print(
            "Warning: No changed tokens found in subset. Creating dummy entry for demo."
        )
        # Create a dummy row where before != after
        dummy_data = pd.DataFrame(
            [
                {
                    "before": "dummy",
                    "after": "normalized",
                    "class": "PLAIN",
                    "sentence_id": "0",
                    "token_id": "0",
                    "id": "0_0",
                }
            ]
        )
        dataset = Seq2SeqDataset(dummy_data, vocab_chars, split="train")

    sample = dataset[0]
    print("Sample Seq2Seq Dataset Item Keys:", sample.keys())

    # 2. DataLoader
    loader = DataLoader(
        dataset, batch_size=Config.SEQ2SEQ_BATCH_SIZE, collate_fn=seq2seq_collate_fn
    )
    batch = next(iter(loader))

    print("Seq2Seq Batch Shapes:")
    print(f"  Src IDs: {batch['src_ids'].shape}")
    if batch["tgt_ids"] is not None:
        print(f"  Tgt IDs: {batch['tgt_ids'].shape}")

    # 3. Model
    model = Seq2SeqModel(num_chars=len(vocab_chars))

    # Forward Pass (Training mode)
    if batch["tgt_ids"] is not None:
        outputs = model(batch["src_ids"], batch["tgt_ids"])
        print(
            f"Seq2Seq Training Output Shape: {outputs.shape}"
        )  # (Batch, Max_Len, Vocab)
        assert outputs.shape[0] == batch["src_ids"].shape[0]
        assert outputs.shape[2] == len(vocab_chars)

    # Generate Pass (Inference mode)
    sos_idx = vocab_chars.stoi[Config.SOS_TOKEN]
    eos_idx = vocab_chars.stoi[Config.EOS_TOKEN]

    generated = model.generate(batch["src_ids"], sos_idx, eos_idx)
    print(f"Seq2Seq Generation Output Shape: {generated.shape}")  # (Batch, Seq)

    return model, loader


def demonstrate_training(
    tagger_model,
    tagger_loader,
    seq2seq_model,
    seq2seq_loader,
    df_grouped,
    vocab_classes,
):
    print("\n>>> Demonstrating Training Engine...")

    trainer = Trainer(device=Config.DEVICE)

    # 1. Train Tagger
    # Calculate class weights
    weights = trainer.get_class_weights(df_grouped, vocab_classes)

    # Run 1 epoch
    trained_tagger = trainer.train_tagger(
        tagger_model, tagger_loader, tagger_loader, class_weights=weights
    )
    assert os.path.exists(
        Config.TAGGER_MODEL_PATH
    ), "Tagger model checkpoint not found!"

    # 2. Train Seq2Seq
    trained_seq2seq = trainer.train_seq2seq(
        seq2seq_model, seq2seq_loader, seq2seq_loader
    )
    assert os.path.exists(
        Config.SEQ2SEQ_MODEL_PATH
    ), "Seq2Seq model checkpoint not found!"

    return trained_tagger, trained_seq2seq


def demonstrate_inference():
    print("\n>>> Demonstrating Inference Pipeline...")

    # The pipeline loads models from disk (which we just saved in training demo)
    # and data from Config.TEST_DATA_PATH (which we set up)

    pipeline = NormalizationPipeline()

    # Run prediction
    # This generates submission.csv
    pipeline.predict(batch_size=Config.BATCH_SIZE)

    assert os.path.exists(
        Config.SUBMISSION_OUTPUT_PATH
    ), "Submission file not generated!"

    # Validate submission format
    df_sub = pd.read_csv(Config.SUBMISSION_OUTPUT_PATH)
    print(f"Submission shape: {df_sub.shape}")
    print("Submission Head:")
    print(df_sub.head())

    assert "id" in df_sub.columns
    assert "after" in df_sub.columns
    assert len(df_sub) > 0


if __name__ == "__main__":
    # Ensure reproducible results
    Config.set_seed(42)

    # 1. Setup
    df_raw_subset = setup_demo_environment()

    # 2. Data Utils
    vocab_words, vocab_chars, vocab_classes, df_grouped = demonstrate_data_utils(
        df_raw_subset
    )

    # 3. Tagger Components
    tagger_model, tagger_loader = demonstrate_tagger_components(
        vocab_words, vocab_chars, vocab_classes, df_grouped
    )

    # 4. Seq2Seq Components
    seq2seq_model, seq2seq_loader = demonstrate_seq2seq_components(
        df_raw_subset, vocab_chars
    )

    # 5. Training Loop
    demonstrate_training(
        tagger_model,
        tagger_loader,
        seq2seq_model,
        seq2seq_loader,
        df_grouped,
        vocab_classes,
    )

    # 6. Inference
    demonstrate_inference()

    print("\n>>> Demo Completed Successfully!")
