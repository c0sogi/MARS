import pandas as pd
import torch
import os
import shutil
import numpy as np
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.vocabulary import get_vocab
from library.dataset import TextNormalizationDataset
from library.model import Encoder, Decoder, Attention, Seq2Seq
from library.trainer import Trainer
from library.inference import generate_submission


def run_demo():
    print("=====================================================")
    print("   Text Normalization Pipeline Demonstration")
    print("=====================================================")

    # ---------------------------------------------------------
    # 1. Configuration Setup
    # ---------------------------------------------------------
    print("\n[1] Setting up Configuration for Demo...")

    # Modify Config for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SIZE = 100  # Small subset size
    Config.N_EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size

    # Reduce Model Complexity for fast initialization/forward pass
    Config.ENC_EMB_DIM = 32
    Config.DEC_EMB_DIM = 32
    Config.HIDDEN_DIM = 64
    Config.N_LAYERS = 1

    # Disable downsampling to ensure our small dummy dataset isn't filtered out
    Config.PLAIN_DOWNSAMPLE_RATIO = 1.0

    # Deterministic behavior
    Config.TEACHER_FORCING_RATIO = 0.0

    # Setup temporary directory for demo outputs
    demo_dir = "./working/demo"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Redirect Config paths to the demo directory
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")

    # Update dependent file paths
    Config.TRAIN_CACHE = os.path.join(Config.CACHE_DIR, "train.parquet")
    Config.VAL_CACHE = os.path.join(Config.CACHE_DIR, "val.parquet")
    Config.TEST_CACHE = os.path.join(Config.CACHE_DIR, "test.parquet")
    Config.VOCAB_CACHE = os.path.join(Config.CACHE_DIR, "vocab.npy")
    Config.MODEL_SAVE_PATH = os.path.join(Config.CACHE_DIR, "model.pt")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Point input files to dummy CSVs we will create shortly
    Config.TRAIN_FILE = os.path.join(demo_dir, "dummy_train.csv")
    Config.VAL_FILE = os.path.join(demo_dir, "dummy_val.csv")
    Config.TEST_FILE = os.path.join(demo_dir, "dummy_test.csv")

    # Initialize directories
    Config.setup()
    Config.set_seed()
    print("    Configuration updated and directories created.")

    # ---------------------------------------------------------
    # 2. Synthetic Data Generation
    # ---------------------------------------------------------
    print("\n[2] Creating Synthetic Data...")

    # Dummy Train Data (includes context and normalization examples)
    train_data = {
        "sentence_id": [0, 0, 0, 1, 1],
        "token_id": [0, 1, 2, 0, 1],
        "class": ["PLAIN", "PLAIN", "PUNCT", "DATE", "PLAIN"],
        "before": ["The", "cat", ".", "2023", "year"],
        "after": ["The", "cat", ".", "twenty twenty three", "year"],
        "id": ["0_0", "0_1", "0_2", "1_0", "1_1"],
    }
    pd.DataFrame(train_data).to_csv(Config.TRAIN_FILE, index=False)

    # Dummy Validation Data
    val_data = {
        "sentence_id": [10, 10],
        "token_id": [0, 1],
        "class": ["PLAIN", "PLAIN"],
        "before": ["Hello", "World"],
        "after": ["Hello", "World"],
        "id": ["10_0", "10_1"],
    }
    pd.DataFrame(val_data).to_csv(Config.VAL_FILE, index=False)

    # Dummy Test Data (No 'after' or 'class' columns)
    test_data = {
        "sentence_id": [20, 20],
        "token_id": [0, 1],
        "before": ["Test", "run"],
        "id": ["20_0", "20_1"],
    }
    pd.DataFrame(test_data).to_csv(Config.TEST_FILE, index=False)
    print("    Dummy CSV files created in ./working/demo/")

    # ---------------------------------------------------------
    # 3. Vocabulary Demonstration
    # ---------------------------------------------------------
    print("\n[3] Building Vocabulary...")

    # Force build from the dummy train file by setting load_cached_data=False
    vocab = get_vocab(load_cached_data=False)
    print(f"    Vocabulary size: {len(vocab)}")

    # Verify encoding/decoding logic
    test_str = "cat"
    encoded = vocab.encode(test_str, add_sos=True, add_eos=True)
    decoded = vocab.decode(encoded, remove_special=True)

    print(f"    Encoding '{test_str}': {encoded}")
    print(f"    Decoded: '{decoded}'")

    # Assertions
    assert decoded == test_str, "Vocabulary decode failed to reconstruct string."
    assert (
        vocab.stoi[Config.SOS_TOKEN] == encoded[0]
    ), "Encoded sequence must start with SOS."
    assert (
        vocab.stoi[Config.EOS_TOKEN] == encoded[-1]
    ), "Encoded sequence must end with EOS."
    print("    Vocabulary logic verified.")

    # ---------------------------------------------------------
    # 4. Dataset Demonstration
    # ---------------------------------------------------------
    print("\n[4] Loading and Processing Dataset...")

    # Load Train Dataset
    train_ds = TextNormalizationDataset("train", vocab, load_cached_data=False)
    print(f"    Train dataset size: {len(train_ds)}")

    # Verify Item Structure
    item = train_ds[0]
    # Item 0 is "The" (0_0). Prev is empty. Next is "cat". Input: "|The|cat"
    src_text = vocab.decode(item["src"], remove_special=True)

    print(f"    Sample Item 0 (Input Text): '{src_text}'")
    print(
        f"    Sample Item 0 (Target Text): '{vocab.decode(item['tgt'], remove_special=True)}'"
    )

    assert "src" in item and "tgt" in item, "Dataset item missing keys."
    assert isinstance(item["src"], torch.Tensor), "Source is not a tensor."
    assert Config.SEP_TOKEN in src_text, "Context separator token missing in input."

    # Verify Collate Function
    batch_list = [train_ds[i] for i in range(min(len(train_ds), 3))]
    batch = TextNormalizationDataset.collate_fn(batch_list)
    print(f"    Batch 'src' shape: {batch['src'].shape}")

    assert batch["src"].shape[0] == 3, "Batch size mismatch in collate_fn."
    assert batch["src"].ndim == 2, "Batch src should be 2D [Batch, Seq_Len]."
    print("    Dataset and Collate logic verified.")

    # ---------------------------------------------------------
    # 5. Model Demonstration
    # ---------------------------------------------------------
    print("\n[5] Initializing Model...")
    device = Config.get_device()

    attn = Attention(Config.HIDDEN_DIM, Config.HIDDEN_DIM)
    enc = Encoder(
        len(vocab), Config.ENC_EMB_DIM, Config.HIDDEN_DIM, Config.HIDDEN_DIM, 0.1
    )
    dec = Decoder(
        len(vocab), Config.DEC_EMB_DIM, Config.HIDDEN_DIM, Config.HIDDEN_DIM, 0.1, attn
    )
    model = Seq2Seq(enc, dec, device).to(device)

    # Run Dummy Forward Pass
    src = batch["src"].to(device)
    src_len = batch["src_len"].to("cpu")
    tgt = batch["tgt"].to(device)

    output = model(src, src_len, tgt)
    print(f"    Model output shape: {output.shape}")

    # Expected: [Batch Size, Target Length, Vocab Size]
    assert output.shape == (3, tgt.shape[1], len(vocab)), "Model output shape mismatch."
    print("    Model forward pass verified.")

    # ---------------------------------------------------------
    # 6. Training Demonstration
    # ---------------------------------------------------------
    print("\n[6] Running Training Loop (1 Epoch)...")

    val_ds = TextNormalizationDataset("val", vocab, load_cached_data=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        collate_fn=TextNormalizationDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        collate_fn=TextNormalizationDataset.collate_fn,
    )

    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.fit()

    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model file was not saved after training."
    print(f"    Model saved to {Config.MODEL_SAVE_PATH}")

    # ---------------------------------------------------------
    # 7. Inference Demonstration
    # ---------------------------------------------------------
    print("\n[7] Generating Submission...")

    # Generate submission using the trained model and dummy test data
    # We set load_cached_data=False to ensure it processes the dummy test CSV
    generate_submission(load_cached_data=False, debug=True)

    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not generated."

    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"    Submission file rows: {len(sub_df)}")
    print("    Head of submission:")
    print(sub_df.head())

    # Verify row count matches dummy test data
    expected_count = pd.read_csv(Config.TEST_FILE).shape[0]
    assert (
        len(sub_df) == expected_count
    ), f"Submission row count ({len(sub_df)}) does not match test set ({expected_count})."

    print("\n=====================================================")
    print("   Demonstration Completed Successfully")
    print("=====================================================")


if __name__ == "__main__":
    run_demo()
