import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import library components
from library.config import Config
from library.vocab import Vocabulary
from library.data import InterleavedDataset, collate_fn
from library.model import BifurcatedTransformer
from library.trainer import Trainer
from library.inference import Predictor


def run_demo():
    print("--- Starting Bifurcated Transformer Demo ---")

    # --------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed and Demo Isolation
    # --------------------------------------------------------------------------
    print("Configuring environment...")

    # Define demo paths
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config class attributes directly
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")
    Config.MODEL_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.VOCAB_PATH = os.path.join(DEMO_DIR, "vocab.npy")

    # Override Data Paths
    Config.TRAIN_METADATA = os.path.join(DEMO_DIR, "train_small.csv")
    Config.VAL_METADATA = os.path.join(DEMO_DIR, "val_small.csv")
    Config.TEST_METADATA = os.path.join(DEMO_DIR, "test_small.csv")

    # Reduce Model Size for CPU/Fast Execution
    Config.VOCAB_SIZE = 100  # Small vocab
    Config.EMBED_DIM = 32
    Config.NHEAD = 2
    Config.DIM_FEEDFORWARD = 64
    Config.SHARED_LAYERS = 1
    Config.BRANCH_LAYERS = 1
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    Config.MAX_SEQ_LEN = 32

    # Ensure reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    # --------------------------------------------------------------------------
    # 2. Generate Synthetic Data
    # --------------------------------------------------------------------------
    print("Generating synthetic datasets...")

    # Train data: Simple sentences
    train_data = {
        "id": range(20),
        "sentence": [
            "The quick brown fox jumps over the lazy dog .",
            "Machine learning is fascinating and powerful .",
            "Python is a great programming language for data .",
            "Neural networks can learn complex patterns .",
            "The weather today is sunny and bright .",
        ]
        * 4,
    }
    pd.DataFrame(train_data).to_csv(Config.TRAIN_METADATA, index=False)

    # Val data
    val_data = {
        "id": range(20, 30),
        "sentence": [
            "Validation sets help prevent overfitting models .",
            "Accuracy is a common metric for classification .",
        ]
        * 5,
    }
    pd.DataFrame(val_data).to_csv(Config.VAL_METADATA, index=False)

    # Test data (One word removed)
    test_data = {
        "id": range(10),
        "sentence": [
            "The quick brown jumps over the lazy dog .",  # missing 'fox'
            "Machine is fascinating and powerful .",  # missing 'learning'
            "Python is a programming language .",  # missing 'great'
            "Neural networks learn complex patterns .",  # missing 'can'
            "The weather today sunny and bright .",  # missing 'is'
        ]
        * 2,
    }
    pd.DataFrame(test_data).to_csv(Config.TEST_METADATA, index=False)

    # --------------------------------------------------------------------------
    # 3. Verify Vocabulary
    # --------------------------------------------------------------------------
    print("Verifying Vocabulary...")
    vocab = Vocabulary()
    # Build from the small train csv
    vocab.build(corpus_path=Config.TRAIN_METADATA, load_cached_data=False)

    # Assertions
    assert len(vocab) > 0, "Vocabulary should not be empty"
    assert Config.GAP_TOKEN in vocab.stoi, "GAP token missing from vocab"
    assert vocab.stoi[Config.PAD_TOKEN] == 0, "PAD token index must be 0"

    print(f"Vocabulary size: {len(vocab)}")

    # --------------------------------------------------------------------------
    # 4. Verify Dataset Logic
    # --------------------------------------------------------------------------
    print("Verifying InterleavedDataset logic...")
    # Force rebuild of cache by removing if exists (though dir is new)
    dataset = InterleavedDataset("train", vocab, load_cached_data=False)

    sample = dataset[0]
    input_ids = sample["input_ids"]
    target_loc = sample["target_loc"]
    target_word = sample["target_word"]

    print(f"Sample Input IDs: {input_ids.tolist()}")
    print(f"Target Loc: {target_loc.item()}, Target Word ID: {target_word.item()}")

    # Logic Checks
    # 1. Input should contain GAP tokens at odd indices (1, 3, 5...)
    # GAP token ID
    gap_id = vocab.stoi[Config.GAP_TOKEN]
    # Check a few odd indices
    if len(input_ids) > 1:
        assert input_ids[1] == gap_id, "Index 1 should be a GAP token"
    if len(input_ids) > 3:
        assert input_ids[3] == gap_id, "Index 3 should be a GAP token"

    # 2. Target location should be an odd index (pointing to a GAP)
    if target_loc.item() != -1:
        assert (
            target_loc.item() % 2 != 0
        ), "Target location must be an odd index (gap position)"

    print("Dataset logic verified.")

    # --------------------------------------------------------------------------
    # 5. Verify Model Architecture
    # --------------------------------------------------------------------------
    print("Verifying Model Architecture...")
    model = BifurcatedTransformer()

    # Create dummy batch
    batch_size = 2
    seq_len = 10
    dummy_input = torch.randint(0, Config.VOCAB_SIZE, (batch_size, seq_len))
    dummy_mask = torch.ones((batch_size, seq_len))

    loc_logits, id_logits = model(dummy_input, dummy_mask)

    # Shape Checks
    # loc_logits: (B, S, 1)
    assert loc_logits.shape == (
        batch_size,
        seq_len,
        1,
    ), f"Expected loc_logits shape {(batch_size, seq_len, 1)}, got {loc_logits.shape}"

    # id_logits: (B, S, VocabSize)
    assert id_logits.shape == (
        batch_size,
        seq_len,
        Config.VOCAB_SIZE,
    ), f"Expected id_logits shape {(batch_size, seq_len, Config.VOCAB_SIZE)}, got {id_logits.shape}"

    print("Model forward pass successful.")

    # --------------------------------------------------------------------------
    # 6. Execute Training Loop
    # --------------------------------------------------------------------------
    print("Running Trainer...")
    trainer = Trainer(debug=True)

    # Run 1 epoch
    trainer.train(epochs=1)

    # Verify model checkpoint exists
    assert os.path.exists(Config.MODEL_PATH), "Model checkpoint was not saved."
    print("Training complete and model saved.")

    # --------------------------------------------------------------------------
    # 7. Execute Inference
    # --------------------------------------------------------------------------
    print("Running Inference...")
    predictor = Predictor()
    predictor.predict()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Verify submission content format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "id" in df_sub.columns and "sentence" in df_sub.columns
    ), "Submission missing required columns"
    assert len(df_sub) == len(
        pd.read_csv(Config.TEST_METADATA)
    ), "Submission row count mismatch"

    print("Inference complete. Submission generated.")
    print("--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
