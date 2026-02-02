import os
import shutil
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.symbolic_stats import StatsBuilder, HierarchicalLookup
from library.neural_data import (
    prepare_neural_data,
    NormalizationDataset,
    collate_fn,
    CharTokenizer,
)
from library.transformer import Seq2SeqTransformer
from library.training import Trainer
from library.inference import CascadePredictor


def setup_demo_environment():
    """
    Sets up a temporary environment for the demo by:
    1. Creating a working directory.
    2. Subsetting the real data to create small demo datasets.
    3. Patching the Config class to point to these demo resources.
    """
    print("--- Setting up Demo Environment ---")

    # define paths
    demo_dir = "./working/demo_env"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Load a small subset of the real data
    # We must preserve sentence structure for context-based logic (prev/next tokens)
    print("Creating data subsets...")

    # Helper to subset by sentence_id
    def create_subset(source_path, dest_path, n_sentences=200):
        df = pd.read_parquet(source_path)
        # Get first n unique sentences
        sent_ids = df["sentence_id"].unique()[:n_sentences]
        subset = df[df["sentence_id"].isin(sent_ids)].copy()
        subset.to_parquet(dest_path, index=False)
        return len(subset)

    # Create subsets
    train_path = os.path.join(demo_dir, "train.parquet")
    val_path = os.path.join(demo_dir, "val.parquet")
    test_path = os.path.join(demo_dir, "test.parquet")

    n_train = create_subset(Config.TRAIN_META, train_path, n_sentences=500)
    n_val = create_subset(Config.VAL_META, val_path, n_sentences=50)
    n_test = create_subset(Config.TEST_META, test_path, n_sentences=50)

    print(f"Created subsets: Train={n_train}, Val={n_val}, Test={n_test} rows")

    # Patch Config to use demo paths
    print("Patching Config...")
    Config.IDEA_DIR = demo_dir
    Config.TRAIN_META = train_path
    Config.VAL_META = val_path
    Config.TEST_META = test_path

    # Update derived paths
    Config.STATS_TRIGRAM = os.path.join(demo_dir, "stats_trigram.parquet")
    Config.STATS_BIGRAM_LEFT = os.path.join(demo_dir, "stats_bigram_left.parquet")
    Config.STATS_BIGRAM_RIGHT = os.path.join(demo_dir, "stats_bigram_right.parquet")
    Config.STATS_UNIGRAM = os.path.join(demo_dir, "stats_unigram.parquet")

    Config.PROCESSED_TRAIN = os.path.join(demo_dir, "train_processed.parquet")
    Config.PROCESSED_VAL = os.path.join(demo_dir, "val_processed.parquet")
    Config.PROCESSED_TEST = os.path.join(demo_dir, "test_processed.parquet")

    Config.TOKENIZER_PATH = os.path.join(demo_dir, "tokenizer.json")
    Config.MODEL_CHECKPOINT = os.path.join(demo_dir, "model_demo.pt")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission_demo.csv")

    # Adjust Hyperparameters for Speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 32
    Config.DEBUG = (
        False  # We already manually subsetted, so we don't need the internal debug flag
    )

    # Set seed
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    print("Environment setup complete.\n")


def demo_symbolic_stats():
    """
    Demonstrates generating symbolic statistics (Trigrams, Bigrams, etc.).
    """
    print("--- Demo: Symbolic Stats Builder ---")

    builder = StatsBuilder()

    # Force computation (ignore cache if any)
    builder.run(load_cached_data=False)

    # Verification
    assert os.path.exists(Config.STATS_TRIGRAM), "Trigram stats file not created"
    assert os.path.exists(Config.STATS_UNIGRAM), "Unigram stats file not created"

    # Check content
    df_uni = pd.read_parquet(Config.STATS_UNIGRAM)
    print(f"Generated {len(df_uni)} unigram stats.")
    assert len(df_uni) > 0, "Unigram stats empty"

    print("Symbolic Stats verified.\n")


def demo_neural_prep():
    """
    Demonstrates tokenizer building and data preprocessing.
    """
    print("--- Demo: Neural Data Preparation ---")

    # Run preparation
    tokenizer = prepare_neural_data(load_cached_data=False)

    # Verification
    assert os.path.exists(Config.TOKENIZER_PATH), "Tokenizer file not created"
    assert len(tokenizer) > 0, "Tokenizer vocabulary is empty"
    print(f"Tokenizer Vocab Size: {len(tokenizer)}")

    assert os.path.exists(Config.PROCESSED_TRAIN), "Processed train data missing"
    assert os.path.exists(Config.PROCESSED_TEST), "Processed test data missing"

    # Verify processed data schema
    df_proc = pd.read_parquet(Config.PROCESSED_TRAIN)
    required_cols = ["prev", "before", "next", "after", "class", "id"]
    for col in required_cols:
        assert col in df_proc.columns, f"Missing column {col} in processed data"

    print("Neural Data Prep verified.\n")
    return tokenizer


def demo_training(tokenizer):
    """
    Demonstrates model initialization and a single training epoch.
    """
    print("--- Demo: Model Training ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Dataset
    train_dataset = NormalizationDataset(
        Config.PROCESSED_TRAIN, tokenizer, split="train", max_len=Config.MAX_SEQ_LEN
    )
    val_dataset = NormalizationDataset(
        Config.PROCESSED_VAL, tokenizer, split="val", max_len=Config.MAX_SEQ_LEN
    )

    # Check if dataset is empty (might happen if filter removes all 'easy' tokens in the small subset)
    if len(train_dataset) == 0:
        print(
            "Warning: Training subset is empty after filtering (all tokens were PLAIN/PUNCT). Skipping training demo."
        )
        return

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn
    )

    # Verify Batch
    batch = next(iter(train_loader))
    assert "src" in batch
    assert "tgt_in" in batch
    assert "tgt_out" in batch
    print(f"Batch shapes - Src: {batch['src'].shape}, Tgt: {batch['tgt_out'].shape}")

    # Init Model
    model = Seq2SeqTransformer(
        num_tokens=len(tokenizer),
        d_model=64,  # Reduced for demo speed
        nhead=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
    ).to(device)

    # Init Trainer
    trainer = Trainer(model, tokenizer, device)

    # Run 1 Epoch
    print("Running 1 epoch...")
    loss, acc = trainer.train_epoch(train_loader, epoch_idx=1)

    assert not np.isnan(loss), "Training loss is NaN"
    print(f"Epoch complete. Loss: {loss:.4f}, Acc: {acc:.4f}")

    # Save dummy checkpoint for inference demo
    torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)
    print("Model Training verified.\n")


def demo_inference():
    """
    Demonstrates the cascade prediction pipeline.
    """
    print("--- Demo: Inference ---")

    # Instantiate Predictor
    # It will load the symbolic stats and the model checkpoint we just saved
    predictor = CascadePredictor()

    # Load test data
    df_test = pd.read_parquet(Config.PROCESSED_TEST)
    print(f"Predicting on {len(df_test)} test samples...")

    # Run prediction
    results = predictor.predict(df_test)

    # Verification
    assert isinstance(results, dict), "Prediction results should be a dictionary"
    assert len(results) > 0, "No predictions returned"

    # Check a few random IDs
    sample_id = df_test.iloc[0]["id"]
    assert sample_id in results, f"ID {sample_id} missing from results"

    print(f"Sample Prediction: ID={sample_id}, Pred='{results[sample_id]}'")

    # Generate submission file
    predictor.generate_submission(load_cached_data=True)
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    print("Inference verified.\n")


if __name__ == "__main__":
    # 1. Setup Environment
    setup_demo_environment()

    # 2. Run Symbolic Stats
    demo_symbolic_stats()

    # 3. Run Neural Prep
    tokenizer = demo_neural_prep()

    # 4. Run Training
    demo_training(tokenizer)

    # 5. Run Inference
    demo_inference()

    print("=== All Demos Completed Successfully ===")
