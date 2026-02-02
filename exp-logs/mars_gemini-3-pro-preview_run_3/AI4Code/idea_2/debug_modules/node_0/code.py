import os
import shutil
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import library components
from library.config import Config
from library.utils import set_seed, count_inversions, compute_kendall_tau
from library.feature_extractor import EmbeddingGenerator
from library.dataset import NotebookSequenceDataset
from library.model import DSAPR
from library.trainer import ModelTrainer
from library.inference import InferenceEngine


def run_demo():
    print("=== Starting AI4Code Solution Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Fast Demo
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Create a separate directory for demo outputs
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths to point to the demo directory
    # We leave INPUT_DIR and METADATA_DIR as is, since those are read-only/pre-generated
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "dsapr_model.pth")

    # Update cache paths to use the demo directory
    Config.TRAIN_CACHE_PATH = os.path.join(DEMO_DIR, "train_features.parquet")
    Config.VAL_CACHE_PATH = os.path.join(DEMO_DIR, "val_features.parquet")
    Config.TEST_CACHE_PATH = os.path.join(DEMO_DIR, "test_features.parquet")

    # Override Hyperparameters for speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.MAX_SEQ_LEN = 16  # Short context for speed
    Config.TRANSFORMER_LAYERS = 1
    Config.TRANSFORMER_HEADS = 2
    # EMBED_DIM must match the sentence-transformer model (384 for all-MiniLM-L6-v2)
    Config.EMBED_DIM = 384
    Config.NUM_WORKERS = 0  # Disable multiprocessing for demo stability

    # Set seed for reproducibility
    set_seed(42)
    print("Configuration updated.")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test count_inversions
    # Array: [2, 0, 1] -> Sorted target: [0, 1, 2]
    # Swaps needed: (2,0), (2,1) -> 2 swaps
    inv_count = count_inversions([2, 0, 1])
    assert inv_count == 2, f"Expected 2 inversions, got {inv_count}"

    # Test compute_kendall_tau
    # Perfect match
    score_perfect = compute_kendall_tau([["a", "b", "c"]], [["a", "b", "c"]])
    assert abs(score_perfect - 1.0) < 1e-6, "Perfect match should be 1.0"

    # Worst case: reverse order
    # n=3. Normalization = 3*(2) = 6.
    # Inversions for [c, b, a] vs [a, b, c] is 3.
    # K = 1 - 4 * (3/6) = -1.0
    score_worst = compute_kendall_tau([["c", "b", "a"]], [["a", "b", "c"]])
    assert (
        abs(score_worst - (-1.0)) < 1e-6
    ), f"Worst case should be -1.0, got {score_worst}"
    print("  - Utility functions verified.")

    # -------------------------------------------------------------------------
    # 3. Feature Extraction Demo
    # -------------------------------------------------------------------------
    print("\n[3] Running Feature Extraction (Subset)...")

    # Limit to 5 notebooks per split to save time
    DEBUG_LIMIT = 5

    generator = EmbeddingGenerator()

    # Generate features for Train
    print("  - Processing Train split...")
    df_train = generator.process_split(
        "train", load_cached_data=False, debug_limit=DEBUG_LIMIT
    )
    assert os.path.exists(Config.TRAIN_CACHE_PATH), "Train parquet not saved."
    assert not df_train.empty, "Train features DataFrame is empty."
    assert "embedding" in df_train.columns, "Embeddings missing from train features."

    # Generate features for Val
    print("  - Processing Val split...")
    df_val = generator.process_split(
        "val", load_cached_data=False, debug_limit=DEBUG_LIMIT
    )
    assert os.path.exists(Config.VAL_CACHE_PATH), "Val parquet not saved."

    # Generate features for Test
    print("  - Processing Test split...")
    df_test = generator.process_split(
        "test", load_cached_data=False, debug_limit=DEBUG_LIMIT
    )
    assert os.path.exists(Config.TEST_CACHE_PATH), "Test parquet not saved."

    print(f"  - Features generated for {DEBUG_LIMIT} notebooks per split.")

    # -------------------------------------------------------------------------
    # 4. Dataset Class Demo
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Dataset Class...")

    # This will process the parquet files into pytorch samples and cache them
    train_dataset = NotebookSequenceDataset(
        split="train", load_cached_data=False, debug_limit=DEBUG_LIMIT
    )
    assert len(train_dataset) > 0, "Dataset should not be empty."

    sample = train_dataset[0]

    # Verify structure
    required_keys = {"query", "context", "mask", "label", "ids"}
    assert required_keys.issubset(
        sample.keys()
    ), f"Missing keys in dataset sample: {sample.keys()}"

    # Verify tensor shapes
    # Query: (Embed_Dim)
    assert sample["query"].shape == (
        Config.EMBED_DIM,
    ), f"Query shape mismatch: {sample['query'].shape}"
    # Context: (Max_Seq_Len, Embed_Dim)
    assert sample["context"].shape == (
        Config.MAX_SEQ_LEN,
        Config.EMBED_DIM,
    ), f"Context shape mismatch: {sample['context'].shape}"
    # Mask: (Max_Seq_Len)
    assert sample["mask"].shape == (
        Config.MAX_SEQ_LEN,
    ), f"Mask shape mismatch: {sample['mask'].shape}"
    # Label: Scalar
    assert sample["label"].numel() == 1, "Label should be scalar."

    print("  - Dataset integrity checks passed.")

    # -------------------------------------------------------------------------
    # 5. Model Architecture Demo
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Model Architecture...")

    model = DSAPR().to(Config.DEVICE)
    model.eval()

    # Create dummy batch (Batch Size = 2)
    dummy_query = torch.randn(2, Config.EMBED_DIM).to(Config.DEVICE)
    dummy_context = torch.randn(2, Config.MAX_SEQ_LEN, Config.EMBED_DIM).to(
        Config.DEVICE
    )
    dummy_mask = torch.ones(2, Config.MAX_SEQ_LEN).to(Config.DEVICE)

    with torch.no_grad():
        output = model(dummy_query, dummy_context, dummy_mask)

    # Expected output shape: (Batch_Size,)
    assert output.shape == (
        2,
    ), f"Model output shape mismatch. Expected (2,), got {output.shape}"
    # Output should be in [0, 1] due to Sigmoid
    assert (
        output.min() >= 0 and output.max() <= 1
    ), "Model output should be sigmoid (0-1)."

    print("  - Model forward pass passed.")

    # -------------------------------------------------------------------------
    # 6. Training Loop Demo
    # -------------------------------------------------------------------------
    print("\n[6] Running Training Loop (1 Epoch)...")

    # Initialize trainer with debug limit
    trainer = ModelTrainer(debug_limit=DEBUG_LIMIT)

    # Run training
    trainer.train()

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    print("  - Training cycle completed.")

    # -------------------------------------------------------------------------
    # 7. Inference and Submission Demo
    # -------------------------------------------------------------------------
    print("\n[7] Running Inference and Generating Submission...")

    inference_engine = InferenceEngine(debug_limit=DEBUG_LIMIT)
    inference_engine.generate_submission()

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    # Validate submission format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert list(sub_df.columns) == ["id", "cell_order"], "Submission columns mismatch."
    assert len(sub_df) > 0, "Submission file is empty."

    # Check content format
    sample_order = sub_df.iloc[0]["cell_order"]
    assert (
        isinstance(sample_order, str) and len(sample_order.split()) > 0
    ), "Invalid cell_order format."

    print("  - Submission generated successfully.")
    print(f"  - Output location: {Config.SUBMISSION_PATH}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
