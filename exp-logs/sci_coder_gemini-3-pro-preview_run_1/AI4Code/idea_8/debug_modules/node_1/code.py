import os
import sys
import shutil
import pandas as pd
import torch
import numpy as np
import logging
from torch.utils.data import DataLoader

# Suppress warnings and logs for cleaner output
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)

# Import library modules
from library.config import Config
from library.utils import set_seed, compute_kendall_tau
from library.preprocess import FeatureExtractor
from library.dataset import CachedDataset, collate_fn
from library.model import DCAN
from library.engine import Engine


def run_demo():
    print("=== Starting AI4Code Solution Demo ===")

    # 1. Setup
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Create Subset Metadata
    # We use a very small subset of the actual metadata to ensure the demo runs quickly.
    print("\n[Step 1] Creating subset metadata for demonstration...")

    # Load original metadata
    orig_train_meta = pd.read_csv("./metadata/train_metadata.csv")
    orig_val_meta = pd.read_csv("./metadata/val_metadata.csv")
    orig_test_meta = pd.read_csv("./metadata/test_metadata.csv")

    # Sample 10 notebooks for train, 5 for val, 5 for test
    subset_train = orig_train_meta.head(10).copy()
    subset_val = orig_val_meta.head(5).copy()
    subset_test = orig_test_meta.head(5).copy()

    # Define paths for demo metadata
    demo_train_meta_path = os.path.join(DEMO_DIR, "train_metadata.csv")
    demo_val_meta_path = os.path.join(DEMO_DIR, "val_metadata.csv")
    demo_test_meta_path = os.path.join(DEMO_DIR, "test_metadata.csv")

    subset_train.to_csv(demo_train_meta_path, index=False)
    subset_val.to_csv(demo_val_meta_path, index=False)
    subset_test.to_csv(demo_test_meta_path, index=False)

    print(f"Created subset metadata in {DEMO_DIR}")

    # 3. Override Config
    # We monkey-patch the Config class attributes to point to our demo files.
    # This ensures that when library classes instantiate Config(), they get our paths.
    print("\n[Step 2] Configuring paths and hyperparameters...")
    Config.TRAIN_METADATA_PATH = demo_train_meta_path
    Config.VAL_METADATA_PATH = demo_val_meta_path
    Config.TEST_METADATA_PATH = demo_test_meta_path

    Config.TRAIN_FEATS_PATH = os.path.join(DEMO_DIR, "train_features.parquet")
    Config.VAL_FEATS_PATH = os.path.join(DEMO_DIR, "val_features.parquet")
    Config.TEST_FEATS_PATH = os.path.join(DEMO_DIR, "test_features.parquet")

    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Reduce batch size and epochs for demo speed
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 2
    Config.PATIENCE = 1

    # 4. Preprocessing
    print("\n[Step 3] Running Feature Extractor (Preprocessing)...")
    extractor = FeatureExtractor()

    # Process Train
    df_train = extractor.process_dataset(
        Config.TRAIN_METADATA_PATH, Config.TRAIN_FEATS_PATH, load_cached_data=False
    )
    assert os.path.exists(Config.TRAIN_FEATS_PATH), "Train features parquet not created"
    assert len(df_train) == 10, f"Expected 10 train samples, got {len(df_train)}"

    # Process Val
    df_val = extractor.process_dataset(
        Config.VAL_METADATA_PATH, Config.VAL_FEATS_PATH, load_cached_data=False
    )

    # Process Test
    df_test = extractor.process_dataset(
        Config.TEST_METADATA_PATH, Config.TEST_FEATS_PATH, load_cached_data=False
    )
    print("Preprocessing completed successfully.")

    # 5. Dataset & DataLoader Verification
    print("\n[Step 4] Verifying Dataset and DataLoader...")
    train_ds = CachedDataset(mode="train", load_cached_data=True)

    # Check single item
    item = train_ds[0]
    print(f"Sample ID: {item['id']}")
    print(f"Code Embeddings Shape: {item['code_embeddings'].shape}")
    print(f"Markdown Embeddings Shape: {item['markdown_embeddings'].shape}")
    print(f"Labels Shape: {item['labels'].shape}")

    # Assertions for single item
    assert item["code_embeddings"].ndim == 2
    assert item["code_embeddings"].shape[1] == 768  # MPNet dim
    assert item["markdown_embeddings"].shape[0] == item["labels"].shape[0]

    # Check Batch Collation
    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn, shuffle=True
    )

    batch = next(iter(train_loader))
    print(f"Batch Keys: {batch.keys()}")
    print(f"Batch Code Shape: {batch['code_embeddings'].shape}")

    # Assertions for batch
    # Shape: (Batch, Max_Seq_Len, Dim)
    assert batch["code_embeddings"].dim() == 3
    assert batch["markdown_embeddings"].dim() == 3
    assert batch["labels"].dim() == 2
    # Check masking/padding logic: labels should be -100 in padded areas
    # We can't strictly check -100 without knowing lengths, but we can check lengths match tensor sizes
    max_md_len_in_batch = batch["markdown_embeddings"].size(1)
    assert batch["labels"].size(1) == max_md_len_in_batch

    print("Dataset and DataLoader verified.")

    # 6. Model Initialization & Forward Pass
    print("\n[Step 5] Initializing Model (DCAN)...")
    model = DCAN().to(device)

    # Move batch to device
    code_emb = batch["code_embeddings"].to(device)
    md_emb = batch["markdown_embeddings"].to(device)
    code_lens = batch["code_lens"].to(device)
    md_lens = batch["md_lens"].to(device)

    # Forward pass
    logits = model(code_emb, md_emb, code_lens, md_lens)
    print(f"Logits Shape: {logits.shape}")

    # Assertions
    # Output shape: (Batch, Max_MD, Max_Code + 1)
    # Max_Code + 1 accounts for the EOS token position
    expected_last_dim = batch["code_embeddings"].size(1) + 1
    assert logits.shape[0] == Config.BATCH_SIZE
    assert logits.shape[1] == batch["markdown_embeddings"].size(1)
    assert logits.shape[2] == expected_last_dim
    print("Model forward pass successful.")

    # 7. Training Loop
    print("\n[Step 6] Running Training Loop (Engine)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    engine = Engine(model, device, optimizer)

    val_loader = DataLoader(
        CachedDataset(mode="val", load_cached_data=True),
        batch_size=Config.BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=False,
    )

    # Train for defined epochs
    engine.fit(
        train_loader,
        val_loader,
        df_train,
        df_val,
        epochs=Config.NUM_EPOCHS,
        patience=Config.PATIENCE,
    )

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    print("Training loop completed.")

    # 8. Inference
    print("\n[Step 7] Running Inference...")
    # Load test dataset
    test_ds = CachedDataset(mode="test", load_cached_data=True)
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn, shuffle=False
    )

    # Load best model weights
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Generate submission
    df_submission = engine.predict(test_loader, raw_df=test_ds.df)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."
    assert len(df_submission) == len(df_test), "Submission length mismatch."
    assert "id" in df_submission.columns and "cell_order" in df_submission.columns
    print(f"Inference successful. Submission head:\n{df_submission.head(2)}")

    # 9. Metric Verification
    print("\n[Step 8] Verifying Metric (Kendall Tau)...")
    # Case 1: Perfect match
    df_gt = pd.DataFrame({"id": ["nb1"], "cell_order": ["a b c d"]})
    df_pred_perfect = pd.DataFrame({"id": ["nb1"], "cell_order": ["a b c d"]})
    score_perfect = compute_kendall_tau(df_pred_perfect, df_gt)
    print(f"Perfect Match Score: {score_perfect}")
    assert np.isclose(score_perfect, 1.0), "Metric failed on perfect match"

    # Case 2: Complete reversal
    # n=4. Pairs = 4*3/2 = 6.
    # Reversal requires 6 swaps.
    # K = 1 - 4 * (6 / 6) = 1 - 4 = -3 ???
    # Wait, the formula in the task description is: K = 1 - 4 * (S / (n*(n-1)))
    # For n=4, n*(n-1) = 12.
    # Worst case swaps S = n(n-1)/2 = 6.
    # K = 1 - 4 * (6 / 12) = 1 - 2 = -1. Correct.
    df_pred_reverse = pd.DataFrame({"id": ["nb1"], "cell_order": ["d c b a"]})
    score_reverse = compute_kendall_tau(df_pred_reverse, df_gt)
    print(f"Reverse Match Score: {score_reverse}")
    assert np.isclose(score_reverse, -1.0), "Metric failed on reverse match"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
