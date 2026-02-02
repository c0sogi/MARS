import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import PathConfig, TrainConfig, ModelConfig, TARGET_COLS
from library.utils import seed_everything, compute_spearmanr
from library.dataset import load_data, QuestDataset
from library.model import SegmentAwareCrossEncoder
from library.engine import run_backbone_training
from library.feature_pipeline import run_feature_extraction
from library.ridge_head import train_ridge_and_predict


def main():
    print("Starting demonstration of library components...")

    # -------------------------------------------------------------------------
    # 0. Setup and Configuration Overrides
    # -------------------------------------------------------------------------
    # Ensure reproducibility
    seed_everything(42)

    # Ensure working directory exists
    os.makedirs(PathConfig.WORKING_DIR, exist_ok=True)

    # OVERRIDE CONFIGURATION FOR SPEED
    # We monkey-patch the configuration class to run a minimal version of the task
    print("Overriding TrainConfig for fast demonstration...")
    TrainConfig.epochs = 1
    TrainConfig.batch_size = 4
    TrainConfig.grad_acc_steps = 1
    TrainConfig.num_workers = 0  # Avoid multiprocessing overhead for tiny data
    TrainConfig.ridge_alphas = (1.0,)  # Single alpha for fast RidgeCV

    # -------------------------------------------------------------------------
    # 1. Prepare Subset Data (Optimization)
    # -------------------------------------------------------------------------
    print("\n[Step 1] Creating small data subsets for speed...")

    # Load raw metadata
    train_meta = pd.read_csv(PathConfig.TRAIN_META)
    val_meta = pd.read_csv(PathConfig.VAL_META)
    test_meta = pd.read_csv(PathConfig.TEST_META)

    # Create tiny subsets
    train_subset = train_meta.head(20).copy()
    val_subset = val_meta.head(10).copy()
    test_subset = test_meta.head(10).copy()

    # Pre-process text columns to match library expectations (fillna)
    text_cols = ["question_title", "question_body", "answer"]
    for df in [train_subset, val_subset, test_subset]:
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

    # Save as parquet to the cache location defined in library.dataset.load_data
    # This tricks the library into loading our small subsets instead of the full data
    train_cache = os.path.join(PathConfig.WORKING_DIR, "train_processed.parquet")
    val_cache = os.path.join(PathConfig.WORKING_DIR, "val_processed.parquet")
    test_cache = os.path.join(PathConfig.WORKING_DIR, "test_processed.parquet")

    train_subset.to_parquet(train_cache, index=False)
    val_subset.to_parquet(val_cache, index=False)
    test_subset.to_parquet(test_cache, index=False)

    print(f"Cached subset data to {PathConfig.WORKING_DIR}")

    # Verify load_data picks up the cache
    df_train, df_val, df_test = load_data(load_cached_data=True)
    assert len(df_train) == 20, "Train data should be subset size 20"
    assert len(df_val) == 10, "Val data should be subset size 10"
    assert len(df_test) == 10, "Test data should be subset size 10"
    print("Data loading verification successful.")

    # -------------------------------------------------------------------------
    # 2. Verify Dataset Class
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying QuestDataset...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.model_name)

    ds = QuestDataset(df_train, tokenizer, max_len=128, mode="train")
    item = ds[0]

    # Check keys
    expected_keys = {"input_ids", "attention_mask", "q_mask", "a_mask", "labels"}
    assert (
        set(item.keys()) == expected_keys
    ), f"Dataset item keys mismatch. Found: {item.keys()}"

    # Check shapes
    assert item["input_ids"].shape == (128,), "Incorrect input_ids shape"
    assert item["labels"].shape == (30,), "Incorrect labels shape"
    assert item["q_mask"].sum() > 0, "q_mask should not be empty"
    print("QuestDataset verification successful.")

    # -------------------------------------------------------------------------
    # 3. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying SegmentAwareCrossEncoder...")
    device = torch.device("cpu")  # Use CPU for simple logic check
    model = SegmentAwareCrossEncoder()
    model.to(device)
    model.eval()

    # Create a dummy batch
    batch_input_ids = item["input_ids"].unsqueeze(0).to(device)
    batch_att_mask = item["attention_mask"].unsqueeze(0).to(device)
    batch_q_mask = item["q_mask"].unsqueeze(0).to(device)
    batch_a_mask = item["a_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits, features = model(
            batch_input_ids, batch_att_mask, batch_q_mask, batch_a_mask
        )

    # Check Output Shapes
    # Logits: [batch_size, num_labels] -> [1, 30]
    assert logits.shape == (1, 30), f"Logits shape mismatch: {logits.shape}"
    # Features: [batch_size, hidden * 4] -> [1, 768 * 4] -> [1, 3072]
    expected_feat_dim = 768 * 4
    assert features.shape == (
        1,
        expected_feat_dim,
    ), f"Features shape mismatch: {features.shape}"
    print("Model forward pass verification successful.")

    # -------------------------------------------------------------------------
    # 4. Run Backbone Training
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running Backbone Training (Fine-tuning)...")

    # We need to recreate dataloaders inside the engine, but since we cached the data,
    # the engine will pick up the small subsets.

    # We need to manually construct loaders here to pass to run_backbone_training
    # because run_backbone_training expects loaders as input?
    # Checking engine.py: run_backbone_training(train_loader, val_loader)

    from torch.utils.data import DataLoader

    train_dataset = QuestDataset(df_train, tokenizer, max_len=128, mode="train")
    val_dataset = QuestDataset(df_val, tokenizer, max_len=128, mode="train")

    train_loader = DataLoader(
        train_dataset, batch_size=TrainConfig.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=TrainConfig.batch_size, shuffle=False
    )

    # Run training
    run_backbone_training(train_loader, val_loader)

    # Verify model was saved
    assert os.path.exists(PathConfig.MODEL_SAVE_PATH), "Model file was not saved."
    print("Backbone training verification successful.")

    # -------------------------------------------------------------------------
    # 5. Run Feature Extraction
    # -------------------------------------------------------------------------
    print("\n[Step 5] Running Feature Extraction...")

    # Force reload from disk to ensure pipeline works end-to-end
    # We set load_cached_data=False inside the call logic for extraction?
    # No, run_feature_extraction(load_cached_data=True) checks for *features*.
    # Since features don't exist yet, it will compute them.

    # Note: run_feature_extraction calls load_data internally.
    # It will use our cached parquet files.

    train_feats, train_targs, val_feats, val_targs, test_feats = run_feature_extraction(
        load_cached_data=True
    )

    # Verify shapes
    assert train_feats.shape == (
        20,
        expected_feat_dim,
    ), f"Train features shape mismatch: {train_feats.shape}"
    assert train_targs.shape == (
        20,
        30,
    ), f"Train targets shape mismatch: {train_targs.shape}"
    assert test_feats.shape == (
        10,
        expected_feat_dim,
    ), f"Test features shape mismatch: {test_feats.shape}"

    # Verify feature cache files exist
    assert os.path.exists(
        PathConfig.TRAIN_FEATURES_CACHE
    ), "Train features cache missing"
    assert os.path.exists(PathConfig.TEST_FEATURES_CACHE), "Test features cache missing"
    print("Feature extraction verification successful.")

    # -------------------------------------------------------------------------
    # 6. Run Ridge Head Training & Prediction
    # -------------------------------------------------------------------------
    print("\n[Step 6] Running Ridge Regression Head...")

    # This will load features from cache, train Ridge, and generate submission
    train_ridge_and_predict(load_cached_model=False)

    # Verify submission file
    assert os.path.exists(PathConfig.SUBMISSION_FILE), "Submission file missing"

    sub_df = pd.read_csv(PathConfig.SUBMISSION_FILE)
    print(f"Submission shape: {sub_df.shape}")

    # Verify submission content
    assert sub_df.shape == (
        10,
        31,
    ), f"Submission shape mismatch. Expected (10, 31), got {sub_df.shape}"
    assert "qa_id" in sub_df.columns, "qa_id column missing in submission"

    # Check value range
    numeric_cols = [c for c in sub_df.columns if c != "qa_id"]
    min_val = sub_df[numeric_cols].min().min()
    max_val = sub_df[numeric_cols].max().max()

    assert min_val >= 0.0, "Predictions contain negative values"
    assert max_val <= 1.0, "Predictions contain values > 1.0"

    print("Ridge head and submission verification successful.")

    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    main()
