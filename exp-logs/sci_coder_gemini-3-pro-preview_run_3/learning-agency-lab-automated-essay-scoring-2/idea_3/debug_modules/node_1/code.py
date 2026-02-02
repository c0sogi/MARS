import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Ensure the library modules can be imported
sys.path.append("./library")

from library.config import Config
from library.utils import seed_everything, compute_qwk, get_llrd_optimizer_params
from library.data import get_dataloaders, get_test_loader, process_data
from library.model import EssayModel
from library.engine import run_training
from library.stacking import extract_features, train_lgbm, predict_stacking


def create_subset_metadata(source_path, dest_path, n_samples=50):
    """Creates a small subset of the metadata csv for demonstration speed."""
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    df = pd.read_csv(source_path)
    # Sample subset
    subset = df.head(n_samples).copy()
    subset.to_csv(dest_path, index=False)
    print(f"Created subset {dest_path} with {len(subset)} samples.")
    return subset


def main():
    print("=== Starting Essay Scoring Demo ===")

    # 1. Setup & Configuration Overrides
    seed_everything(Config.seed)

    # Override Config to use a temporary working directory for this demo
    # to avoid conflicts with existing caches.
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Create small subsets of data for speed
    subset_train_path = os.path.join(Config.WORKING_DIR, "train_subset.csv")
    subset_val_path = os.path.join(Config.WORKING_DIR, "val_subset.csv")
    subset_test_path = os.path.join(Config.WORKING_DIR, "test_subset.csv")

    create_subset_metadata(Config.TRAIN_META, subset_train_path, n_samples=32)
    create_subset_metadata(Config.VAL_META, subset_val_path, n_samples=16)
    create_subset_metadata(Config.TEST_META, subset_test_path, n_samples=16)

    # Patch Config paths to point to subsets
    Config.TRAIN_META = subset_train_path
    Config.VAL_META = subset_val_path
    Config.TEST_META = subset_test_path

    # Reduce training parameters for speed
    Config.epochs = 1
    Config.train_batch_size = 4
    Config.valid_batch_size = 4
    Config.gradient_accumulation_steps = 1

    print("\n=== Data Loading & Processing ===")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Force reload to ensure we use the subsets (load_cached_data=False)
    train_loader, val_loader = get_dataloaders(tokenizer, load_cached_data=False)

    # Verify DataLoader
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")

    # Check shapes
    # Input IDs: [batch_size, max_length]
    assert batch["input_ids"].shape == (
        Config.train_batch_size,
        Config.max_length,
    ), f"Incorrect input_ids shape: {batch['input_ids'].shape}"
    # Meta features: [batch_size, 8] (8 features defined in data.py)
    assert batch["meta_features"].shape == (
        Config.train_batch_size,
        8,
    ), f"Incorrect meta_features shape: {batch['meta_features'].shape}"
    # Labels: [batch_size]
    assert batch["labels"].shape == (
        Config.train_batch_size,
    ), f"Incorrect labels shape: {batch['labels'].shape}"

    print("Data loading verification passed.")

    print("\n=== Model Initialization & Forward Pass ===")
    # Initialize model
    model = EssayModel(pretrained=True)
    model.to(Config.device)

    # Run dummy forward pass
    input_ids = batch["input_ids"].to(Config.device)
    attention_mask = batch["attention_mask"].to(Config.device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    # Verify outputs
    # Logits: [batch_size, 1]
    assert outputs["logits"].shape == (
        Config.train_batch_size,
        1,
    ), f"Incorrect logits shape: {outputs['logits'].shape}"
    # Embeddings: [batch_size, hidden_size] (DeBERTa-large hidden size is 1024)
    assert outputs["embeddings"].shape == (
        Config.train_batch_size,
        1024,
    ), f"Incorrect embeddings shape: {outputs['embeddings'].shape}"

    print("Model forward pass verification passed.")

    print("\n=== Stage 1: Backbone Fine-tuning ===")
    # Setup Optimizer with LLRD
    optimizer_grouped_parameters = get_llrd_optimizer_params(
        model,
        encoder_lr=Config.backbone_lr,
        head_lr=Config.head_lr,
        weight_decay=Config.weight_decay,
        llrd_decay=Config.llrd_decay,
    )

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    # Setup Scheduler
    num_training_steps = len(train_loader) * Config.epochs
    num_warmup_steps = int(num_training_steps * Config.num_warmup_steps_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Run Training
    best_qwk = run_training(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        Config.device,
        num_epochs=Config.epochs,
        patience=1,
    )

    print(f"Stage 1 Training Complete. Best QWK: {best_qwk:.4f}")

    print("\n=== Stage 2: Feature Extraction & Stacking ===")
    # Load best model weights (simulated by using current model since we ran 1 epoch)
    # In a real run, we would reload: model.load_state_dict(torch.load(path))

    # Load DataFrames for feature extraction
    # We use process_data to get the dataframe with meta-features computed
    train_df = process_data(Config.TRAIN_META, "train_subset", load_cached_data=True)
    val_df = process_data(Config.VAL_META, "val_subset", load_cached_data=True)

    # Extract Features (Embeddings + Meta)
    # We disable cache loading here to force execution of the extraction logic
    train_feats, train_labels, _ = extract_features(
        train_df,
        model,
        tokenizer,
        Config.device,
        "train_subset",
        load_cached_data=False,
    )
    val_feats, val_labels, _ = extract_features(
        val_df, model, tokenizer, Config.device, "val_subset", load_cached_data=False
    )

    # Verify Feature Shapes
    # Shape should be [n_samples, 1024 + 8]
    expected_dim = 1024 + 8
    assert (
        train_feats.shape[1] == expected_dim
    ), f"Incorrect feature dimension: {train_feats.shape[1]}, expected {expected_dim}"

    print(f"Extracted Train Features: {train_feats.shape}")
    print(f"Extracted Val Features: {val_feats.shape}")

    # Train LightGBM
    lgbm_model = train_lgbm(train_feats, train_labels, val_feats, val_labels)

    print("\n=== Inference on Test Set ===")
    # Load Test Data
    test_df = process_data(Config.TEST_META, "test_subset", load_cached_data=False)

    # Extract Test Features
    test_feats, _, test_ids = extract_features(
        test_df, model, tokenizer, Config.device, "test_subset", load_cached_data=False
    )

    # Predict
    predict_stacking(lgbm_model, test_feats, test_ids)

    # Verify Submission File
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(sub_path), "Submission file was not created."

    sub_df = pd.read_csv(sub_path)
    print(f"Submission file loaded. Shape: {sub_df.shape}")
    print(sub_df.head())

    # Verify QWK Metric Function
    print("\n=== Verifying Metric Calculation ===")
    y_true = np.array([1, 2, 3, 4, 5, 6])
    y_pred = np.array([1.1, 2.0, 2.9, 4.2, 4.8, 6.0])  # Slight noise
    score = compute_qwk(y_true, y_pred)
    print(f"Test QWK Score (should be close to 1.0): {score:.4f}")
    assert score > 0.9, "QWK calculation seems incorrect for near-perfect predictions."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
