import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import warnings
import shutil

# Import from provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import (
    load_data,
    create_dataloader,
    load_mlm_corpus,
    create_mlm_dataloader,
)
from library.features import get_tfidf_features
from library.modeling import CustomTransformer, StatisticalModel
from library.engine import train_fn, eval_fn, inference_fn, train_mlm
from library.optimization import optimize_weights, apply_ensemble

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_demo():
    print("--- Starting Library Usage Demonstration ---")

    # 1. Setup and Configuration Overrides
    print("\n[1] Setup and Configuration")
    seed_everything(Config.SEED)

    # Override Config for speed
    Config.EPOCHS = 1
    Config.MLM_EPOCHS = 1
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Very small subset for demo
    Config.BATCH_SIZE = 4
    Config.MLM_BATCH_SIZE = 4

    # Ensure working directory is clean for this run if needed,
    # but we will just use the existing structure.
    print(f"Device: {Config.DEVICE}")
    print(f"Debug Mode: {Config.DEBUG}")

    # 2. Data Loading
    print("\n[2] Data Loading (dataset.py)")
    # Load full data
    train_df, val_df, test_df = load_data(load_cached_data=True)

    # Slice for speed
    train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE).copy()
    val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE).copy()
    test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE).copy()

    print(f"Sliced Train shape: {train_df.shape}")
    print(f"Sliced Val shape: {val_df.shape}")

    assert len(train_df) == Config.DEBUG_SAMPLE_SIZE
    assert "label" in train_df.columns
    assert "text" in train_df.columns

    # 3. Tokenizer and DataLoader
    print("\n[3] Tokenizer and DataLoader")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_DEBERTA)

    train_loader = create_dataloader(
        train_df, tokenizer, batch_size=Config.BATCH_SIZE, shuffle=True
    )
    val_loader = create_dataloader(
        val_df, tokenizer, batch_size=Config.BATCH_SIZE, shuffle=False
    )
    test_loader = create_dataloader(
        test_df, tokenizer, batch_size=Config.BATCH_SIZE, is_test=True, shuffle=False
    )

    # Verify Batch Structure
    batch = next(iter(train_loader))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "label" in batch
    assert batch["input_ids"].shape[0] == Config.BATCH_SIZE
    print("DataLoader batch structure verified.")

    # 4. MLM Data Loading
    print("\n[4] MLM Data Loading")
    # Load corpus (subset for speed)
    mlm_texts = load_mlm_corpus(load_cached_data=True)
    mlm_texts = mlm_texts[: Config.DEBUG_SAMPLE_SIZE]

    mlm_loader = create_mlm_dataloader(
        mlm_texts, tokenizer, batch_size=Config.MLM_BATCH_SIZE
    )
    mlm_batch = next(iter(mlm_loader))

    # In MLM, 'labels' are created by the collator
    assert "input_ids" in mlm_batch
    assert "labels" in mlm_batch
    print("MLM DataLoader batch structure verified.")

    # 5. TF-IDF Features
    print("\n[5] TF-IDF Features (features.py)")
    # Note: get_tfidf_features computes on the full dataset found in metadata.
    # We will load it, but for the demo of the StatisticalModel, we will slice the resulting matrices.
    X_train_full, y_train_full, X_val_full, y_val_full, X_test_full = (
        get_tfidf_features(load_cached_data=True)
    )

    # Slice sparse matrices for the demo
    X_train_demo = X_train_full[: Config.DEBUG_SAMPLE_SIZE]
    y_train_demo = y_train_full[: Config.DEBUG_SAMPLE_SIZE]
    X_val_demo = X_val_full[: Config.DEBUG_SAMPLE_SIZE]

    print(f"TF-IDF Train shape (sliced): {X_train_demo.shape}")
    assert X_train_demo.shape[0] == Config.DEBUG_SAMPLE_SIZE
    assert X_train_demo.shape[1] == X_val_demo.shape[1]

    # 6. Statistical Model
    print("\n[6] Statistical Model (modeling.py)")
    stat_model = StatisticalModel(weight_lr=0.6)
    stat_model.fit(X_train_demo, y_train_demo)

    stat_preds = stat_model.predict(X_val_demo)
    stat_probs = stat_model.predict_proba(X_val_demo)

    assert len(stat_preds) == Config.DEBUG_SAMPLE_SIZE
    assert stat_probs.shape == (Config.DEBUG_SAMPLE_SIZE, Config.NUM_CLASSES)
    # Check probabilities sum to 1
    assert np.allclose(stat_probs.sum(axis=1), 1.0)
    print("Statistical Model fit and prediction successful.")

    # 7. Neural Network Model
    print("\n[7] Neural Network Model (modeling.py)")
    # Initialize model
    model = CustomTransformer(
        model_name=Config.MODEL_DEBERTA,
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
        dropout_p=0.1,
    )
    model.to(Config.DEVICE)

    # Verify Forward Pass
    dummy_input = batch["input_ids"].to(Config.DEVICE)
    dummy_mask = batch["attention_mask"].to(Config.DEVICE)
    with torch.no_grad():
        logits = model(dummy_input, dummy_mask)

    assert logits.shape == (Config.BATCH_SIZE, Config.NUM_CLASSES)
    print("CustomTransformer forward pass verified.")

    # 8. Training Engine & AWP
    print("\n[8] Training Engine & AWP (engine.py, awp.py)")
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)
    )

    # Run one epoch of training with AWP
    # Note: AWP starts at Config.AWP_START_EPOCH. We set epoch=Config.AWP_START_EPOCH to trigger it.
    print("Running training epoch...")
    train_loss = train_fn(
        train_loader,
        model,
        optimizer,
        scheduler,
        epoch=Config.AWP_START_EPOCH,
        device=Config.DEVICE,
        use_awp=True,
    )
    print(f"Train Loss: {train_loss:.4f}")
    assert isinstance(train_loss, float)

    # Run evaluation
    print("Running evaluation...")
    val_loss, val_preds = eval_fn(val_loader, model, Config.DEVICE)
    print(f"Val Loss: {val_loss:.4f}")
    assert val_preds.shape == (Config.DEBUG_SAMPLE_SIZE, Config.NUM_CLASSES)

    # Run inference
    print("Running inference...")
    test_preds = inference_fn(test_loader, model, Config.DEVICE)
    assert test_preds.shape == (Config.DEBUG_SAMPLE_SIZE, Config.NUM_CLASSES)

    # 9. MLM Training (Brief Check)
    print("\n[9] MLM Training (engine.py)")
    # We won't run a full epoch as it takes time, but we call the function.
    # To make it fast, we will hack the loader to have length 1 for this loop
    # or just rely on the small dataset size (20 samples / 4 batch = 5 steps).
    mlm_output_dir = os.path.join(Config.WORKING_DIR, "demo_mlm_model")
    train_mlm(mlm_loader, Config.MODEL_DEBERTA, mlm_output_dir, Config.DEVICE, epochs=1)

    assert os.path.exists(os.path.join(mlm_output_dir, "config.json"))
    print("MLM Training executed and model saved.")

    # 10. Optimization
    print("\n[10] Ensemble Optimization (optimization.py)")
    # Simulate OOF predictions for 3 models
    y_true = np.random.randint(0, 3, size=100)

    # Create synthetic probability distributions
    def generate_probs(n):
        p = np.random.rand(n, 3)
        return p / p.sum(axis=1, keepdims=True)

    oof_dict = {
        "model_a": generate_probs(100),
        "model_b": generate_probs(100),
        "model_c": generate_probs(100),
    }

    # Optimize weights
    weights = optimize_weights(oof_dict, y_true)

    assert len(weights) == 3
    assert np.isclose(sum(weights.values()), 1.0)

    # Apply ensemble
    test_preds_dict = {
        "model_a": generate_probs(50),
        "model_b": generate_probs(50),
        "model_c": generate_probs(50),
    }

    final_preds = apply_ensemble(test_preds_dict, weights)

    assert final_preds.shape == (50, 3)
    # Check row normalization
    row_sums = final_preds.sum(axis=1)
    assert np.allclose(row_sums, 1.0)
    # Check clipping (values should not be exactly 0 or 1 if clipped correctly,
    # but strictly within 1e-15 and 1-1e-15)
    assert final_preds.min() >= 1e-15
    assert final_preds.max() <= 1.0 - 1e-15

    print("Ensemble optimization and application verified.")

    print("\n--- Demonstration Complete ---")


if __name__ == "__main__":
    run_demo()
