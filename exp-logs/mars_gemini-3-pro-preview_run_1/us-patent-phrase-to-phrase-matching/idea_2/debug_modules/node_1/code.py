import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import logging
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import library modules
from library.config import Config
from library.utils import seed_everything, compute_pearson
from library.dataset import load_and_preprocess_data, PhraseDataset
from library.model import PhraseSimilarityModel
from library.engine import train_fn, eval_fn, inference_fn

# Suppress warnings and logs for cleaner output
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


def run_demonstration():
    print("=" * 40)
    print("Running Patent Phrase Similarity Demo")
    print("=" * 40)

    # 1. Configuration Override for Speed
    # We modify the Config class attributes directly to optimize for a quick demo run.
    print("[1] Configuring environment...")
    Config.debug = True
    Config.debug_sample_size = 50  # Use only 50 samples
    Config.epochs = 2
    Config.train_batch_size = 8
    Config.valid_batch_size = 8
    Config.model_name = "prajjwal1/bert-tiny"  # Use a tiny model for speed
    Config.learning_rate = 1e-4
    Config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure reproducibility
    seed_everything(Config.seed)
    print(f"    Device: {Config.device}")
    print(f"    Model: {Config.model_name}")
    print(f"    Debug Mode: {Config.debug}")

    # 2. Data Loading
    print("\n[2] Loading and Preprocessing Data...")
    # Load small subsets of data
    df_train = load_and_preprocess_data("train", load_cached_data=False, debug=True)
    df_val = load_and_preprocess_data("val", load_cached_data=False, debug=True)
    df_test = load_and_preprocess_data("test", load_cached_data=False, debug=True)

    # Validate DataFrames
    assert (
        len(df_train) == Config.debug_sample_size
    ), f"Train size mismatch: {len(df_train)}"
    assert "score" in df_train.columns, "Train data missing 'score' column"
    assert "context" in df_train.columns, "Train data missing 'context' column"
    print(f"    Train shape: {df_train.shape}")
    print(f"    Val shape: {df_val.shape}")
    print(f"    Test shape: {df_test.shape}")

    # 3. Dataset and DataLoader
    print("\n[3] Initializing Datasets and DataLoaders...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    train_dataset = PhraseDataset(df_train, tokenizer, mode="train")
    val_dataset = PhraseDataset(df_val, tokenizer, mode="val")
    test_dataset = PhraseDataset(df_test, tokenizer, mode="test")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=Config.train_batch_size, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=Config.valid_batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=Config.valid_batch_size, shuffle=False
    )

    # Verify batch structure
    sample_batch = next(iter(train_loader))
    assert "input_ids" in sample_batch
    assert "attention_mask" in sample_batch
    assert "labels" in sample_batch
    assert sample_batch["input_ids"].shape[0] == Config.train_batch_size
    print("    DataLoader verification passed.")

    # 4. Model Initialization
    print("\n[4] Initializing Model...")
    model = PhraseSimilarityModel(model_name=Config.model_name, pretrained=True)
    model.to(Config.device)

    # Verify forward pass
    print("    Verifying forward pass...")
    model.eval()
    with torch.no_grad():
        input_ids = sample_batch["input_ids"].to(Config.device)
        attention_mask = sample_batch["attention_mask"].to(Config.device)
        labels = sample_batch["labels"].to(Config.device)

        output = model(input_ids, attention_mask, labels=labels)
        assert output.loss is not None, "Model output missing loss"
        assert output.logits.shape == (
            Config.train_batch_size,
            1,
        ), f"Logits shape mismatch: {output.logits.shape}"
    print("    Forward pass successful.")

    # 5. Training Loop Demonstration
    print("\n[5] Running Training Loop (Demo)...")
    optimizer = AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    num_training_steps = len(train_loader) * Config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Initialize GradScaler for mixed precision (required by engine.py)
    scaler = torch.cuda.amp.GradScaler(enabled=(Config.device.type == "cuda"))

    for epoch in range(Config.epochs):
        print(f"    Epoch {epoch + 1}/{Config.epochs}")

        # Train
        train_loss, train_pearson = train_fn(
            model, train_loader, optimizer, scheduler, Config.device, scaler, epoch
        )

        # Validate
        val_loss, val_pearson = eval_fn(model, val_loader, Config.device)

        print(
            f"        Train Loss: {train_loss:.4f} | Train Pearson: {train_pearson:.4f}"
        )
        print(f"        Val Loss:   {val_loss:.4f} | Val Pearson:   {val_pearson:.4f}")

        # Basic assertions to ensure metrics are valid
        assert not np.isnan(train_loss), "Training loss is NaN"
        assert (
            -1.0 <= train_pearson <= 1.0 or train_pearson == 0.0
        ), "Invalid Pearson score"

    # 6. Inference Demonstration
    print("\n[6] Running Inference on Test Set...")
    predictions = inference_fn(model, test_loader, Config.device)

    # Verify predictions
    assert len(predictions) == len(df_test), "Prediction count mismatch"
    assert isinstance(predictions, np.ndarray), "Predictions should be a numpy array"

    # Clip predictions to valid range for submission
    predictions = np.clip(predictions, 0.0, 1.0)

    print(f"    Generated {len(predictions)} predictions.")
    print(f"    Sample predictions: {predictions[:5]}")

    # 7. Metric Utility Verification
    print("\n[7] Verifying Metric Utility...")
    dummy_preds = np.array([0.0, 0.5, 1.0])
    dummy_targets = np.array([0.0, 0.5, 1.0])
    perfect_score = compute_pearson(dummy_preds, dummy_targets)
    assert np.isclose(
        perfect_score, 1.0
    ), f"Metric utility failed: expected 1.0, got {perfect_score}"

    dummy_preds_inv = np.array([1.0, 0.5, 0.0])
    inverse_score = compute_pearson(dummy_preds_inv, dummy_targets)
    assert np.isclose(
        inverse_score, -1.0
    ), f"Metric utility failed: expected -1.0, got {inverse_score}"
    print("    Metric verification passed.")

    print("\n" + "=" * 40)
    print("Demonstration Completed Successfully")
    print("=" * 40)


if __name__ == "__main__":
    run_demonstration()
