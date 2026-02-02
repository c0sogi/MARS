import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, get_score, find_optimal_weights
from library.data_factory import get_dataloaders
from library.model_factory import ToxicityModel
from library.optimization import get_optimizer, get_scheduler
from library.awp import AWP
from library.loops import train_fn, valid_fn, inference_fn
from library.linear_model import train_linear_pipeline


def create_mini_dataset(n_samples=50):
    """Creates a mini dataset in the working directory for speed."""
    print(f"Creating mini dataset with {n_samples} samples...")

    # Read original metadata
    train_df = pd.read_csv(Config.TRAIN_PATH).head(n_samples)
    val_df = pd.read_csv(Config.VAL_PATH).head(n_samples)
    test_df = pd.read_csv(Config.TEST_PATH).head(n_samples)

    # Define new paths
    mini_dir = os.path.join(Config.WORKING_DIR, "mini_data")
    os.makedirs(mini_dir, exist_ok=True)

    train_path = os.path.join(mini_dir, "train.csv")
    val_path = os.path.join(mini_dir, "val.csv")
    test_path = os.path.join(mini_dir, "test.csv")

    # Save mini datasets
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, val_path, test_path


def demo_linear_model():
    print("\n=== DEMO: Linear Model Pipeline ===")

    # Ensure we don't use cached features from a previous full run if they exist
    Config.TFIDF_CACHE_DIR = os.path.join(Config.WORKING_DIR, "mini_tfidf_cache")
    if os.path.exists(Config.TFIDF_CACHE_DIR):
        shutil.rmtree(Config.TFIDF_CACHE_DIR)

    # Run pipeline
    # Note: train_linear_pipeline internally loads data from Config paths
    val_preds, test_preds = train_linear_pipeline(load_cached_data=False)

    # Verify outputs
    print("Verifying Linear Model outputs...")
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    assert val_preds.shape == (
        len(val_df),
        6,
    ), f"Val preds shape mismatch: {val_preds.shape}"
    assert test_preds.shape == (
        len(test_df),
        6,
    ), f"Test preds shape mismatch: {test_preds.shape}"

    print("Linear Model verification passed.")
    return val_preds


def demo_deep_learning():
    print("\n=== DEMO: Deep Learning Components ===")

    # 1. Configure for Speed
    # Use a tiny model to avoid large downloads and OOM on small instances
    Config.MODEL_A_NAME = "prajjwal1/bert-tiny"
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 8
    Config.GRAD_ACCUM_STEPS = 1
    Config.EPOCHS = 1
    Config.MAX_LEN = 32  # Short sequence length for speed
    Config.AWP_START_EPOCH = 0  # Force AWP to run immediately for demo

    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Factory
    print("Initializing Tokenizer and DataLoaders...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_A_NAME)

    # We use debug=False here because we already pointed Config paths to our mini dataset
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer=tokenizer,
        train_batch_size=Config.TRAIN_BATCH_SIZE,
        valid_batch_size=Config.VALID_BATCH_SIZE,
        max_len=Config.MAX_LEN,
        debug=False,
    )

    # Verify DataLoader
    batch = next(iter(train_loader))
    assert "input_ids" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape == (Config.TRAIN_BATCH_SIZE, Config.MAX_LEN)
    print("DataLoader verification passed.")

    # 3. Model Factory
    print("Initializing ToxicityModel...")
    model = ToxicityModel(model_name=Config.MODEL_A_NAME, pretrained=True)
    model.to(device)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_input = batch["input_ids"].to(device)
        dummy_mask = batch["attention_mask"].to(device)
        outputs = model(dummy_input, dummy_mask)
        assert outputs.shape == (
            Config.TRAIN_BATCH_SIZE,
            6,
        ), f"Model output shape mismatch: {outputs.shape}"
    print("Model initialization verification passed.")

    # 4. Optimization
    print("Initializing Optimizer and Scheduler...")
    optimizer = get_optimizer(
        model, learning_rate=1e-4, weight_decay=0.01, llrd_decay=0.9
    )
    scheduler = get_scheduler(
        optimizer, num_train_steps=len(train_loader) * Config.EPOCHS
    )

    # Verify Optimizer Groups (LLRD)
    # bert-tiny has 2 layers + embeddings + head.
    # We expect multiple param groups.
    assert (
        len(optimizer.param_groups) > 1
    ), "Optimizer should have multiple param groups for LLRD."
    print("Optimization setup verification passed.")

    # 5. AWP
    print("Initializing AWP...")
    awp = AWP(model, optimizer, adv_lr=1e-4, adv_eps=1e-2, start_epoch=0)

    # 6. Training Loop
    print("Running Training Loop (1 Epoch)...")
    criterion = nn.BCEWithLogitsLoss()

    avg_loss = train_fn(
        train_loader=train_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epoch=0,
        scheduler=scheduler,
        device=device,
        awp=awp,
        config=Config,
    )
    assert not np.isnan(avg_loss), "Training loss is NaN."
    print(f"Training Loop finished. Avg Loss: {avg_loss:.4f}")

    # 7. Validation Loop
    print("Running Validation Loop...")
    val_loss, val_score, val_preds = valid_fn(
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        device=device,
        config=Config,
    )
    assert val_preds.shape == (len(pd.read_csv(Config.VAL_PATH)), 6)
    print(f"Validation Loop finished. Score: {val_score:.4f}")

    # 8. Inference Loop
    print("Running Inference Loop...")
    test_preds = inference_fn(
        test_loader=test_loader, model=model, device=device, config=Config
    )
    assert test_preds.shape == (len(pd.read_csv(Config.TEST_PATH)), 6)
    print("Inference Loop finished.")

    return val_preds


def demo_utils(dl_preds, linear_preds):
    print("\n=== DEMO: Utility Functions ===")

    # Load ground truth
    val_df = pd.read_csv(Config.VAL_PATH)
    y_true = val_df[Config.LABEL_COLS].values

    # 1. Test get_score
    print("Testing get_score...")
    score = get_score(y_true, dl_preds)
    print(f"DL Model Score: {score:.4f}")
    assert 0 <= score <= 1, "Score out of range."

    # 2. Test find_optimal_weights (Ensembling)
    print("Testing find_optimal_weights...")
    preds_list = [dl_preds, linear_preds]

    weights = find_optimal_weights(preds_list, y_true)

    print(f"Optimal Weights: {weights}")
    assert np.isclose(np.sum(weights), 1.0), "Weights do not sum to 1."

    # Calculate ensemble score
    ensemble_preds = weights[0] * dl_preds + weights[1] * linear_preds
    ensemble_score = get_score(y_true, ensemble_preds)
    print(f"Ensemble Score: {ensemble_score:.4f}")

    print("Utility verification passed.")


if __name__ == "__main__":
    # 1. Setup
    seed_everything(42)

    # Create a temporary working directory for this demo
    Config.WORKING_DIR = "./working/demo_execution_script"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Create mini datasets and override Config paths
    # This ensures all subsequent library calls use this subset
    mini_train, mini_val, mini_test = create_mini_dataset(n_samples=100)
    Config.TRAIN_PATH = mini_train
    Config.VAL_PATH = mini_val
    Config.TEST_PATH = mini_test

    # 2. Run Linear Model Demo
    linear_val_preds = demo_linear_model()

    # 3. Run Deep Learning Demo
    dl_val_preds = demo_deep_learning()

    # 4. Run Utils/Ensemble Demo
    demo_utils(dl_val_preds, linear_val_preds)

    print("\n=== ALL DEMOS COMPLETED SUCCESSFULLY ===")
