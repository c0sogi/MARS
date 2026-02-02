import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler  # Cite {debug_lesson_10}
from transformers import get_linear_schedule_with_warmup

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_score, AWP
from library.data import (
    get_tokenizer,
    get_train_val_loaders,
    get_test_loader,
    get_mlm_loader,
)
from library.model import ToxicModel
from library.engine import run_mlm, train_fn, valid_fn


def demonstrate_utils():
    print("\n=== Demonstrating library.utils ===")

    # 1. Test Seed Everything
    seed_everything(42)
    r1 = np.random.rand(5)
    seed_everything(42)
    r2 = np.random.rand(5)
    assert np.allclose(
        r1, r2
    ), "seed_everything failed to produce reproducible numpy results"
    print("seed_everything: Verified reproducibility.")

    # 2. Test get_score (Mean Column-wise ROC AUC)
    # Case A: Perfect prediction
    y_true = np.array([[0, 1, 0, 1, 0, 0], [1, 0, 1, 0, 1, 1]])
    y_pred_perfect = np.array(
        [[0.1, 0.9, 0.1, 0.9, 0.1, 0.1], [0.9, 0.1, 0.9, 0.1, 0.9, 0.9]]
    )
    score_perfect = get_score(y_true, y_pred_perfect)
    assert score_perfect == 1.0, f"Expected score 1.0, got {score_perfect}"

    # Case B: Random/Bad prediction (should be around 0.5 or lower if inverted)
    y_pred_bad = np.array(
        [[0.9, 0.1, 0.9, 0.1, 0.9, 0.9], [0.1, 0.9, 0.1, 0.9, 0.1, 0.1]]
    )
    score_bad = get_score(y_true, y_pred_bad)
    assert score_bad < 1.0, "Bad predictions shouldn't have perfect score"

    print(f"get_score: Verified (Perfect: {score_perfect}, Inverted: {score_bad})")


def demonstrate_data_and_config():
    print("\n=== Demonstrating library.config and library.data ===")

    # Overriding Config for Speed and Isolation
    Config.debug = True  # Limits data to 1000 rows
    Config.working_dir = "./working/demo_execution"
    Config.output_dir = os.path.join(Config.working_dir, "output")
    Config.dapt_model_path = os.path.join(Config.working_dir, "dapt_backbone")

    # Reduce batch sizes and epochs for demo
    Config.train_batch_size = 4  # Cite {debug_lesson_10}
    Config.valid_batch_size = 4
    Config.dapt_batch_size = 4
    Config.epochs = 1
    Config.dapt_epochs = 1
    Config.print_freq = 10

    # Ensure directories exist
    os.makedirs(Config.working_dir, exist_ok=True)
    os.makedirs(Config.output_dir, exist_ok=True)

    print(f"Config configured for debug mode in: {Config.working_dir}")

    # Initialize Tokenizer
    tokenizer = get_tokenizer()
    print(f"Tokenizer loaded: {tokenizer.name_or_path}")

    # Test Data Loaders
    # Note: This will trigger _load_data which reads metadata and raw files
    print("Loading Train/Val Loaders (Debug Mode)...")
    train_loader, val_loader = get_train_val_loaders(tokenizer, load_cached_data=False)

    # Verify Train Loader
    batch = next(iter(train_loader))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape == (Config.train_batch_size, Config.max_length)
    assert batch["labels"].shape == (Config.train_batch_size, Config.num_labels)
    print(f"Train Loader verified. Batch shape: {batch['input_ids'].shape}")

    # Verify Val Loader
    val_batch = next(iter(val_loader))
    assert val_batch["input_ids"].shape[0] == Config.valid_batch_size
    print(f"Val Loader verified. Batch shape: {val_batch['input_ids'].shape}")

    return train_loader, val_loader, tokenizer


def demonstrate_model(train_loader):
    print("\n=== Demonstrating library.model ===")

    device = Config.device
    model = ToxicModel(Config.model_name)
    model.to(device)

    # Verify Model Architecture Components
    assert hasattr(model, "fusion"), "Model missing MultiLayerFusion"
    assert hasattr(model, "attention_pool"), "Model missing LinearAttentionPooling"
    print("Model architecture components verified.")

    # Run Forward Pass
    model.eval()
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    # Check Output Shape
    assert logits.shape == (Config.train_batch_size, Config.num_labels)
    print(f"Forward pass successful. Logits shape: {logits.shape}")

    return model


def demonstrate_engine(model, train_loader, val_loader, tokenizer):
    print("\n=== Demonstrating library.engine ===")

    # 1. Run MLM (Domain Adaptive Pre-training)
    # This function internally creates its own loader and saves the model
    print("Running DAPT (MLM) - Short run...")

    # Temporarily move model to CPU to avoid OOM during MLM
    model.cpu()
    torch.cuda.empty_cache()

    run_mlm()

    # Move model back to GPU
    model.to(Config.device)

    # Verify DAPT output
    assert os.path.exists(Config.dapt_model_path), "DAPT model directory not created"
    assert os.path.exists(
        os.path.join(Config.dapt_model_path, "config.json")
    ), "DAPT config not saved"
    print("DAPT execution verified.")

    # 2. Run Supervised Training Step
    print("Running Supervised Training Step...")

    # Setup Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=Config.learning_rate)
    num_train_steps = len(train_loader) * Config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    # Setup Scaler for AMP
    scaler = GradScaler()  # Cite {debug_lesson_10}

    # Setup AWP (Adversarial Weight Perturbation)
    awp = AWP(
        model, optimizer, adv_lr=Config.awp_lr, adv_eps=Config.awp_eps, scaler=scaler
    )

    # Run one epoch of training
    # We force epoch=Config.awp_start_epoch to trigger AWP logic usage for demonstration
    epoch_idx = max(1, Config.awp_start_epoch)
    avg_loss = train_fn(
        model,
        train_loader,
        optimizer,
        scheduler,
        Config.device,
        epoch=epoch_idx,
        awp=awp,
        scaler=scaler,
    )

    assert not np.isnan(avg_loss), "Training Loss is NaN"
    print(f"Training step complete. Avg Loss: {avg_loss:.4f}")

    # 3. Run Validation Step
    print("Running Validation Step...")
    val_loss, val_score = valid_fn(model, val_loader, Config.device)

    assert not np.isnan(val_loss), "Validation Loss is NaN"
    assert 0.0 <= val_score <= 1.0, f"Validation Score out of bounds: {val_score}"
    print(f"Validation complete. Loss: {val_loss:.4f}, AUC: {val_score:.4f}")


if __name__ == "__main__":
    # Ensure clean state
    seed_everything(42)

    try:
        # 1. Verify Utils
        demonstrate_utils()

        # 2. Verify Data Loading & Config
        train_loader, val_loader, tokenizer = demonstrate_data_and_config()

        # 3. Verify Model
        model = demonstrate_model(train_loader)

        # 4. Verify Engine (Training/Evaluation loops)
        demonstrate_engine(model, train_loader, val_loader, tokenizer)

        print("\nAll demonstrations and verifications passed successfully!")

    except AssertionError as e:
        print(f"\n[FAIL] Assertion failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n[FAIL] An unexpected error occurred: {e}")
        # Print stack trace for debugging
        import traceback

        traceback.print_exc()
        exit(1)
