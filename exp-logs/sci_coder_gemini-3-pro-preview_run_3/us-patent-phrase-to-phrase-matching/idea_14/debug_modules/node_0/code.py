import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd
import warnings
from transformers import AutoTokenizer

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import prepare_loaders
from library.model import ScalarMixingModel
from library.loss import HybridLoss
from library.awp import AWP
from library.ema import ModelEMA
from library.engine import train_fn, valid_fn, inference_fn

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_demo():
    print("============================================================")
    print("      Phrase Matching Task - Library Usage Demonstration    ")
    print("============================================================")

    # -------------------------------------------------------------------------
    # 1. Configuration Patching
    # -------------------------------------------------------------------------
    # We modify the Config class attributes at runtime to adapt for a fast demo.
    # This ensures we don't modify the original file but change behavior.

    print("\n[1] Configuring environment for fast demonstration...")

    # Use a tiny model for speed
    Config.model_name = "prajjwal1/bert-tiny"

    # Enable debug mode to use a very small subset of data (100 samples)
    Config.debug = True

    # Reduce training duration
    Config.epochs = 1
    Config.train_batch_size = 16
    Config.valid_batch_size = 16
    Config.inference_batch_size = 16

    # Update paths to use a separate demo directory
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    Config.output_dir = demo_dir
    Config.cpc_context_map_path = os.path.join(demo_dir, "context_map.parquet")
    Config.train_cache_path = os.path.join(demo_dir, "train_cache.parquet")
    Config.val_cache_path = os.path.join(demo_dir, "val_cache.parquet")
    Config.test_cache_path = os.path.join(demo_dir, "test_cache.parquet")
    Config.model_save_path = os.path.join(demo_dir, "model_demo")

    # Set device
    Config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"    Model: {Config.model_name}")
    print(f"    Device: {Config.device}")
    print(f"    Output Directory: {Config.output_dir}")

    # Set random seed for reproducibility
    seed_everything(Config.seed)

    # -------------------------------------------------------------------------
    # 2. Data Preparation
    # -------------------------------------------------------------------------
    print("\n[2] Preparing Data...")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Prepare Loaders (this handles caching, context mapping, and tokenization)
    # We force load_cached_data=False to demonstrate the processing logic
    train_loader, val_loader, test_loader = prepare_loaders(
        tokenizer, load_cached_data=False
    )

    # Verification
    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")

    batch = next(iter(train_loader))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "label" in batch
    assert batch["input_ids"].shape[0] == Config.train_batch_size
    print("    Assertion Passed: DataLoader produces correct batch structure.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("\n[3] Initializing ScalarMixingModel...")

    model = ScalarMixingModel(pretrained=True)
    model.to(Config.device)

    # Verification: Forward pass with a dummy batch
    with torch.no_grad():
        dummy_ids = batch["input_ids"].to(Config.device)
        dummy_mask = batch["attention_mask"].to(Config.device)
        dummy_type = batch["token_type_ids"].to(Config.device)

        logits, aux_logits = model(dummy_ids, dummy_mask, dummy_type)

    assert logits.shape == (Config.train_batch_size,)
    assert aux_logits.shape == (Config.train_batch_size, Config.num_aux_classes)
    print("    Assertion Passed: Model forward pass returns correct shapes.")

    # -------------------------------------------------------------------------
    # 4. Training Setup (Loss, Optimizer, AWP, EMA)
    # -------------------------------------------------------------------------
    print("\n[4] Setting up Training Components...")

    # Hybrid Loss (MSE + Pearson + CrossEntropy)
    loss_fn = HybridLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * Config.epochs
    )

    # Adversarial Weight Perturbation (AWP)
    # We set start_epoch=0 to verify it runs immediately in this demo
    awp = AWP(model, optimizer, adv_lr=1e-4, adv_eps=1e-3, start_epoch=0)

    # Exponential Moving Average (EMA)
    ema = ModelEMA(model, decay=0.999)

    print("    Components initialized: HybridLoss, AdamW, AWP, EMA.")

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    print("\n[5] Running Training (1 Epoch)...")

    train_loss = train_fn(
        train_loader=train_loader,
        model=model,
        optimizer=optimizer,
        device=Config.device,
        scheduler=scheduler,
        epoch=0,
        config=Config,
        awp=awp,
        ema=ema,
        loss_fn=loss_fn,
    )

    print(f"    Training finished. Average Loss: {train_loss:.6f}")
    assert not np.isnan(train_loss), "Training loss resulted in NaN."

    # -------------------------------------------------------------------------
    # 6. Validation
    # -------------------------------------------------------------------------
    print("\n[6] Running Validation...")

    val_loss, val_score = valid_fn(
        val_loader=val_loader,
        model=model,
        device=Config.device,
        config=Config,
        ema=ema,
        loss_fn=loss_fn,
    )

    print(f"    Validation Loss: {val_loss:.6f}")
    print(f"    Pearson Score:   {val_score:.6f}")
    # Note: Score can be negative if correlation is inverse, but usually > -1
    assert -1.0 <= val_score <= 1.0, "Pearson score out of expected range."

    # -------------------------------------------------------------------------
    # 7. Inference
    # -------------------------------------------------------------------------
    print("\n[7] Running Inference on Test Set...")

    predictions = inference_fn(
        test_loader=test_loader,
        model=model,
        device=Config.device,
        config=Config,
        ema=ema,
    )

    print(f"    Predictions generated: {len(predictions)} samples.")
    print(f"    Sample predictions: {predictions[:5]}")

    assert len(predictions) == len(test_loader.dataset)
    assert (predictions >= 0).all() and (
        predictions <= 1
    ).all(), "Predictions contain values outside [0, 1]."

    # -------------------------------------------------------------------------
    # 8. Submission Generation (Mock)
    # -------------------------------------------------------------------------
    print("\n[8] Generating Submission File...")

    # Load sample submission to get IDs
    # Note: In debug mode, test set is truncated, so we must match indices
    df_test = pd.read_csv(Config.test_path)
    if Config.debug:
        df_test = df_test.iloc[:100]

    submission = pd.DataFrame({"id": df_test["id"], "score": predictions})

    submission_path = os.path.join(Config.output_dir, "submission_demo.csv")
    submission.to_csv(submission_path, index=False)
    print(f"    Submission saved to: {submission_path}")

    print("\n============================================================")
    print("      Demonstration Completed Successfully                  ")
    print("============================================================")


if __name__ == "__main__":
    run_demo()
