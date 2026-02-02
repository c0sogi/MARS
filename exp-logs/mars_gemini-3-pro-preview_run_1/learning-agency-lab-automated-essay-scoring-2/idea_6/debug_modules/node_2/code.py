import os
import sys
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

# Import classes and functions from the provided library
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import get_supervised_loaders, get_mlm_loader
from library.model import EssayModel
from library.engine import (
    get_optimizer_params,
    train_mlm,
    train_fn,
    valid_fn,
    inference_fn,
)
from library.awp import AWP


def run_essay_scoring_demo():
    # ==========================================
    # 1. Configuration for Fast Demonstration
    # ==========================================
    print(">>> [1/7] Configuring for Speed...")

    # Override Config defaults to ensure the demo runs quickly (Debug Mode)
    Config.debug = True  # Uses only ~100 samples
    Config.model_name = "microsoft/deberta-v3-xsmall"  # Use a tiny model for speed
    Config.max_length = 128  # Shorten sequence length
    Config.train_batch_size = 4
    Config.valid_batch_size = 8
    Config.mlm_batch_size = 4
    Config.epochs = 1
    Config.mlm_epochs = 1
    Config.print_freq = 10
    Config.gradient_accumulation_steps = 1
    Config.awp_start_epoch = 0  # Enable AWP immediately for demonstration

    # Ensure reproducibility
    seed_everything(Config.seed)
    logger = get_logger()
    logger.info("Configuration updated for demo mode.")

    # ==========================================
    # 2. Tokenizer and Data Loading
    # ==========================================
    print(">>> [2/7] Loading Tokenizer and Data...")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Get Supervised DataLoaders (Train, Val, Test)
    # load_cached_data=False ensures we process the 'debug' subset fresh
    train_loader, val_loader, test_loader = get_supervised_loaders(
        tokenizer, load_cached_data=False
    )

    # Verify SFT Batch Structure
    sft_batch = next(iter(train_loader))
    assert "input_ids" in sft_batch
    assert "attention_mask" in sft_batch
    assert "labels" in sft_batch
    assert sft_batch["input_ids"].shape[0] <= Config.train_batch_size
    print("SFT DataLoaders verified.")

    # Get MLM DataLoader
    mlm_loader = get_mlm_loader(tokenizer, load_cached_data=False)

    # Verify MLM Batch Structure
    mlm_batch = next(iter(mlm_loader))
    assert "input_ids" in mlm_batch
    assert "labels" in mlm_batch  # DataCollatorForLanguageModeling adds labels
    print("MLM DataLoader verified.")

    # ==========================================
    # 3. Stage 1: Domain Adaptation (MLM)
    # ==========================================
    print(">>> [3/7] Running Stage 1: MLM Training...")

    # Train MLM (Saves checkpoint to Config.mlm_model_dir)
    train_mlm(mlm_loader)

    # Verify Checkpoint Creation
    assert os.path.exists(Config.mlm_model_dir), "MLM checkpoint dir not created"
    # Check for model files (e.g., config.json, pytorch_model.bin or safetensors)
    files = os.listdir(Config.mlm_model_dir)
    assert any("config.json" in f for f in files), "MLM config not saved"
    print(f"MLM Training complete. Checkpoint saved to {Config.mlm_model_dir}")

    # ==========================================
    # 4. Model Initialization (SFT)
    # ==========================================
    print(">>> [4/7] Initializing EssayModel from MLM Checkpoint...")

    # Initialize model using the domain-adapted weights
    model = EssayModel(checkpoint_path=Config.mlm_model_dir, pretrained=True)
    model.to(Config.device)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_ids = sft_batch["input_ids"].to(Config.device)
        dummy_mask = sft_batch["attention_mask"].to(Config.device)
        dummy_out = model(dummy_ids, dummy_mask)

    assert dummy_out.shape == (
        dummy_ids.size(0),
    ), f"Output shape mismatch: {dummy_out.shape}"
    print("Model initialized and forward pass verified.")

    # ==========================================
    # 5. Stage 2: Supervised Training Loop
    # ==========================================
    print(">>> [5/7] Running Stage 2: Supervised Fine-Tuning...")

    # Optimizer
    optimizer_params = get_optimizer_params(
        model, Config.learning_rate, Config.weight_decay, Config.llrd_decay
    )
    optimizer = torch.optim.AdamW(optimizer_params)

    # Adversarial Weight Perturbation (AWP)
    awp = AWP(
        model,
        optimizer,
        adv_lr=Config.awp_lr,
        adv_eps=Config.awp_eps,
        start_epoch=Config.awp_start_epoch,
    )

    # Training Epoch
    train_loss = train_fn(
        model,
        train_loader,
        optimizer,
        scheduler=None,  # Skipping scheduler for simple demo
        epoch=0,
        awp=awp,
    )
    print(f"Epoch 0 Training Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # Validation
    val_loss, val_kappa = valid_fn(model, val_loader)
    print(f"Validation Loss: {val_loss:.4f}, QWK Score: {val_kappa:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"

    # ==========================================
    # 6. Inference
    # ==========================================
    print(">>> [6/7] Running Inference on Test Set...")

    predictions = inference_fn(model, test_loader)

    # Verify Predictions
    assert len(predictions) == len(test_loader.dataset), "Prediction count mismatch"
    print(f"Generated {len(predictions)} predictions.")

    # ==========================================
    # 7. Submission Generation
    # ==========================================
    print(">>> [7/7] Generating Submission File...")

    # In debug mode, test_loader is a subset. We create a submission for just that subset.
    test_ids = test_loader.dataset.df[Config.id_col].values

    submission = pd.DataFrame({Config.id_col: test_ids, "score": predictions})

    # Apply rounding and clipping as per metric requirements
    submission["score"] = np.round(np.clip(submission["score"], 1, 6)).astype(int)

    # Save
    submission.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
    print("Sample predictions:")
    print(submission.head())

    print("\n>>> Demo Completed Successfully!")


if __name__ == "__main__":
    run_essay_scoring_demo()
