import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import library modules
from library.config import Config
from library.dataset import load_dataset, PhraseDataset
from library.model import CustomModel
from library.loss import HybridLoss
from library.engine import get_optimizer_params, train_fn, valid_fn, inference_fn
from library.utils import seed_everything, AWP


def run_demo():
    # 1. Setup and Configuration Override
    print(">>> Setting up configuration...")
    seed_everything(Config.seed)

    # Override Config for speed (Tiny model, few epochs, small batch)
    Config.model_name = "prajjwal1/bert-tiny"
    Config.epochs = 1
    Config.batch_size = 8
    Config.print_freq = 2
    Config.debug = True

    # Check device
    device = Config.device
    print(f"Using device: {device}")

    # 2. Data Loading & Preprocessing
    print(">>> Loading and preprocessing data...")
    # Load raw data with context mapping
    # We use 'train' mode which reads from metadata/train.csv
    df_full = load_dataset(mode="train", load_cached_data=False)

    # Take a small subset for demonstration (50 samples)
    # Ensure we have some variety in scores for the stratified split logic to work generally,
    # though here we just take head for speed.
    df_demo = df_full.head(50).copy()

    # Split into train/val (80/20)
    train_len = int(len(df_demo) * 0.8)
    df_train = df_demo.iloc[:train_len].reset_index(drop=True)
    df_val = df_demo.iloc[train_len:].reset_index(drop=True)

    print(f"Train shape: {df_train.shape}, Val shape: {df_val.shape}")

    # 3. Tokenizer and Dataset
    print(">>> Initializing Tokenizer and Datasets...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    train_dataset = PhraseDataset(df_train, tokenizer, max_length=64)
    val_dataset = PhraseDataset(df_val, tokenizer, max_length=64)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=0,  # 0 for simple debugging
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Verification: Check batch structure
    sample_batch = next(iter(train_loader))
    assert "input_ids" in sample_batch
    assert "label" in sample_batch
    assert sample_batch["input_ids"].shape[0] <= Config.batch_size
    print("Dataset verification passed.")

    # 4. Model Initialization
    print(">>> Initializing Model...")
    # pretrained=True downloads weights for the backbone
    model = CustomModel(config_path=Config.model_name, pretrained=True)
    model.to(device)

    # Verification: Check output shape
    dummy_input = sample_batch["input_ids"].to(device)
    dummy_mask = sample_batch["attention_mask"].to(device)
    with torch.no_grad():
        dummy_out = model(dummy_input, dummy_mask)

    # Expecting dict with 'logits' (B, 1) and 'class_logits' (B, 5)
    assert "logits" in dummy_out
    assert "class_logits" in dummy_out
    assert dummy_out["logits"].shape == (dummy_input.size(0), 1)
    assert dummy_out["class_logits"].shape == (dummy_input.size(0), 5)
    print("Model forward pass verification passed.")

    # 5. Optimizer, Scheduler, Loss
    print(">>> Setting up Optimizer, Scheduler, and Loss...")
    optimizer_parameters = get_optimizer_params(
        model,
        encoder_lr=Config.encoder_lr,
        decoder_lr=Config.head_lr,
        weight_decay=Config.weight_decay,
    )

    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=Config.encoder_lr, eps=Config.eps, betas=Config.betas
    )

    num_train_steps = len(train_loader) * Config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    criterion = HybridLoss()

    # Initialize AWP (Adversarial Weight Perturbation)
    # We set start_epoch to 0 for demo purposes to verify it runs
    awp = AWP(
        model, optimizer, adv_lr=Config.awp_lr, adv_eps=Config.awp_eps, start_epoch=0
    )

    # 6. Training Loop
    print(">>> Starting Training...")
    avg_loss = train_fn(
        fold=0,
        train_loader=train_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epoch=0,
        scheduler=scheduler,
        device=device,
        awp=awp,
        scaler=None,  # Not using mixed precision for simple CPU/Tiny demo
    )

    print(f"Training Epoch 0 complete. Avg Loss: {avg_loss:.4f}")
    assert not np.isnan(avg_loss), "Training loss is NaN"

    # 7. Validation Loop
    print(">>> Starting Validation...")
    val_loss, val_score = valid_fn(
        val_loader=val_loader, model=model, criterion=criterion, device=device
    )
    print(f"Validation complete. Loss: {val_loss:.4f}, Pearson Score: {val_score:.4f}")

    # 8. Inference
    print(">>> Running Inference...")
    predictions = inference_fn(val_loader, model, device)

    # Verification: Predictions
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5]}")

    assert len(predictions) == len(df_val)
    assert (predictions >= 0).all() and (
        predictions <= 1
    ).all(), "Predictions out of range [0, 1]"

    print("\n>>> Demo completed successfully!")


if __name__ == "__main__":
    run_demo()
