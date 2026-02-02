import os
import time
import gc
import numpy as np
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup

from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_dataloaders, get_tokenizer
from library.model_core import BackboneWrapper


def train_fold(
    model_name_or_path, fold_idx=0, debug=False, load_cached_data=True, epochs=None
):
    """
    Executes Phase 2: Supervised Fine-Tuning for a single fold.
    Loads data, initializes the backbone (potentially DAPT-adapted), and trains
    using BCEWithLogitsLoss. Implements Early Stopping based on Validation Loss.

    Args:
        model_name_or_path (str): Path to DAPT checkpoint or HF model name.
        fold_idx (int): Index of the current fold (used for naming outputs).
        debug (bool): If True, runs on a subset of data.
        load_cached_data (bool): Whether to use cached dataframes.
        epochs (int, optional): Override default epoch count from Config.

    Returns:
        str: Path to the saved best model state dict.
    """
    seed_everything(Config.SEED)

    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Determine output path
    output_model_path = os.path.join(
        Config.WORKING_DIR, f"finetuned_model_fold{fold_idx}.pth"
    )

    num_epochs = epochs if epochs is not None else Config.EPOCHS

    print(f"\n[FineTune] Starting Fold {fold_idx}")
    print(f"[FineTune] Model: {model_name_or_path}")
    print(f"[FineTune] Output: {output_model_path}")
    print(f"[FineTune] Device: {device}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    # We need the tokenizer corresponding to the model we are loading
    # If loading a local DAPT checkpoint, the tokenizer files should be there.
    tokenizer = get_tokenizer(model_name_or_path)

    train_loader, val_loader, _ = get_dataloaders(
        tokenizer=tokenizer,
        train_batch_size=Config.TRAIN_BATCH_SIZE,
        valid_batch_size=Config.VALID_BATCH_SIZE,
        load_cached_data=load_cached_data,
        debug=debug,
    )

    print(f"[FineTune] Train Batches: {len(train_loader)}")
    print(f"[FineTune] Val Batches: {len(val_loader)}")

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    model = BackboneWrapper(model_name_or_path, num_labels=len(Config.TARGET_COLS))
    model.to(device)

    # --------------------------------------------------------------------------
    # 4. Optimizer & Scheduler
    # --------------------------------------------------------------------------
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Calculate total training steps
    num_update_steps_per_epoch = len(train_loader) // Config.GRAD_ACCUMULATION_STEPS
    max_train_steps = num_update_steps_per_epoch * num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(max_train_steps * Config.WARMUP_RATIO),
        num_training_steps=max_train_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(Config.DEVICE == "cuda"))

    # --------------------------------------------------------------------------
    # 5. Training Loop
    # --------------------------------------------------------------------------
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        # --- Training ---
        model.train()
        train_loss_accum = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            q_mask = batch["q_mask"].to(device)
            a_mask = batch["a_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass with Mixed Precision
            with torch.cuda.amp.autocast(enabled=(Config.DEVICE == "cuda")):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    q_mask=q_mask,
                    a_mask=a_mask,
                    labels=labels,
                )
                loss = outputs["loss"]

                # Normalize loss for gradient accumulation
                loss = loss / Config.GRAD_ACCUMULATION_STEPS

            # Backward pass
            scaler.scale(loss).backward()

            train_loss_accum += loss.item() * Config.GRAD_ACCUMULATION_STEPS

            # Optimizer Step
            if (step + 1) % Config.GRAD_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss_accum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                q_mask = batch["q_mask"].to(device)
                a_mask = batch["a_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.cuda.amp.autocast(enabled=(Config.DEVICE == "cuda")):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        q_mask=q_mask,
                        a_mask=a_mask,
                        labels=labels,
                    )
                    loss = outputs["loss"]

                val_loss_accum += loss.item()

        avg_val_loss = val_loss_accum / len(val_loader)
        elapsed = time.time() - start_time

        # --- Logging ---
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Time: {elapsed:.0f}s | "
            f"Train Loss: {avg_train_loss:.8f} | "
            f"Val Loss: {avg_val_loss:.8f}"
        )

        # --- Early Stopping & Saving ---
        # We monitor Validation Loss strictly.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print(f"  -> Validation Loss Improved. Saving model to {output_model_path}")
            torch.save(model.state_dict(), output_model_path)
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("[FineTune] Early stopping triggered.")
            break

        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()

    print(f"[FineTune] Training finished. Best Val Loss: {best_val_loss:.8f}")
    return output_model_path
