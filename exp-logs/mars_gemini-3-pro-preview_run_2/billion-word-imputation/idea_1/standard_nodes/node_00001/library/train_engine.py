import os
import time
import math
import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.data_factory import get_dataloaders
from library.model_factory import load_model_and_tokenizer


def evaluate(model, dataloader, device, use_fp16):
    """
    Evaluates the model on the validation set.
    Returns the average loss and perplexity.
    """
    model.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast(enabled=use_fp16):
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss

            total_loss += loss.item()
            total_steps += 1

    avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float("inf")

    return avg_loss, perplexity


def run_training(
    epochs: int = Config.EPOCHS,
    max_steps: int = Config.MAX_STEPS,
    load_cached_data: bool = True,
    patience: int = 3,
):
    """
    Main training function.

    Args:
        epochs: Number of training epochs.
        max_steps: Maximum number of training steps (overrides epochs if > 0).
        load_cached_data: Whether to load pre-tokenized data from cache.
        patience: Number of evaluation intervals to wait for improvement before early stopping.
    """
    # 1. Load Model and Tokenizer
    model, tokenizer = load_model_and_tokenizer()
    device = Config.get_device()

    # 2. Prepare DataLoaders
    train_dataloader, val_dataloader = get_dataloaders(
        tokenizer, load_cached_data=load_cached_data
    )

    # 3. Setup Optimization
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": Config.WEIGHT_DECAY,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=Config.LEARNING_RATE, eps=Config.ADAM_EPSILON
    )

    # Calculate total training steps
    num_update_steps_per_epoch = len(train_dataloader)
    if max_steps > 0:
        t_total = max_steps
        epochs = math.ceil(max_steps / num_update_steps_per_epoch)
    else:
        t_total = num_update_steps_per_epoch * epochs

    num_warmup_steps = int(t_total * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
    )

    # Mixed Precision Scaler
    scaler = GradScaler(enabled=Config.USE_FP16)

    # 4. Training Loop
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataloader.dataset)}")
    print(f"  Num Epochs = {epochs}")
    print(f"  Instantaneous batch size per device = {Config.TRAIN_BATCH_SIZE}")
    print(f"  Total optimization steps = {t_total}")

    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0

    model.zero_grad()

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # Forward pass with Mixed Precision
            with autocast(enabled=Config.USE_FP16):
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss

            # Backward pass
            scaler.scale(loss).backward()

            # Unscale and clip gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            model.zero_grad()

            global_step += 1
            epoch_loss += loss.item()

            # Logging
            if global_step % Config.LOGGING_STEPS == 0:
                cur_loss = epoch_loss / (step + 1)
                elapsed = time.time() - start_time
                print(
                    f"Step {global_step}/{t_total} | Loss: {cur_loss} | Time: {elapsed:.2f}s"
                )

            # Evaluation & Checkpointing
            if global_step % Config.EVAL_STEPS == 0 or (
                max_steps > 0 and global_step >= max_steps
            ):
                print(f"Running evaluation at step {global_step}...")
                val_loss, val_perplexity = evaluate(
                    model, val_dataloader, device, Config.USE_FP16
                )

                print(f"Validation Loss: {val_loss}")
                print(f"Validation Perplexity: {val_perplexity}")

                if val_loss < best_val_loss:
                    print(f"New best model found! Saving to {Config.MODEL_SAVE_PATH}")
                    best_val_loss = val_loss
                    patience_counter = 0

                    # Save model and tokenizer
                    model.save_pretrained(Config.MODEL_SAVE_PATH)
                    tokenizer.save_pretrained(Config.MODEL_SAVE_PATH)
                else:
                    patience_counter += 1
                    print(f"No improvement. Patience: {patience_counter}/{patience}")

                model.train()

                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    return

                if max_steps > 0 and global_step >= max_steps:
                    print("Max steps reached.")
                    return

        # End of Epoch Evaluation (if not already evaluated at this exact step)
        if global_step % Config.EVAL_STEPS != 0:
            print(f"End of Epoch {epoch+1} evaluation...")
            val_loss, val_perplexity = evaluate(
                model, val_dataloader, device, Config.USE_FP16
            )

            print(f"Validation Loss: {val_loss}")
            print(f"Validation Perplexity: {val_perplexity}")

            if val_loss < best_val_loss:
                print(f"New best model found! Saving to {Config.MODEL_SAVE_PATH}")
                best_val_loss = val_loss
                patience_counter = 0
                model.save_pretrained(Config.MODEL_SAVE_PATH)
                tokenizer.save_pretrained(Config.MODEL_SAVE_PATH)
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training completed.")
