import os
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.model import EssayScorerModel
from library.dataset import get_dataloaders
from library.utils import set_seed, compute_qwk, get_llrd_optimizer_params


def train_fn(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    scaler,
    device,
    cfg,
    best_score,
    epoch,
):
    """
    Executes the training loop for a single epoch, handling gradient accumulation
    and mid-epoch validation.
    """
    model.train()

    running_loss = 0.0
    dataset_size = 0
    global_step = 0

    # Calculate total steps and validation interval steps
    num_batches = len(train_loader)
    val_steps = int(num_batches * cfg.val_check_interval)
    if val_steps == 0:
        val_steps = num_batches  # Validate at least once at the end

    print(
        f"Epoch {epoch+1}/{cfg.epochs} - Training started. Validation every {val_steps} steps."
    )

    start_time = time.time()

    for step, batch in enumerate(train_loader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        batch_size = input_ids.size(0)

        # Mixed Precision Forward Pass
        with autocast():
            outputs = model(input_ids, attention_mask)
            # Outputs are (batch_size, 1), labels are (batch_size)
            # Squeeze output to match label shape
            loss = nn.MSELoss()(outputs.view(-1), labels)

            # Scale loss for gradient accumulation
            loss = loss / cfg.gradient_accumulation_steps

        # Backward Pass
        scaler.scale(loss).backward()

        running_loss += loss.item() * cfg.gradient_accumulation_steps * batch_size
        dataset_size += batch_size

        # Optimizer Step (Gradient Accumulation)
        if (step + 1) % cfg.gradient_accumulation_steps == 0 or (
            step + 1
        ) == num_batches:
            # Unscale gradients
            scaler.unscale_(optimizer)

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            # Step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

        # Validation Check
        if (step + 1) % val_steps == 0 or (step + 1) == num_batches:
            current_train_loss = running_loss / dataset_size
            elapsed = time.time() - start_time

            print(
                f"Step {step+1}/{num_batches} | Train Loss: {current_train_loss:.6f} | Time: {elapsed:.0f}s"
            )

            # Validate
            val_score, val_loss = valid_fn(model, val_loader, device, cfg)
            print(f"Validation - QWK: {val_score} | Loss: {val_loss:.6f}")

            # Save Best Model
            if val_score > best_score:
                print(f"Score Improved ({best_score} -> {val_score}). Saving model...")
                best_score = val_score
                torch.save(model.state_dict(), cfg.model_save_path)
            else:
                print(f"Score did not improve (Best: {best_score}).")

            # Revert to train mode
            model.train()

    return best_score


def valid_fn(model, val_loader, device, cfg):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    preds = []
    targets = []
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            batch_size = input_ids.size(0)

            with autocast():
                outputs = model(input_ids, attention_mask)
                loss = nn.MSELoss()(outputs.view(-1), labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Collect predictions and targets
            preds.append(outputs.view(-1).float().cpu().numpy())
            targets.append(labels.cpu().numpy())

    all_preds = np.concatenate(preds)
    all_targets = np.concatenate(targets)

    avg_loss = running_loss / dataset_size
    qwk = compute_qwk(all_targets, all_preds)

    return qwk, avg_loss


def inference_fn(model, test_loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds = []
    essay_ids = []

    print("Starting Inference on Test Set...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ids = batch["essay_id"]

            with autocast():
                outputs = model(input_ids, attention_mask)

            preds.append(outputs.view(-1).float().cpu().numpy())
            essay_ids.extend(ids)

    all_preds = np.concatenate(preds)
    return essay_ids, all_preds


def run_training():
    """
    Main function to orchestrate training, validation, and submission.
    """
    cfg = Config()
    set_seed(cfg.seed)

    # 1. Prepare Data
    print("Preparing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(cfg, load_cached_data=True)

    # 2. Initialize Model
    print(f"Initializing Model: {cfg.model_name}")
    model = EssayScorerModel(cfg, pretrained=True)
    model.to(cfg.device)

    # 3. Optimizer (LLRD) and Scheduler
    optimizer_grouped_parameters = get_llrd_optimizer_params(
        model,
        base_lr=cfg.lr,
        head_lr=cfg.head_lr,
        weight_decay=cfg.weight_decay,
        llrd_decay=cfg.llrd_decay,
    )

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg.lr)

    # Calculate total training steps
    num_update_steps_per_epoch = len(train_loader) // cfg.gradient_accumulation_steps
    max_train_steps = cfg.epochs * num_update_steps_per_epoch

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(max_train_steps * cfg.warmup_ratio),
        num_training_steps=max_train_steps,
    )

    scaler = GradScaler()

    # 4. Training Loop
    best_score = -np.inf

    for epoch in range(cfg.epochs):
        best_score = train_fn(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            scaler,
            cfg.device,
            cfg,
            best_score,
            epoch,
        )

    print(f"Training Completed. Best Validation QWK: {best_score}")

    # 5. Inference and Submission
    # Load best model
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(cfg.model_save_path, map_location=cfg.device))

    ids, raw_preds = inference_fn(model, test_loader, cfg.device)

    # Post-processing: Clip to [1, 6] and round
    final_preds = np.clip(raw_preds, 1, 6)
    final_preds = np.round(final_preds).astype(int)

    # Create submission dataframe
    submission_df = pd.DataFrame({"essay_id": ids, "score": final_preds})

    # Save submission
    print(f"Saving submission to {cfg.submission_path}...")
    submission_df.to_csv(cfg.submission_path, index=False)
    print("Submission saved successfully.")

    # Verify submission format
    print("Head of submission file:")
    print(submission_df.head())


if __name__ == "__main__":
    run_training()
