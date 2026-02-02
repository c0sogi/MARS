import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from library.config import config
from library.utils import seed_everything, compute_spearman_metric
from library.data import get_dataloaders
from library.model import DebertaDualEncoder


def train_fn(model, dataloader, optimizer, scheduler, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    final_loss = 0.0
    count = 0

    for batch in dataloader:
        # Move batch to device
        q_input_ids = batch["q_input_ids"].to(device)
        q_attention_mask = batch["q_attention_mask"].to(device)
        a_input_ids = batch["a_input_ids"].to(device)
        a_attention_mask = batch["a_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(q_input_ids, q_attention_mask, a_input_ids, a_attention_mask)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        optimizer.step()
        scheduler.step()

        final_loss += loss.item() * labels.size(0)
        count += labels.size(0)

    return final_loss / count


def eval_fn(model, dataloader, criterion, device):
    """
    Performs evaluation on the validation set.
    """
    model.eval()
    final_loss = 0.0
    count = 0
    preds = []
    targets = []

    with torch.no_grad():
        for batch in dataloader:
            q_input_ids = batch["q_input_ids"].to(device)
            q_attention_mask = batch["q_attention_mask"].to(device)
            a_input_ids = batch["a_input_ids"].to(device)
            a_attention_mask = batch["a_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                q_input_ids,
                q_attention_mask,
                a_input_ids,
                a_attention_mask,
            )

            loss = criterion(logits, labels)

            final_loss += loss.item() * labels.size(0)
            count += labels.size(0)

            # Apply sigmoid for predictions
            preds.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(labels.cpu().numpy())

    avg_loss = final_loss / count
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    # Compute metric
    score = compute_spearman_metric(targets, preds)

    return avg_loss, score, preds


def get_optimizer_params(model):
    """
    Sets up differential learning rates and weight decay exclusion.
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = []

    for n, p in param_optimizer:
        if not p.requires_grad:
            continue

        # Determine Learning Rate
        # If parameter belongs to backbone, use lower LR
        if "backbone" in n:
            lr = config.lr_backbone
        else:
            lr = config.lr_head

        # Determine Weight Decay
        # Exclude bias and LayerNorms
        if any(nd in n for nd in no_decay):
            wd = 0.0
        else:
            wd = config.weight_decay

        optimizer_parameters.append({"params": [p], "lr": lr, "weight_decay": wd})

    return optimizer_parameters


def inference_fn(model, dataloader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            q_input_ids = batch["q_input_ids"].to(device)
            q_attention_mask = batch["q_attention_mask"].to(device)
            a_input_ids = batch["a_input_ids"].to(device)
            a_attention_mask = batch["a_attention_mask"].to(device)

            logits = model(
                q_input_ids,
                q_attention_mask,
                a_input_ids,
                a_attention_mask,
            )

            preds.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(preds, axis=0)


def run_training():
    # 1. Setup
    seed_everything(config.seed)
    device = torch.device(config.device)
    os.makedirs(config.working_dir, exist_ok=True)

    print(f"Device: {device}")

    # 2. Data
    # Initialize tokenizer from model name
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_loader, val_loader, test_loader, meta_dims = get_dataloaders(
        config, tokenizer, load_cached_data=True
    )

    # 3. Model
    model = DebertaDualEncoder(meta_dims)
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer_parameters = get_optimizer_params(model)
    optimizer = AdamW(
        optimizer_parameters,
        lr=config.lr_head,  # Base LR, overridden by per-param groups
        eps=config.eps,
        betas=config.betas,
    )

    num_train_steps = len(train_loader) * config.epochs
    num_warmup_steps = int(num_train_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_score = -1.0
    patience = 0
    patience_limit = 3  # Early stopping patience

    print("Starting training...")

    for epoch in range(config.epochs):
        start_time = time.time()

        # Train
        train_loss = train_fn(
            model, train_loader, optimizer, scheduler, criterion, device
        )

        # Validate
        val_loss, val_score, _ = eval_fn(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"Time: {elapsed:.0f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Spearman: {val_score}"
        )

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), config.model_save_path)
            print(f"Score Improved. Model Saved to {config.model_save_path}")
            patience = 0
        else:
            patience += 1
            print(f"Score did not improve. Patience: {patience}/{patience_limit}")

        if patience >= patience_limit:
            print("Early stopping triggered.")
            break

    # 6. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(config.model_save_path, map_location=device))

    test_preds = inference_fn(model, test_loader, device)

    # 7. Create Submission
    # Load test metadata to get qa_ids
    test_df = pd.read_csv(config.test_path)

    # Ensure we only take the qa_ids corresponding to the number of predictions
    # (In case debug mode truncated the dataset)
    if len(test_preds) != len(test_df):
        print(
            f"Warning: Prediction count {len(test_preds)} != Test DF length {len(test_df)}"
        )
        test_df = test_df.iloc[: len(test_preds)]

    submission = pd.DataFrame(test_preds, columns=config.target_cols)
    submission.insert(0, "qa_id", test_df["qa_id"].values)

    submission.to_csv(config.submission_path, index=False)
    print(f"Submission saved to {config.submission_path}")


if __name__ == "__main__":
    run_training()
