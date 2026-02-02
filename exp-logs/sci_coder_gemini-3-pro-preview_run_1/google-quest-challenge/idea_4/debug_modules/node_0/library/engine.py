import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import seed_everything, compute_spearman_metric
from library.dataset import load_data, QuestDataset, CollateFactory, get_tokenizer
from library.model import QuestModel


def train_fn(model, dataloader, optimizer, scheduler, device, epoch):
    """
    Executes one training epoch with gradient accumulation.
    """
    model.train()
    running_loss = 0
    accumulation_steps = Config.gradient_accumulation_steps

    criterion = nn.BCEWithLogitsLoss()
    optimizer.zero_grad()

    for step, data in enumerate(dataloader):
        # Move inputs to device
        q_ids = data["q_input_ids"].to(device)
        q_mask = data["q_attention_mask"].to(device)
        a_ids = data["a_input_ids"].to(device)
        a_mask = data["a_attention_mask"].to(device)
        targets = data["labels"].to(device)

        # Forward pass
        logits = model(q_ids, q_mask, a_ids, a_mask)

        # Compute loss
        loss = criterion(logits, targets)

        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()

        # Track unscaled loss for reporting
        running_loss += loss.item() * accumulation_steps

        # Update weights every `accumulation_steps`
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # Handle remaining gradients if dataloader length is not divisible by accumulation_steps
    # (Optional, but strictly following the loop logic above implies updates happen only on modulo 0)

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def eval_fn(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0
    preds = []
    targets_list = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for data in dataloader:
            q_ids = data["q_input_ids"].to(device)
            q_mask = data["q_attention_mask"].to(device)
            a_ids = data["a_input_ids"].to(device)
            a_mask = data["a_attention_mask"].to(device)
            targets = data["labels"].to(device)

            logits = model(q_ids, q_mask, a_ids, a_mask)
            loss = criterion(logits, targets)

            running_loss += loss.item()

            # Apply sigmoid to get probabilities [0, 1]
            batch_preds = torch.sigmoid(logits)

            preds.append(batch_preds.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    preds = np.concatenate(preds, axis=0)
    targets_list = np.concatenate(targets_list, axis=0)

    # Compute metric
    spearman_score = compute_spearman_metric(targets_list, preds)

    return avg_loss, spearman_score, preds


def run_training():
    """
    Main orchestration function for training.
    """
    seed_everything(Config.seed)
    device = Config.device
    os.makedirs(Config.working_dir, exist_ok=True)

    print(f"Device: {device}")

    # 1. Load Data
    train_df, val_df, _ = load_data(load_cached_data=True, debug=Config.debug)

    # 2. Prepare Datasets & Dataloaders
    tokenizer = get_tokenizer()
    collate_fn = CollateFactory(tokenizer)

    train_dataset = QuestDataset(train_df, is_test=False)
    val_dataset = QuestDataset(val_df, is_test=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 3. Initialize Model
    model = QuestModel()
    model.to(device)

    # 4. Optimizer with Differential Learning Rates
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [
        # Backbone parameters (lower LR)
        {
            "params": [
                p
                for n, p in param_optimizer
                if "backbone" in n and not any(nd in n for nd in no_decay)
            ],
            "lr": Config.lr_backbone,
            "weight_decay": Config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in param_optimizer
                if "backbone" in n and any(nd in n for nd in no_decay)
            ],
            "lr": Config.lr_backbone,
            "weight_decay": 0.0,
        },
        # Head/Pooler parameters (higher LR)
        {
            "params": [
                p
                for n, p in param_optimizer
                if "backbone" not in n and not any(nd in n for nd in no_decay)
            ],
            "lr": Config.lr_head,
            "weight_decay": Config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in param_optimizer
                if "backbone" not in n and any(nd in n for nd in no_decay)
            ],
            "lr": Config.lr_head,
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_parameters)

    # 5. Scheduler
    # Calculate total training steps
    num_update_steps_per_epoch = len(train_loader) // Config.gradient_accumulation_steps
    num_training_steps = num_update_steps_per_epoch * Config.epochs
    num_warmup_steps = int(num_training_steps * Config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 6. Training Loop
    best_score = -1.0
    patience = 2  # Early stopping patience
    patience_counter = 0

    for epoch in range(Config.epochs):
        print(f"\nEpoch {epoch + 1}/{Config.epochs}")

        train_loss = train_fn(model, train_loader, optimizer, scheduler, device, epoch)
        print(f"Train Loss: {train_loss}")

        val_loss, val_score, _ = eval_fn(model, val_loader, device)
        print(f"Val Loss: {val_loss}")
        print(f"Val Spearman: {val_score}")

        # Save Best Model
        if val_score > best_score:
            print(f"Score Improved ({best_score} -> {val_score}). Saving Model...")
            best_score = val_score
            torch.save(model.state_dict(), Config.model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Score did not improve. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Best Validation Spearman Score: {best_score}")


def predict_and_submit():
    """
    Loads the best model, predicts on the test set, and saves the submission file.
    """
    print("\nStarting Prediction on Test Set...")
    seed_everything(Config.seed)
    device = Config.device

    # 1. Load Data
    _, _, test_df = load_data(load_cached_data=True, debug=Config.debug)

    # 2. Prepare Test Loader
    tokenizer = get_tokenizer()
    collate_fn = CollateFactory(tokenizer)
    test_dataset = QuestDataset(test_df, is_test=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 3. Load Model
    model = QuestModel()
    if os.path.exists(Config.model_save_path):
        model.load_state_dict(torch.load(Config.model_save_path, map_location=device))
        print(f"Loaded model from {Config.model_save_path}")
    else:
        print("Warning: Best model not found. Using random initialization.")

    model.to(device)
    model.eval()

    # 4. Inference
    preds = []
    with torch.no_grad():
        for data in test_loader:
            q_ids = data["q_input_ids"].to(device)
            q_mask = data["q_attention_mask"].to(device)
            a_ids = data["a_input_ids"].to(device)
            a_mask = data["a_attention_mask"].to(device)

            logits = model(q_ids, q_mask, a_ids, a_mask)
            batch_preds = torch.sigmoid(logits)
            preds.append(batch_preds.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    # 5. Create Submission DataFrame
    submission = pd.DataFrame(preds, columns=Config.target_cols)

    # Insert qa_id (ensure alignment with test_df)
    submission.insert(0, "qa_id", test_df["qa_id"].values)

    # 6. Save
    os.makedirs(Config.submission_dir, exist_ok=True)
    submission.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
