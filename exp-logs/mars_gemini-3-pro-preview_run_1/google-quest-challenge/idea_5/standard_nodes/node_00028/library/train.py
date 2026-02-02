import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from library.config import Config
from library.utils import seed_everything, compute_spearmanr_score
from library.dataset import load_data, QuestDataset, Collate
from library.model import QuestModel


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    """
    Construct parameter groups for the optimizer with differential learning rates
    and weight decay exclusion for bias/LayerNorm.
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = []

    for name, params in param_optimizer:
        if not params.requires_grad:
            continue

        # Determine learning rate: backbone vs head
        # In QuestModel, the pretrained model is assigned to self.backbone
        lr = encoder_lr if "backbone" in name else decoder_lr

        # Determine weight decay: exclude bias and LayerNorm
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay

        optimizer_parameters.append({"params": [params], "weight_decay": wd, "lr": lr})

    return optimizer_parameters


def train_fn(model, dataloader, optimizer, scheduler, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    count = 0

    scaler = torch.cuda.amp.GradScaler()

    for step, batch in enumerate(dataloader):
        # Move batch to device
        q_input_ids = batch["q_input_ids"].to(device)
        q_attention_mask = batch["q_attention_mask"].to(device)
        a_input_ids = batch["a_input_ids"].to(device)
        a_attention_mask = batch["a_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast():
            logits = model(q_input_ids, q_attention_mask, a_input_ids, a_attention_mask)
            loss = criterion(logits, labels)

        if Config.gradient_accumulation_steps > 1:
            loss = loss / Config.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % Config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * Config.gradient_accumulation_steps
        count += 1

    avg_loss = total_loss / count
    return avg_loss


def valid_fn(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
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

            logits = model(q_input_ids, q_attention_mask, a_input_ids, a_attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            count += 1

            # Apply sigmoid for predictions
            probs = torch.sigmoid(logits)
            preds.append(probs.cpu().numpy())
            targets.append(labels.cpu().numpy())

    avg_loss = total_loss / count
    predictions = np.concatenate(preds, axis=0)
    ground_truth = np.concatenate(targets, axis=0)

    return avg_loss, predictions, ground_truth


def inference_fn(model, dataloader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            q_input_ids = batch["q_input_ids"].to(device)
            q_attention_mask = batch["q_attention_mask"].to(device)
            a_input_ids = batch["a_input_ids"].to(device)
            a_attention_mask = batch["a_attention_mask"].to(device)

            logits = model(q_input_ids, q_attention_mask, a_input_ids, a_attention_mask)
            probs = torch.sigmoid(logits)
            preds.append(probs.cpu().numpy())

    predictions = np.concatenate(preds, axis=0)
    return predictions


def run_training():
    # 1. Setup
    seed_everything(Config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_df, val_df, test_df = load_data()

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # 3. Tokenizer & Datasets
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    train_dataset = QuestDataset(train_df, tokenizer, is_test=False)
    val_dataset = QuestDataset(val_df, tokenizer, is_test=False)
    test_dataset = QuestDataset(test_df, tokenizer, is_test=True)

    collate_fn = Collate(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # 4. Model Initialization
    print("Initializing model...")
    model = QuestModel()
    model.to(device)

    # 5. Optimizer & Scheduler
    optimizer_parameters = get_optimizer_params(
        model,
        encoder_lr=Config.lr_backbone,
        decoder_lr=Config.lr_head,
        weight_decay=Config.weight_decay,
    )

    optimizer = optim.AdamW(optimizer_parameters, eps=Config.eps, betas=Config.betas)

    num_train_steps = int(len(train_df) / Config.train_batch_size * Config.epochs)
    num_warmup_steps = int(num_train_steps * Config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    criterion = nn.BCEWithLogitsLoss()

    # 6. Training Loop
    best_score = -1.0

    print("Starting training...")
    for epoch in range(Config.epochs):
        train_loss = train_fn(
            model, train_loader, optimizer, scheduler, criterion, device, epoch
        )
        val_loss, val_preds, val_targets = valid_fn(
            model, val_loader, criterion, device
        )

        val_score = compute_spearmanr_score(val_preds, val_targets)

        print(
            f"Epoch {epoch+1}/{Config.epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Score: {val_score}"
        )

        if val_score > best_score:
            best_score = val_score
            print(f"New best score! Saving model to {Config.output_model_path}")
            torch.save(model.state_dict(), Config.output_model_path)

    print(f"Training complete. Best Val Score: {best_score}")

    # 7. Inference & Submission
    print("Generating submission...")

    # Load best model
    model.load_state_dict(torch.load(Config.output_model_path, map_location=device))
    model.to(device)

    test_preds = inference_fn(model, test_loader, device)

    # Create submission DataFrame
    submission = pd.DataFrame(test_preds, columns=Config.target_cols)
    submission.insert(0, "qa_id", test_df["qa_id"])

    # Save submission
    submission.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
