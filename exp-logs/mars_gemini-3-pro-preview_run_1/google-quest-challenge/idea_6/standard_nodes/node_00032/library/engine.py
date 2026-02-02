import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.model import QuestModel
from library.utils import compute_spearmanr


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_optimizer_params(model):
    """
    Sets up parameter groups for the optimizer:
    1. Differential Learning Rates (Backbone vs Head)
    2. Weight Decay exclusion for Bias and LayerNorm
    """
    # Define parameter groups
    optimizer_parameters = []

    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

    backbone_decay = []
    backbone_no_decay = []
    head_decay = []
    head_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if parameter belongs to backbone
        is_backbone = "backbone" in name

        # Check if parameter should have weight decay
        if any(nd in name for nd in no_decay):
            has_decay = False
        else:
            has_decay = True

        if is_backbone:
            if has_decay:
                backbone_decay.append(param)
            else:
                backbone_no_decay.append(param)
        else:
            if has_decay:
                head_decay.append(param)
            else:
                head_no_decay.append(param)

    optimizer_parameters = [
        {
            "params": backbone_decay,
            "lr": Config.backbone_lr,
            "weight_decay": Config.weight_decay,
        },
        {
            "params": backbone_no_decay,
            "lr": Config.backbone_lr,
            "weight_decay": 0.0,
        },
        {
            "params": head_decay,
            "lr": Config.head_lr,
            "weight_decay": Config.weight_decay,
        },
        {
            "params": head_no_decay,
            "lr": Config.head_lr,
            "weight_decay": 0.0,
        },
    ]

    return optimizer_parameters


def train_fn(train_loader, model, criterion, optimizer, scheduler, epoch, device):
    model.train()
    losses = AverageMeter()
    start_time = time.time()

    for step, batch in enumerate(train_loader):
        # Unpack batch
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        question_mask = batch["question_mask"].to(device)
        answer_mask = batch["answer_mask"].to(device)
        labels = batch["labels"].to(device)

        batch_size = labels.size(0)

        # Forward
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            question_mask=question_mask,
            answer_mask=answer_mask,
        )

        loss = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), batch_size)

        if (step + 1) % Config.print_freq == 0 or (step + 1) == len(train_loader):
            print(
                f"Epoch: [{epoch + 1}][{step + 1}/{len(train_loader)}] "
                f"Elapsed: {time.time() - start_time:.1f}s "
                f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    return losses.avg


def eval_fn(val_loader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    preds_list = []
    labels_list = []

    start_time = time.time()

    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            question_mask = batch["question_mask"].to(device)
            answer_mask = batch["answer_mask"].to(device)
            labels = batch["labels"].to(device)

            batch_size = labels.size(0)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                question_mask=question_mask,
                answer_mask=answer_mask,
            )

            loss = criterion(logits, labels)
            losses.update(loss.item(), batch_size)

            # Apply sigmoid for predictions
            preds = torch.sigmoid(logits)

            preds_list.append(preds.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    predictions = np.concatenate(preds_list)
    targets = np.concatenate(labels_list)

    score = compute_spearmanr(predictions, targets)

    print(
        f"Validation: Loss {losses.avg:.4f} | Score {score} | Time {time.time() - start_time:.1f}s"
    )

    return losses.avg, score


def predict_fn(test_loader, model, device):
    model.eval()
    preds_list = []
    qa_ids_list = []

    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            question_mask = batch["question_mask"].to(device)
            answer_mask = batch["answer_mask"].to(device)
            qa_ids = batch["qa_ids"]

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                question_mask=question_mask,
                answer_mask=answer_mask,
            )

            preds = torch.sigmoid(logits)

            preds_list.append(preds.cpu().numpy())
            qa_ids_list.extend(qa_ids.numpy())

    predictions = np.concatenate(preds_list)
    return qa_ids_list, predictions


def run_training(train_loader, val_loader, test_loader):
    device = Config.device

    # Initialize Model
    print(f"Initializing model: {Config.model_name}")
    model = QuestModel()
    model.to(device)

    # Optimizer
    optimizer_parameters = get_optimizer_params(model)
    optimizer = torch.optim.AdamW(
        optimizer_parameters, eps=Config.eps, betas=Config.betas
    )

    # Scheduler
    num_train_steps = len(train_loader) * Config.epochs
    num_warmup_steps = int(num_train_steps * Config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    best_score = -1.0

    # Training Loop
    for epoch in range(Config.epochs):
        print(f"\nEpoch {epoch + 1}/{Config.epochs}")

        train_loss = train_fn(
            train_loader, model, criterion, optimizer, scheduler, epoch, device
        )
        val_loss, val_score = eval_fn(val_loader, model, criterion, device)

        print(
            f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Score: {val_score}"
        )

        # Save Best Model
        if val_score > best_score:
            print(f"Score Improved ({best_score} -> {val_score}). Saving model...")
            best_score = val_score
            torch.save(model.state_dict(), Config.model_save_path)
        else:
            print(f"Score did not improve from {best_score}.")

    # Load Best Model for Inference
    print("\nLoading best model for inference...")
    model.load_state_dict(torch.load(Config.model_save_path, map_location=device))

    # Predict on Test
    print("Predicting on test set...")
    qa_ids, predictions = predict_fn(test_loader, model, device)

    # Create Submission
    sample_sub = pd.read_csv(Config.sample_submission_path)
    target_cols = [col for col in sample_sub.columns if col != Config.qa_id_col]

    submission_df = pd.DataFrame(predictions, columns=target_cols)
    submission_df.insert(0, Config.qa_id_col, qa_ids)

    # Ensure qa_id is int
    submission_df[Config.qa_id_col] = submission_df[Config.qa_id_col].astype(int)

    submission_df.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
