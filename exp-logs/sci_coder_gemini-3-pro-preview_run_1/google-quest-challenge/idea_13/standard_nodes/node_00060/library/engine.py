import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import seed_everything, compute_spearmanr
from library.dataset import get_dataloaders
from library.model import DualBranchDistilRoBERTa


def train_fn(model, dataloader, optimizer, device, scheduler=None):
    """
    Performs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    count = 0

    for batch in dataloader:
        # Move inputs to device
        question_input_ids = batch["question_input_ids"].to(device)
        question_attention_mask = batch["question_attention_mask"].to(device)
        answer_input_ids = batch["answer_input_ids"].to(device)
        answer_attention_mask = batch["answer_attention_mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(
            question_input_ids,
            question_attention_mask,
            answer_input_ids,
            answer_attention_mask,
        )

        # Compute loss
        loss = nn.BCEWithLogitsLoss()(logits, targets)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        count += 1

    return total_loss / count


def eval_fn(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0
    count = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            question_input_ids = batch["question_input_ids"].to(device)
            question_attention_mask = batch["question_attention_mask"].to(device)
            answer_input_ids = batch["answer_input_ids"].to(device)
            answer_attention_mask = batch["answer_attention_mask"].to(device)
            targets = batch["targets"].to(device)

            logits = model(
                question_input_ids,
                question_attention_mask,
                answer_input_ids,
                answer_attention_mask,
            )

            loss = nn.BCEWithLogitsLoss()(logits, targets)
            total_loss += loss.item()
            count += 1

            # Apply sigmoid for predictions
            preds = torch.sigmoid(logits)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / count

    # Compute Spearman Correlation
    if len(all_preds) > 0:
        predictions = np.concatenate(all_preds, axis=0)
        ground_truth = np.concatenate(all_targets, axis=0)
        score = compute_spearmanr(predictions, ground_truth)
    else:
        score = 0.0

    return avg_loss, score


def get_optimizer_params(model):
    """
    Configures differential learning rates and weight decay groups.
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    head_params = []
    backbone_params = []

    for n, p in param_optimizer:
        # 'head' and 'layer_norm' belong to the fusion/classifier part
        if "head" in n or "layer_norm" in n:
            head_params.append((n, p))
        # 'backbone' belongs to the pre-trained transformer
        elif "backbone" in n:
            backbone_params.append((n, p))
        else:
            # Fallback to head if unknown
            head_params.append((n, p))

    optimizer_grouped_parameters = [
        # Group 1: Head - Decay
        {
            "params": [
                p for n, p in head_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": Config.WEIGHT_DECAY,
            "lr": Config.LR_HEAD,
        },
        # Group 2: Head - No Decay
        {
            "params": [p for n, p in head_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": Config.LR_HEAD,
        },
        # Group 3: Backbone - Decay
        {
            "params": [
                p for n, p in backbone_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": Config.WEIGHT_DECAY,
            "lr": Config.LR_BACKBONE,
        },
        # Group 4: Backbone - No Decay
        {
            "params": [
                p for n, p in backbone_params if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": Config.LR_BACKBONE,
        },
    ]
    return optimizer_grouped_parameters


def run_training():
    """
    Main execution function for training and inference.
    """
    seed_everything(Config.SEED)

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Initialize Model
    device = Config.DEVICE
    model = DualBranchDistilRoBERTa()
    model.to(device)

    # Initialize Optimizer
    optimizer_params = get_optimizer_params(model)
    optimizer = torch.optim.AdamW(optimizer_params)

    # Initialize Scheduler (Linear Decay)
    num_training_steps = len(train_loader) * Config.EPOCHS
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps
    )

    best_score = -np.inf

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        # --- Freeze/Unfreeze Logic ---
        if epoch == 0:
            print("Epoch 1: Freezing backbone layers.")
            for param in model.backbone.parameters():
                param.requires_grad = False
        elif epoch == 1:
            print("Epoch 2: Unfreezing backbone layers.")
            for param in model.backbone.parameters():
                param.requires_grad = True

        # --- Training ---
        train_loss = train_fn(model, train_loader, optimizer, device, scheduler)

        # --- Validation ---
        val_loss, val_score = eval_fn(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Spearman: {val_score:.6f}"
        )

        # --- Save Best Model ---
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with score: {best_score:.6f}")

    # ==========================================
    # Inference
    # ==========================================
    print("\nLoading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.to(device)
    model.eval()

    all_preds = []
    all_qa_ids = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for batch in test_loader:
            question_input_ids = batch["question_input_ids"].to(device)
            question_attention_mask = batch["question_attention_mask"].to(device)
            answer_input_ids = batch["answer_input_ids"].to(device)
            answer_attention_mask = batch["answer_attention_mask"].to(device)
            qa_ids = batch["qa_ids"]

            logits = model(
                question_input_ids,
                question_attention_mask,
                answer_input_ids,
                answer_attention_mask,
            )

            preds = torch.sigmoid(logits).cpu().numpy()

            all_preds.append(preds)
            all_qa_ids.extend(qa_ids)

    if all_preds:
        final_preds = np.vstack(all_preds)

        # Create submission DataFrame
        submission = pd.DataFrame(final_preds, columns=Config.TARGET_COLS)
        submission.insert(0, "qa_id", all_qa_ids)

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print("Error: No predictions generated.")
