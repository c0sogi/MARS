import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import seed_everything, compute_spearman_metric
from library.dataset import get_dataloaders
from library.model import ContextualDualEncoder


def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        # Move batch to device
        q_input_ids = batch["q_input_ids"].to(device)
        q_attention_mask = batch["q_attention_mask"].to(device)
        a_input_ids = batch["a_input_ids"].to(device)
        a_attention_mask = batch["a_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(q_input_ids, q_attention_mask, a_input_ids, a_attention_mask)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Optimization step
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate_epoch(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in loader:
            q_input_ids = batch["q_input_ids"].to(device)
            q_attention_mask = batch["q_attention_mask"].to(device)
            a_input_ids = batch["a_input_ids"].to(device)
            a_attention_mask = batch["a_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(q_input_ids, q_attention_mask, a_input_ids, a_attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            # Apply sigmoid to get probabilities in [0, 1]
            probs = torch.sigmoid(logits)

            preds_list.append(probs.cpu().numpy())
            targets_list.append(labels.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    spearman_score = compute_spearman_metric(preds, targets)

    return total_loss / len(loader), spearman_score


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds_list = []

    with torch.no_grad():
        for batch in loader:
            q_input_ids = batch["q_input_ids"].to(device)
            q_attention_mask = batch["q_attention_mask"].to(device)
            a_input_ids = batch["a_input_ids"].to(device)
            a_attention_mask = batch["a_attention_mask"].to(device)

            logits = model(q_input_ids, q_attention_mask, a_input_ids, a_attention_mask)
            probs = torch.sigmoid(logits)
            preds_list.append(probs.cpu().numpy())

    return np.concatenate(preds_list, axis=0)


def main(debug=False):
    """
    Main training and inference pipeline.
    """
    seed_everything(Config.SEED)

    print(f"Starting run (Debug={debug})...")

    # 1. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=debug, load_cached_data=True
    )

    # 2. Model Initialization
    model = ContextualDualEncoder()
    model.to(Config.DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    best_score = -1.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # ==========================================
    # Phase 1: Warmup (Frozen Backbone)
    # ==========================================
    print("\n--- Phase 1: Warmup (Frozen Backbone) ---")
    model.freeze_backbone()

    # Optimizer for head only
    optimizer_warmup = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LR_HEAD,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Train for 1 epoch
    train_loss = train_epoch(
        model, train_loader, optimizer_warmup, None, criterion, Config.DEVICE
    )
    val_loss, val_score = validate_epoch(model, val_loader, criterion, Config.DEVICE)

    print(
        f"Epoch 1 (Warmup) - Train Loss: {train_loss} - Val Loss: {val_loss} - Val Spearman: {val_score}"
    )

    # Save initial best model
    best_score = val_score
    torch.save(model.state_dict(), best_model_path)

    # ==========================================
    # Phase 2: Fine-tuning (Unfrozen Backbone)
    # ==========================================
    print("\n--- Phase 2: Fine-tuning (Unfrozen Backbone) ---")
    model.unfreeze_backbone()

    # Parameter groups with differential learning rates
    optimizer_grouped_parameters = [
        {"params": model.backbone.parameters(), "lr": Config.LR_BACKBONE},
        {"params": model.head.parameters(), "lr": Config.LR_HEAD},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=Config.WEIGHT_DECAY)

    # Scheduler
    epochs_finetune = Config.EPOCHS - 1
    if epochs_finetune > 0:
        num_training_steps = len(train_loader) * epochs_finetune
        num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        for epoch in range(epochs_finetune):
            current_epoch = epoch + 2  # 1-based index, continuing from warmup

            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, criterion, Config.DEVICE
            )
            val_loss, val_score = validate_epoch(
                model, val_loader, criterion, Config.DEVICE
            )

            print(
                f"Epoch {current_epoch} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val Spearman: {val_score}"
            )

            # Save if improvement
            if val_score > best_score:
                print(f"Score improved from {best_score} to {val_score}. Saving model.")
                best_score = val_score
                torch.save(model.state_dict(), best_model_path)

    # ==========================================
    # Inference
    # ==========================================
    print("\n--- Inference ---")
    print(f"Loading best model from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))

    print("Generating predictions on test set...")
    predictions = predict(model, test_loader, Config.DEVICE)

    # Load test metadata to align qa_ids
    test_df = pd.read_csv(Config.TEST_PATH)
    if debug:
        test_df = test_df.iloc[:100]

    # Create submission dataframe
    submission = pd.DataFrame(predictions, columns=Config.TARGET_COLS)
    submission.insert(0, "qa_id", test_df["qa_id"])

    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Training and inference completed successfully.")
