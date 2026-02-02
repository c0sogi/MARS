import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup

from library.config import Config, seed_everything
from library.utils import compute_spearman_metric
from library.data import get_dataloaders
from library.model import MultiTaskDualEncoder, get_optimizer_params


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, scaler):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    loss_fn = nn.BCEWithLogitsLoss()

    for step, batch in enumerate(dataloader):
        q_ids = batch["q_input_ids"].to(device)
        q_mask = batch["q_attention_mask"].to(device)
        a_ids = batch["a_input_ids"].to(device)
        a_mask = batch["a_attention_mask"].to(device)
        labels = batch["labels"].to(device)
        aux_labels = batch["aux_labels"].to(device)

        optimizer.zero_grad()

        with autocast():
            # Forward pass
            main_logits, aux_logits = model(q_ids, q_mask, a_ids, a_mask)

            # Compute losses
            loss_main = loss_fn(main_logits, labels)
            loss_aux = loss_fn(aux_logits, aux_labels)

            # Weighted sum
            loss = loss_main + (Config.AUX_LOSS_WEIGHT * loss_aux)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss}")
    return avg_loss


def validate(model, dataloader, device):
    """
    Validates the model on the validation set.
    """
    model.eval()
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in dataloader:
            q_ids = batch["q_input_ids"].to(device)
            q_mask = batch["q_attention_mask"].to(device)
            a_ids = batch["a_input_ids"].to(device)
            a_mask = batch["a_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # We only use main logits for validation/prediction
            main_logits, _ = model(q_ids, q_mask, a_ids, a_mask)
            preds = torch.sigmoid(main_logits)

            preds_list.append(preds.cpu())
            targets_list.append(labels.cpu())

    preds_all = torch.cat(preds_list, dim=0).numpy()
    targets_all = torch.cat(targets_list, dim=0).numpy()

    score = compute_spearman_metric(targets_all, preds_all)
    return score


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds_list = []

    with torch.no_grad():
        for batch in dataloader:
            q_ids = batch["q_input_ids"].to(device)
            q_mask = batch["q_attention_mask"].to(device)
            a_ids = batch["a_input_ids"].to(device)
            a_mask = batch["a_attention_mask"].to(device)

            main_logits, _ = model(q_ids, q_mask, a_ids, a_mask)
            preds = torch.sigmoid(main_logits)

            preds_list.append(preds.cpu())

    preds_all = torch.cat(preds_list, dim=0).numpy()
    return preds_all


def run_training(debug=False):
    """
    Main execution function for training and inference.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Using device: {device}")

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=debug
    )

    # Initialize Model
    model = MultiTaskDualEncoder()
    model.to(device)

    # Optimizer & Scheduler
    optimizer_params = get_optimizer_params(
        model, Config.LR_BACKBONE, Config.LR_HEAD, Config.WEIGHT_DECAY
    )
    optimizer = torch.optim.AdamW(optimizer_params)

    num_train_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_train_steps),
        num_training_steps=num_train_steps,
    )

    scaler = GradScaler()

    best_score = -1.0
    patience = 3
    counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, scaler
        )

        # Validate
        val_score = validate(model, val_loader, device)
        print(f"Epoch {epoch+1} | Val Spearman: {val_score}")

        # Save Best & Early Stopping
        if val_score > best_score:
            print(f"Score Improved ({best_score} -> {val_score}). Saving model...")
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            counter = 0
        else:
            counter += 1
            print(f"No improvement. EarlyStopping counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training Complete. Best Val Score: {best_score}")

    # ==========================================
    # Inference on Test Set
    # ==========================================
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)

    print("Predicting on test set...")
    test_preds = predict(model, test_loader, device)

    # Create Submission
    test_df = pd.read_csv(Config.TEST_PATH)

    submission = pd.DataFrame(test_preds, columns=Config.TARGET_COLS)
    submission.insert(0, "qa_id", test_df["qa_id"])

    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Done.")
