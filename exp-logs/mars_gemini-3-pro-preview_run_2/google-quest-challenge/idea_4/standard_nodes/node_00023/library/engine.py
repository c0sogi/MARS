import os
import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from library.config import Config, set_seed
from library.model import CausalDebertaSiamese
from library.dataset import get_dataloader
from library.utils import compute_spearman_metric


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for step, batch in enumerate(dataloader):
        # Move batch to device
        q_input_ids = batch["q_input_ids"].to(device)
        q_attention_mask = batch["q_attention_mask"].to(device)
        a_input_ids = batch["a_input_ids"].to(device)
        a_attention_mask = batch["a_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        batch_size = q_input_ids.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(q_input_ids, q_attention_mask, a_input_ids, a_attention_mask)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer and Scheduler steps
        optimizer.step()
        scheduler.step()

        # Update stats
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Validates the model on the validation set.
    Returns average loss and Spearman correlation score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in dataloader:
            q_input_ids = batch["q_input_ids"].to(device)
            q_attention_mask = batch["q_attention_mask"].to(device)
            a_input_ids = batch["a_input_ids"].to(device)
            a_attention_mask = batch["a_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            batch_size = q_input_ids.size(0)

            # Forward pass
            logits = model(q_input_ids, q_attention_mask, a_input_ids, a_attention_mask)

            # Compute loss
            loss = criterion(logits, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for predictions
            probs = torch.sigmoid(logits)

            preds_list.append(probs.cpu().numpy())
            targets_list.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate predictions and targets
    predictions = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    # Compute metric
    score = compute_spearman_metric(predictions, targets)

    return epoch_loss, score


def run_training():
    """
    Main driver for the training pipeline.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE

    print(f"Using device: {device}")

    # Ensure output directory exists
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # 2. Data Preparation
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    train_loader = get_dataloader("train", tokenizer, load_cached_data=True)
    val_loader = get_dataloader("val", tokenizer, load_cached_data=True)

    # 3. Model Initialization
    model = CausalDebertaSiamese()
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer_grouped_parameters = Config.get_optimizer_params(model)
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, eps=Config.EPS, betas=Config.BETAS
    )

    criterion = nn.BCEWithLogitsLoss()

    num_train_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(num_train_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # 5. Training Loop
    best_score = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    save_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, epoch
        )

        # Validate
        val_loss, val_score = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Spearman Score: {val_score}"
        )

        # Checkpointing
        if val_score > best_score:
            print(
                f"Validation score improved ({best_score} --> {val_score}). Saving model..."
            )
            best_score = val_score
            # Deepcopy to avoid reference issues
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)

    print(f"Training complete. Best Validation Spearman Score: {best_score}")

    # Load best weights for inference
    model.load_state_dict(best_model_wts)

    return model, tokenizer


def predict_and_submit(model, tokenizer):
    """
    Generates predictions for the test set and creates the submission file.
    """
    print("Generating predictions for test set...")
    device = Config.DEVICE
    model.eval()

    test_loader = get_dataloader("test", tokenizer, load_cached_data=True)

    preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            q_input_ids = batch["q_input_ids"].to(device)
            q_attention_mask = batch["q_attention_mask"].to(device)
            a_input_ids = batch["a_input_ids"].to(device)
            a_attention_mask = batch["a_attention_mask"].to(device)

            # Forward pass
            logits = model(q_input_ids, q_attention_mask, a_input_ids, a_attention_mask)

            # Apply sigmoid
            probs = torch.sigmoid(logits)
            preds_list.append(probs.cpu().numpy())

    # Concatenate all predictions
    all_preds = np.concatenate(preds_list, axis=0)

    # Load test metadata to get qa_ids
    test_df = pd.read_csv(Config.TEST_PATH)

    # Verify shape
    if all_preds.shape[0] != len(test_df):
        print(
            f"Warning: Prediction count {all_preds.shape[0]} != Test ID count {len(test_df)}"
        )

    # Create submission DataFrame
    submission = pd.DataFrame(all_preds, columns=Config.TARGET_COLS)
    submission.insert(0, "qa_id", test_df["qa_id"])

    # Save submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    # Run training
    model, tokenizer = run_training()

    # Run inference
    predict_and_submit(model, tokenizer)


# Note: The if __name__ == "__main__": block is omitted as per instructions.
