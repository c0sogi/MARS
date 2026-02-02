import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import seed_everything, get_device
from library.data import process_data, make_loader
from library.model import BiasAwareDeberta
from library.losses import HybridBiasLoss
from library.metrics import evaluate_predictions


def train_fn(model, data_loader, optimizer, scheduler, loss_fn, device, epoch):
    """
    Executes one training epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(data_loader)

    for step, batch in enumerate(data_loader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device)
        sample_weights = batch["sample_weight"].to(device)
        aux_identities = batch["aux_identities"].to(device)
        aux_identity_attack = batch["aux_identity_attack"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask)

        # Calculate Loss
        loss = loss_fn(
            outputs, targets, sample_weights, aux_identities, aux_identity_attack
        )

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss


def eval_fn(model, data_loader, device):
    """
    Evaluates the model on the validation set.
    Returns probability predictions.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)

            # We only use the primary toxicity head for evaluation/inference
            logits = outputs["logits"].squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()

            preds.append(probs)

    predictions = np.concatenate(preds)
    return predictions


def inference_fn(model, data_loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)

            # Primary toxicity head
            logits = outputs["logits"].squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()

            preds.append(probs)

    predictions = np.concatenate(preds)
    return predictions


def run_training(debug=Config.DEBUG):
    """
    Main orchestration function for training, evaluation, and submission.
    """
    seed_everything(Config.SEED)
    device = get_device()

    print(f"Starting training run (Debug={debug})...")

    # ==========================================
    # 1. Data Loading & Processing
    # ==========================================
    # Load processed datasets (handles caching internally)
    train_dataset = process_data(mode="train", load_cached_data=True, debug=debug)
    val_dataset = process_data(mode="val", load_cached_data=True, debug=debug)
    test_dataset = process_data(mode="test", load_cached_data=True, debug=debug)

    # Create DataLoaders
    train_loader = make_loader(
        train_dataset, batch_size=Config.TRAIN_BATCH_SIZE, mode="train"
    )
    val_loader = make_loader(
        val_dataset, batch_size=Config.VALID_BATCH_SIZE, mode="val"
    )
    test_loader = make_loader(
        test_dataset, batch_size=Config.VALID_BATCH_SIZE, mode="test"
    )

    # Load Validation DataFrame for Metric Calculation
    # We need the metadata to align predictions with identity columns
    val_df = pd.read_csv(Config.VAL_PATH)
    if debug:
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)

    # ==========================================
    # 2. Model & Optimizer Setup
    # ==========================================
    model = BiasAwareDeberta()
    model.to(device)

    # Optimizer (AdamW)
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler (Linear with Warmup)
    num_training_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Loss Function
    loss_fn = HybridBiasLoss()
    loss_fn.to(device)

    # ==========================================
    # 3. Training Loop
    # ==========================================
    best_score = -float("inf")
    patience_counter = 0

    print(f"Training for {Config.EPOCHS} epochs on device: {device}")

    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")

        # Train
        train_loss = train_fn(
            model, train_loader, optimizer, scheduler, loss_fn, device, epoch
        )
        print(f"  Average Train Loss: {train_loss:.6f}")

        # Validate
        val_preds = eval_fn(model, val_loader, device)

        # Calculate Metrics
        # Assign predictions to the dataframe copy for metric calculation
        val_df["prediction"] = val_preds

        # Evaluate using the competition metric
        score, _ = evaluate_predictions(val_df, prediction_col="prediction")

        # Early Stopping & Checkpointing
        if score > best_score:
            print(
                f"  Score improved from {best_score:.16f} to {score:.16f}. Saving model..."
            )
            best_score = score
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            patience_counter = 0
        else:
            print(f"  Score did not improve (Best: {best_score:.16f}).")
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("  Early stopping triggered.")
                break

        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

    # ==========================================
    # 4. Inference & Submission
    # ==========================================
    print("\nLoading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device))
    model.to(device)

    print("Generating test predictions...")
    test_preds = inference_fn(model, test_loader, device)

    # Prepare Submission
    # Load IDs from test metadata
    test_df = pd.read_csv(Config.TEST_PATH)

    # If debug, we only predicted a subset, so we slice the df
    if debug:
        test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)

    submission = pd.DataFrame({"id": test_df["id"], "prediction": test_preds})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
