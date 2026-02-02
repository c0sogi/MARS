import os
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from library.config import Config
from library.utils import seed_everything, compute_spearman_metric, save_checkpoint
from library.dataset import get_dataloaders
from library.model import HybridDeberta


def train_fn(dataloader, model, criterion, optimizer, scheduler, device, epoch):
    """
    Training loop for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    start_time = time.time()

    for step, batch in enumerate(dataloader):
        # Move batch to device
        view1_input_ids = batch["view1_input_ids"].to(device)
        view1_attention_mask = batch["view1_attention_mask"].to(device)
        view2_input_ids = batch["view2_input_ids"].to(device)
        view2_attention_mask = batch["view2_attention_mask"].to(device)
        view2_q_mask = batch["view2_q_mask"].to(device)
        view2_a_mask = batch["view2_a_mask"].to(device)
        labels = batch["labels"].to(device)

        # Optional: token_type_ids if available (DeBERTa v3 doesn't strictly need them if masks are good, but handled in model)
        view2_token_type_ids = None
        if "view2_token_type_ids" in batch:
            view2_token_type_ids = batch["view2_token_type_ids"].to(device)

        batch_size = view1_input_ids.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(
            view1_input_ids=view1_input_ids,
            view1_attention_mask=view1_attention_mask,
            view2_input_ids=view2_input_ids,
            view2_attention_mask=view2_attention_mask,
            view2_q_mask=view2_q_mask,
            view2_a_mask=view2_a_mask,
            view2_token_type_ids=view2_token_type_ids,
        )

        # Loss calculation
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        # Optimizer step
        optimizer.step()

        # Scheduler step (CosineAnnealingWarmRestarts usually steps per epoch, but can step per batch)
        # Here we follow common practice for transformers: step scheduler if it's batch-based,
        # but Config suggests WarmRestarts which is often epoch-based.
        # However, standard PyTorch WarmRestarts is usually called at end of epoch.
        # We will update it at the end of the epoch in the main loop.

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    elapsed = time.time() - start_time

    print(
        f"Epoch {epoch + 1}/{Config.epochs} | Train Loss: {epoch_loss:.6f} | Time: {elapsed:.2f}s"
    )
    return epoch_loss


def eval_fn(dataloader, model, criterion, device):
    """
    Validation loop.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in dataloader:
            view1_input_ids = batch["view1_input_ids"].to(device)
            view1_attention_mask = batch["view1_attention_mask"].to(device)
            view2_input_ids = batch["view2_input_ids"].to(device)
            view2_attention_mask = batch["view2_attention_mask"].to(device)
            view2_q_mask = batch["view2_q_mask"].to(device)
            view2_a_mask = batch["view2_a_mask"].to(device)
            labels = batch["labels"].to(device)

            view2_token_type_ids = None
            if "view2_token_type_ids" in batch:
                view2_token_type_ids = batch["view2_token_type_ids"].to(device)

            batch_size = view1_input_ids.size(0)

            logits = model(
                view1_input_ids=view1_input_ids,
                view1_attention_mask=view1_attention_mask,
                view2_input_ids=view2_input_ids,
                view2_attention_mask=view2_attention_mask,
                view2_q_mask=view2_q_mask,
                view2_a_mask=view2_a_mask,
                view2_token_type_ids=view2_token_type_ids,
            )

            loss = criterion(logits, labels)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for predictions
            probs = torch.sigmoid(logits)

            preds_list.append(probs.cpu().numpy())
            targets_list.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    predictions = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    spearman_score = compute_spearman_metric(predictions, targets)

    return epoch_loss, spearman_score


def predict_fn(dataloader, model, device):
    """
    Inference loop for test set.
    """
    model.eval()
    preds_list = []

    with torch.no_grad():
        for batch in dataloader:
            view1_input_ids = batch["view1_input_ids"].to(device)
            view1_attention_mask = batch["view1_attention_mask"].to(device)
            view2_input_ids = batch["view2_input_ids"].to(device)
            view2_attention_mask = batch["view2_attention_mask"].to(device)
            view2_q_mask = batch["view2_q_mask"].to(device)
            view2_a_mask = batch["view2_a_mask"].to(device)

            view2_token_type_ids = None
            if "view2_token_type_ids" in batch:
                view2_token_type_ids = batch["view2_token_type_ids"].to(device)

            logits = model(
                view1_input_ids=view1_input_ids,
                view1_attention_mask=view1_attention_mask,
                view2_input_ids=view2_input_ids,
                view2_attention_mask=view2_attention_mask,
                view2_q_mask=view2_q_mask,
                view2_a_mask=view2_a_mask,
                view2_token_type_ids=view2_token_type_ids,
            )

            probs = torch.sigmoid(logits)
            preds_list.append(probs.cpu().numpy())

    predictions = np.concatenate(preds_list, axis=0)
    return predictions


def run_training(load_cached_data=True):
    """
    Main function to run training, validation, and submission generation.
    """
    seed_everything(Config.seed)

    # 1. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 2. Model Initialization
    print("Initializing Model...")
    device = Config.device
    model = HybridDeberta()
    model.to(device)

    # 3. Optimizer with Differential Learning Rates
    optimizer_grouped_parameters = [
        {"params": model.backbone.parameters(), "lr": Config.lr_backbone},
        {"params": model.head_intrinsic.parameters(), "lr": Config.lr_head},
        {"params": model.head_relational.parameters(), "lr": Config.lr_head},
    ]

    optimizer = optim.AdamW(
        optimizer_grouped_parameters, weight_decay=Config.weight_decay
    )

    # 4. Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.T_0, T_mult=1, eta_min=Config.min_lr
    )

    # 5. Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # 6. Training Loop
    best_score = -1.0
    best_model_path = Config.model_save_path

    # Early stopping parameters
    patience = 3
    counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(Config.epochs):
        # Train
        train_loss = train_fn(
            train_loader, model, criterion, optimizer, scheduler, device, epoch
        )

        # Validate
        val_loss, val_score = eval_fn(val_loader, model, criterion, device)

        # Step scheduler
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{Config.epochs} | Val Loss: {val_loss:.6f} | Val Spearman: {val_score}"
        )

        # Save Best Model
        if val_score > best_score:
            print(
                f"Validation Score Improved ({best_score} -> {val_score}). Saving model..."
            )
            best_score = val_score
            save_checkpoint(model, best_model_path)
            counter = 0  # Reset early stopping counter
        else:
            counter += 1
            print(f"No improvement. Early stopping counter: {counter}/{patience}")

        if counter >= patience:
            print("Early stopping triggered.")
            break

    # 7. Prediction and Submission
    print("\nLoading best model for inference...")
    # Re-initialize model structure and load weights
    best_model = HybridDeberta()
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    best_model.to(device)

    print("Generating predictions on test set...")
    test_preds = predict_fn(test_loader, best_model, device)

    # Prepare Submission DataFrame
    # Load test metadata to get qa_ids
    test_df = pd.read_csv(Config.test_path)
    qa_ids = test_df["qa_id"].values

    # Ensure prediction shape matches
    if test_preds.shape[0] != len(qa_ids):
        print(
            f"Warning: Number of predictions ({test_preds.shape[0]}) does not match number of test rows ({len(qa_ids)})."
        )

    submission_df = pd.DataFrame(test_preds, columns=Config.target_cols)
    submission_df.insert(0, "qa_id", qa_ids)

    # Save submission
    print(f"Saving submission to {Config.submission_path}...")
    submission_df.to_csv(Config.submission_path, index=False)
    print("Submission saved successfully.")
