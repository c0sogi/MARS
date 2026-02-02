import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from library.utils import get_score
from library.awp import AWP


def train_fn(train_loader, model, optimizer, scheduler, device, epoch, cfg):
    """
    Executes one training epoch.
    Includes Adversarial Weight Perturbation (AWP) if configured.
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    losses = []

    # Initialize AWP if enabled and start epoch is reached
    awp = None
    if cfg.use_awp and epoch >= cfg.awp_start_epoch:
        awp = AWP(
            model,
            optimizer,
            adv_lr=cfg.awp_lr,
            adv_eps=cfg.awp_eps,
            start_epoch=cfg.awp_start_epoch,
            scaler=scaler,
        )

    for step, data in enumerate(train_loader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["labels"].to(device)

        batch_size = input_ids.size(0)

        # Standard Forward Pass
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]

        # Standard Backward Pass
        scaler.scale(loss).backward()

        # Adversarial Weight Perturbation Step
        if awp is not None:
            awp.attack()  # Perturb weights based on gradients

            # Forward pass with perturbed weights
            with torch.cuda.amp.autocast(enabled=True):
                outputs_adv = model(input_ids, attention_mask, labels)
                loss_adv = outputs_adv["loss"]

            # Backward pass with adversarial loss
            scaler.scale(loss_adv).backward()

            awp._restore()  # Restore original weights

        # Optimizer Step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())

    return np.mean(losses)


def valid_fn(val_loader, model, device, cfg):
    """
    Evaluates the model on the validation set.
    Returns loss, ROC AUC score, and predictions.
    """
    model.eval()
    preds = []
    targets = []
    losses = []

    with torch.no_grad():
        for data in val_loader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            labels = data["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs["loss"]
                logits = outputs["logits"]

            losses.append(loss.item())
            # Apply sigmoid to get probabilities
            preds.append(torch.sigmoid(logits).float().detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    loss = np.mean(losses)

    # Calculate ROC AUC
    score = get_score(targets, preds)

    return loss, score, preds


def inference_fn(test_loader, model, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for data in test_loader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)

            with torch.cuda.amp.autocast(enabled=True):
                # Pass labels=None for inference
                outputs = model(input_ids, attention_mask, labels=None)
                logits = outputs["logits"]

            preds.append(torch.sigmoid(logits).float().detach().cpu().numpy())

    preds = np.concatenate(preds)
    return preds


def run_training(cfg, model, train_loader, val_loader, optimizer, scheduler, device):
    """
    Orchestrates the training process across epochs with Early Stopping.
    """
    best_score = -np.inf
    patience = 3  # Number of epochs to wait for improvement
    counter = 0

    for epoch in range(cfg.epochs):
        print(f"Epoch {epoch + 1}/{cfg.epochs}")

        # Train
        train_loss = train_fn(
            train_loader, model, optimizer, scheduler, device, epoch, cfg
        )

        # Validate
        val_loss, val_score, _ = valid_fn(val_loader, model, device, cfg)

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f} | Val AUC: {val_score:.6f}")

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            print(f"Score Improved. Saving model to {cfg.model_save_path}")
            torch.save(model.state_dict(), cfg.model_save_path)
            counter = 0
        else:
            counter += 1
            print(f"No improvement. Early stopping counter: {counter}/{patience}")
            if counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training Complete. Best Val AUC: {best_score:.6f}")


def generate_submission(cfg, model, test_loader, device):
    """
    Generates the final submission CSV using the best trained model.
    """
    print("Generating predictions on test set...")

    # Load the best model weights
    if os.path.exists(cfg.model_save_path):
        state_dict = torch.load(cfg.model_save_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Loaded best model weights.")
    else:
        print("Warning: Model weight file not found. Using current weights.")

    # Generate predictions
    preds = inference_fn(test_loader, model, device)

    # Load sample submission to get correct IDs and structure
    sample_sub = pd.read_csv(cfg.sample_submission_path)

    # Ensure prediction shape matches
    if preds.shape[0] != len(sample_sub):
        print(
            f"Warning: Prediction count {preds.shape[0]} does not match sample submission {len(sample_sub)}"
        )
        if cfg.debug:
            print("Debug mode: Slicing sample submission to match prediction length.")
            sample_sub = sample_sub.iloc[: preds.shape[0]]

    # Fill submission DataFrame
    for i, col in enumerate(cfg.target_cols):
        sample_sub[col] = preds[:, i]

    # Save to file
    sample_sub.to_csv(cfg.submission_path, index=False)
    print(f"Submission saved to {cfg.submission_path}")
