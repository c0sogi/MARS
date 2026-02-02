import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import AverageMeter, compute_spearmanr
from library.model import SymmetricDualEncoder
from library.dataset import get_dataloaders


def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)


def train_fn(dataloader, model, optimizer, device, scheduler, epoch):
    model.train()

    # Head Warmup Strategy: Freeze backbones during the first epoch (epoch 0)
    if epoch == 0:
        model.q_encoder.requires_grad_(False)
        model.a_encoder.requires_grad_(False)
    else:
        model.q_encoder.requires_grad_(True)
        model.a_encoder.requires_grad_(True)

    loss_score = AverageMeter()

    # Reset gradients
    optimizer.zero_grad()

    for step, data in enumerate(dataloader):
        # Move inputs to device
        input_ids_q = data["input_ids_q"].to(device)
        attention_mask_q = data["attention_mask_q"].to(device)
        input_ids_a = data["input_ids_a"].to(device)
        attention_mask_a = data["attention_mask_a"].to(device)
        targets = data["labels"].to(device)

        # Forward pass
        outputs = model(input_ids_q, attention_mask_q, input_ids_a, attention_mask_a)

        # Compute loss
        loss = loss_fn(outputs, targets)

        # Gradient Accumulation
        loss = loss / Config.ACCUMULATION_STEPS
        loss.backward()

        if (step + 1) % Config.ACCUMULATION_STEPS == 0:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        loss_score.update(loss.item() * Config.ACCUMULATION_STEPS, targets.size(0))

    return loss_score.avg


def eval_fn(dataloader, model, device):
    model.eval()
    loss_score = AverageMeter()
    preds = []
    targets_list = []

    with torch.no_grad():
        for data in dataloader:
            input_ids_q = data["input_ids_q"].to(device)
            attention_mask_q = data["attention_mask_q"].to(device)
            input_ids_a = data["input_ids_a"].to(device)
            attention_mask_a = data["attention_mask_a"].to(device)
            targets = data["labels"].to(device)

            outputs = model(
                input_ids_q, attention_mask_q, input_ids_a, attention_mask_a
            )

            loss = loss_fn(outputs, targets)
            loss_score.update(loss.item(), targets.size(0))

            # Apply Sigmoid for predictions
            preds.append(torch.sigmoid(outputs).cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    preds = np.concatenate(preds)
    targets_list = np.concatenate(targets_list)

    score = compute_spearmanr(targets_list, preds)
    return loss_score.avg, score, preds


def inference_fn(dataloader, model, device):
    model.eval()
    preds = []
    qa_ids = []

    with torch.no_grad():
        for data in dataloader:
            input_ids_q = data["input_ids_q"].to(device)
            attention_mask_q = data["attention_mask_q"].to(device)
            input_ids_a = data["input_ids_a"].to(device)
            attention_mask_a = data["attention_mask_a"].to(device)

            outputs = model(
                input_ids_q, attention_mask_q, input_ids_a, attention_mask_a
            )

            preds.append(torch.sigmoid(outputs).cpu().numpy())
            qa_ids.extend(data["qa_id"])

    preds = np.concatenate(preds)
    return preds, qa_ids


def run_training(debug=False):
    device = torch.device(Config.DEVICE)

    # 1. Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=debug, load_cached_data=True
    )

    # 2. Model Initialization
    model = SymmetricDualEncoder()
    model.to(device)

    # 3. Optimizer with Differential Learning Rates
    # Group parameters
    backbone_params = list(model.q_encoder.parameters()) + list(
        model.a_encoder.parameters()
    )

    # Head params include the bridge, layer norm, head projection, and final projection
    head_params = (
        list(model.alignment_bridge.parameters())
        + list(model.layer_norm.parameters())
        + list(model.head_proj.parameters())
        + list(model.final_proj.parameters())
    )

    optimizer_parameters = [
        {"params": backbone_params, "lr": Config.LR_BACKBONE},
        {"params": head_params, "lr": Config.LR_HEAD},
    ]

    optimizer = torch.optim.AdamW(optimizer_parameters)

    # 4. Phantom Scheduling
    # Calculate total steps based on PHANTOM_EPOCHS (7)
    # But we only train for ACTUAL_EPOCHS (3)
    num_train_steps = int(
        len(train_loader) / Config.ACCUMULATION_STEPS * Config.PHANTOM_EPOCHS
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_train_steps),
        num_training_steps=num_train_steps,
    )

    best_score = -1.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(
        f"Starting training for {Config.ACTUAL_EPOCHS} epochs (Phantom Schedule: {Config.PHANTOM_EPOCHS} epochs)..."
    )

    for epoch in range(Config.ACTUAL_EPOCHS):
        # Train
        train_loss = train_fn(train_loader, model, optimizer, device, scheduler, epoch)

        # Evaluate
        val_loss, val_score, _ = eval_fn(val_loader, model, device)

        print(f"Epoch {epoch+1}/{Config.ACTUAL_EPOCHS}")
        print(f"Train Loss: {train_loss:.16f}")
        print(f"Val Loss:   {val_loss:.16f}")
        print(f"Val Score:  {val_score:.16f}")

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"New best score! Model saved to {best_model_path}")

    # 5. Inference & Submission
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_preds, test_ids = inference_fn(test_loader, model, device)

    # Create Submission DataFrame
    submission = pd.DataFrame(test_preds, columns=Config.TARGET_COLS)
    submission.insert(0, "qa_id", test_ids)

    # Ensure submission directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    return best_score
