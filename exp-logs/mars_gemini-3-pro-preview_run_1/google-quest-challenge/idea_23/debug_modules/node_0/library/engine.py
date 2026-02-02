import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from scipy.stats import spearmanr
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from library.config import Config, seed_everything


def train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch_index):
    """
    Trains the model for one epoch with Head Warmup and Gradient Accumulation.
    """
    # Head Warmup Logic: Freeze backbone in Epoch 0
    if epoch_index == 0:
        # Identify head parameters by ID
        head_params = list(model.head.parameters()) + list(
            model.layer_norm.parameters()
        )
        head_param_ids = {id(p) for p in head_params}

        for p in model.parameters():
            if id(p) in head_param_ids:
                p.requires_grad = True
            else:
                p.requires_grad = False
    else:
        # Unfreeze everything
        for p in model.parameters():
            p.requires_grad = True

    model.train()
    total_loss = 0.0
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        q_ids = batch["q_input_ids"].to(device)
        q_mask = batch["q_attention_mask"].to(device)
        a_ids = batch["a_input_ids"].to(device)
        a_mask = batch["a_attention_mask"].to(device)
        targets = batch["targets"].to(device)

        outputs = model(q_ids, q_mask, a_ids, a_mask)
        loss = loss_fn(outputs, targets)

        # Gradient Accumulation
        loss = loss / Config.accum_steps
        loss.backward()

        if (step + 1) % Config.accum_steps == 0 or (step + 1) == len(train_loader):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * Config.accum_steps

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch_index + 1} Training Loss: {avg_loss}")
    return avg_loss


def validate(model, val_loader, device):
    """
    Evaluates the model and computes Mean Column-wise Spearman's Correlation.
    """
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in val_loader:
            q_ids = batch["q_input_ids"].to(device)
            q_mask = batch["q_attention_mask"].to(device)
            a_ids = batch["a_input_ids"].to(device)
            a_mask = batch["a_attention_mask"].to(device)
            y = batch["targets"].cpu().numpy()

            outputs = model(q_ids, q_mask, a_ids, a_mask)
            # Apply sigmoid to map logits to [0, 1]
            probs = torch.sigmoid(outputs).cpu().numpy()

            preds.append(probs)
            targets.append(y)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    spearman_scores = []
    for i in range(targets.shape[1]):
        # Avoid NaNs if a column is constant
        if np.std(targets[:, i]) < 1e-9 or np.std(preds[:, i]) < 1e-9:
            score = 0.0
        else:
            score = spearmanr(targets[:, i], preds[:, i]).correlation
        spearman_scores.append(score)

    mean_spearman = np.nanmean(spearman_scores)
    print(f"Validation Spearman Correlation: {mean_spearman}")
    return mean_spearman


def run_training(model, train_loader, val_loader, device):
    """
    Main driver for training loop with Differential Learning Rates and Phantom Scheduling.
    """
    seed_everything(Config.seed)
    model.to(device)

    # Differential Learning Rates
    head_params = list(model.head.named_parameters()) + list(
        model.layer_norm.named_parameters()
    )
    head_ids = {id(p) for n, p in head_params}

    backbone_params = []
    for n, p in model.named_parameters():
        if id(p) not in head_ids:
            backbone_params.append(p)

    optimizer_grouped_parameters = [
        {"params": backbone_params, "lr": Config.lr_backbone},
        {"params": [p for n, p in head_params], "lr": Config.lr_head},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=Config.weight_decay)

    # Phantom Scheduling
    # We calculate steps based on phantom_epochs (7) but loop for epochs (3)
    num_update_steps_per_epoch = len(train_loader) // Config.accum_steps
    max_train_steps = Config.phantom_epochs * num_update_steps_per_epoch

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,  # Head warmup is handled by freezing, not scheduler warmup
        num_training_steps=max_train_steps,
    )

    best_score = -1.0
    best_model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")

    for epoch in range(Config.epochs):
        print(f"Starting Epoch {epoch + 1}/{Config.epochs}...")

        train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        score = validate(model, val_loader, device)

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_model_path)
            print(f"New Best Model Saved! Score: {best_score}")

    print(f"Training Complete. Best Validation Score: {best_score}")
    return best_score


def predict_and_submit(model, test_loader, test_df, device):
    """
    Generates predictions for the test set and creates the submission file.
    """
    # Load best model weights
    best_model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model for inference.")
    else:
        print("Warning: Best model not found. Using current model weights.")

    model.eval()
    model.to(device)

    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            q_ids = batch["q_input_ids"].to(device)
            q_mask = batch["q_attention_mask"].to(device)
            a_ids = batch["a_input_ids"].to(device)
            a_mask = batch["a_attention_mask"].to(device)

            outputs = model(q_ids, q_mask, a_ids, a_mask)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)

    final_preds = np.concatenate(all_preds, axis=0)

    # Create submission DataFrame
    sub_df = pd.DataFrame(final_preds, columns=Config.target_cols)

    # Insert qa_id from test dataframe
    sub_df.insert(0, "qa_id", test_df["qa_id"].values)

    # Save submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
