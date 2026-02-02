import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import get_cosine_schedule_with_warmup
from library.config import Config
from library.utils import compute_spearmanr, save_checkpoint
from library.model import SiameseCoAttentionNetwork


def get_optimizer_params(model):
    """
    Sets up differential learning rates and weight decay groups.
    Backbone gets lr_backbone, Head/CoAttn gets lr_head.
    """
    backbone_params = list(model.backbone.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # Backbone parameter groups
    backbone_decay = [
        p for n, p in backbone_params if not any(nd in n for nd in no_decay)
    ]
    backbone_no_decay = [
        p for n, p in backbone_params if any(nd in n for nd in no_decay)
    ]

    # Identify backbone parameter IDs to separate them from the rest
    backbone_ids = {id(p) for p in model.backbone.parameters()}

    # Head/Interaction parameter groups
    head_decay = []
    head_no_decay = []

    for n, p in model.named_parameters():
        if id(p) not in backbone_ids:
            if any(nd in n for nd in no_decay):
                head_no_decay.append(p)
            else:
                head_decay.append(p)

    optimizer_grouped_parameters = [
        {
            "params": backbone_decay,
            "lr": Config.lr_backbone,
            "weight_decay": Config.weight_decay,
        },
        {"params": backbone_no_decay, "lr": Config.lr_backbone, "weight_decay": 0.0},
        {
            "params": head_decay,
            "lr": Config.lr_head,
            "weight_decay": Config.weight_decay,
        },
        {"params": head_no_decay, "lr": Config.lr_head, "weight_decay": 0.0},
    ]
    return optimizer_grouped_parameters


def train_fn(dataloader, model, criterion, optimizer, scheduler, device, epoch):
    """
    Training loop for one epoch.
    """
    model.train()
    final_loss = 0
    count = 0

    for step, batch in enumerate(dataloader):
        # Move inputs to device
        q_input_ids = batch["q_input_ids"].to(device)
        q_attention_mask = batch["q_attention_mask"].to(device)
        q_token_type_ids = batch["q_token_type_ids"].to(device)
        a_input_ids = batch["a_input_ids"].to(device)
        a_attention_mask = batch["a_attention_mask"].to(device)
        cats = batch["cats"].to(device)
        targets = batch["labels"].to(device)

        # Forward pass
        outputs = model(
            q_input_ids,
            q_attention_mask,
            q_token_type_ids,
            a_input_ids,
            a_attention_mask,
            cats,
        )

        loss = criterion(outputs, targets)

        # Gradient Accumulation
        loss = loss / Config.accumulation_steps
        loss.backward()

        if (step + 1) % Config.accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        final_loss += (
            loss.item() * Config.accumulation_steps
        )  # Scale back up for reporting
        count += 1

    avg_loss = final_loss / count
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss}")
    return avg_loss


def eval_fn(dataloader, model, criterion, device):
    """
    Evaluation loop for validation set.
    """
    model.eval()
    final_loss = 0
    count = 0
    preds = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            q_input_ids = batch["q_input_ids"].to(device)
            q_attention_mask = batch["q_attention_mask"].to(device)
            q_token_type_ids = batch["q_token_type_ids"].to(device)
            a_input_ids = batch["a_input_ids"].to(device)
            a_attention_mask = batch["a_attention_mask"].to(device)
            cats = batch["cats"].to(device)
            targets = batch["labels"].to(device)

            outputs = model(
                q_input_ids,
                q_attention_mask,
                q_token_type_ids,
                a_input_ids,
                a_attention_mask,
                cats,
            )

            loss = criterion(outputs, targets)
            final_loss += loss.item()
            count += 1

            preds.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())

    avg_loss = final_loss / count
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    # Compute metric
    score = compute_spearmanr(labels, preds)

    return avg_loss, score, preds


def predict_fn(dataloader, model, device):
    """
    Inference loop for test set.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            q_input_ids = batch["q_input_ids"].to(device)
            q_attention_mask = batch["q_attention_mask"].to(device)
            q_token_type_ids = batch["q_token_type_ids"].to(device)
            a_input_ids = batch["a_input_ids"].to(device)
            a_attention_mask = batch["a_attention_mask"].to(device)
            cats = batch["cats"].to(device)

            outputs = model(
                q_input_ids,
                q_attention_mask,
                q_token_type_ids,
                a_input_ids,
                a_attention_mask,
                cats,
            )

            preds.append(outputs.cpu().numpy())

    return np.concatenate(preds)


def run_training(train_loader, val_loader):
    """
    Main function to run the training process.
    """
    device = Config.device

    # Initialize Model
    model = SiameseCoAttentionNetwork()
    model.to(device)

    # Optimizer
    optimizer_parameters = get_optimizer_params(model)
    optimizer = torch.optim.AdamW(optimizer_parameters)

    # Scheduler
    num_train_steps = int(len(train_loader) * Config.epochs / Config.accumulation_steps)
    num_warmup_steps = int(num_train_steps * Config.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # Loss Function
    criterion = nn.BCELoss()

    best_score = -1.0
    best_model_path = os.path.join(Config.output_dir, "best_model.pth")

    print(f"Starting training for {Config.epochs} epochs...")

    for epoch in range(Config.epochs):
        train_loss = train_fn(
            train_loader, model, criterion, optimizer, scheduler, device, epoch
        )
        val_loss, val_score, _ = eval_fn(val_loader, model, criterion, device)

        print(f"Epoch {epoch+1} | Val Loss: {val_loss} | Val Score: {val_score}")

        if val_score > best_score:
            best_score = val_score
            print(f"New best score: {best_score}. Saving model to {best_model_path}...")
            save_checkpoint(model, best_model_path)

    return best_score


def generate_submission(test_loader):
    """
    Loads the best model, predicts on test set, and saves submission.csv.
    """
    device = Config.device
    best_model_path = os.path.join(Config.output_dir, "best_model.pth")

    print("Loading best model for submission...")
    model = SiameseCoAttentionNetwork()
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    print("Predicting on test set...")
    preds = predict_fn(test_loader, model, device)

    # Load test metadata to get qa_ids
    df_test = pd.read_csv(Config.test_path)

    # Handle debug mode slicing to align with predictions
    if Config.debug:
        df_test = df_test.iloc[: Config.debug_sample_size]

    # Ensure lengths match
    if len(preds) != len(df_test):
        raise ValueError(
            f"Prediction length {len(preds)} does not match test dataframe length {len(df_test)}"
        )

    print("Creating submission dataframe...")
    submission = pd.DataFrame(preds, columns=Config.target_cols)
    submission.insert(0, "qa_id", df_test["qa_id"].values)

    print(f"Saving submission to {Config.submission_path}...")
    submission.to_csv(Config.submission_path, index=False)
    print("Submission saved successfully.")
