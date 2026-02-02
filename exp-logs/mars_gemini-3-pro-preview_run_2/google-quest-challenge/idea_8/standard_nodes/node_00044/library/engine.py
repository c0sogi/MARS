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
    Sets up Layer-wise Learning Rate Decay (LLRD) and differential learning rates.
    Cite solution_lesson_node_00006 (Transfer Learning Dominance) - preserving lower layers.
    """
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []

    # Base LR for backbone
    lr = Config.lr_backbone
    decay = Config.llrd_decay

    # 1. Backbone Layers (DeBERTa-v3-base has 12 layers: 0-11)
    # We assign LRs: layer 11 -> lr, layer 10 -> lr*decay, ..., embeddings -> lr*decay^12

    # Group parameters by layer
    # DeBERTa structure: backbone.embeddings..., backbone.encoder.layer.X...

    layer_params = {i: {"decay": [], "no_decay": []} for i in range(-1, 12)}
    head_params = {"decay": [], "no_decay": []}

    for n, p in model.named_parameters():
        if "backbone" in n:
            # Determine layer index
            if "encoder.layer" in n:
                # Format: backbone.encoder.layer.5.output...
                try:
                    # Split by dot, find index after 'layer'
                    parts = n.split(".")
                    layer_idx = int(parts[parts.index("layer") + 1])
                except:
                    # Fallback
                    layer_idx = 0
            elif "embeddings" in n:
                layer_idx = -1
            else:
                # Other backbone params (e.g. final layernorm if exists, or pooler inside backbone)
                # Treat as top layer
                layer_idx = 11

            if any(nd in n for nd in no_decay):
                layer_params[layer_idx]["no_decay"].append(p)
            else:
                layer_params[layer_idx]["decay"].append(p)
        else:
            # Head / CoAttention / Custom Pooler
            if any(nd in n for nd in no_decay):
                head_params["no_decay"].append(p)
            else:
                head_params["decay"].append(p)

    # Create optimizer groups with calculated LRs
    num_layers = 12

    for layer_idx in range(-1, 12):
        # Calculate decay power
        # layer 11: power 0 -> lr
        # layer 0: power 11 -> lr * decay^11
        # embeddings: power 12 -> lr * decay^12

        if layer_idx == -1:
            power = num_layers
        else:
            power = (num_layers - 1) - layer_idx

        cur_lr = lr * (decay**power)

        # Add groups
        if layer_params[layer_idx]["decay"]:
            optimizer_grouped_parameters.append(
                {
                    "params": layer_params[layer_idx]["decay"],
                    "lr": cur_lr,
                    "weight_decay": Config.weight_decay,
                }
            )
        if layer_params[layer_idx]["no_decay"]:
            optimizer_grouped_parameters.append(
                {
                    "params": layer_params[layer_idx]["no_decay"],
                    "lr": cur_lr,
                    "weight_decay": 0.0,
                }
            )

    # Head Parameters
    optimizer_grouped_parameters.append(
        {
            "params": head_params["decay"],
            "lr": Config.lr_head,
            "weight_decay": Config.weight_decay,
        }
    )
    optimizer_grouped_parameters.append(
        {"params": head_params["no_decay"], "lr": Config.lr_head, "weight_decay": 0.0}
    )

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
