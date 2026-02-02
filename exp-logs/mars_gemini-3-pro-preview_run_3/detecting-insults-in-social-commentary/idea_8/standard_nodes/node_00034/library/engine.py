import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from library.configuration import Config
from library.architecture import TransformerModel
from library.dataset import get_dataloader
from library.utilities import set_seed


def train_fn(data_loader, model, optimizer, scheduler, device, grad_acc_steps):
    """
    Executes one epoch of training with gradient accumulation.
    """
    model.train()
    final_loss = 0

    # Zero gradients at the start of the epoch
    optimizer.zero_grad()

    for step, data in enumerate(data_loader):
        input_ids = data["input_ids"].to(device, dtype=torch.long)
        attention_mask = data["attention_mask"].to(device, dtype=torch.long)
        targets = data["target"].to(device, dtype=torch.float)

        # Forward pass
        outputs = model(input_ids, attention_mask)

        # Compute loss (BCEWithLogitsLoss is appropriate for binary classification)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(outputs.squeeze(), targets)

        # Normalize loss for gradient accumulation
        loss = loss / grad_acc_steps
        loss.backward()

        final_loss += loss.item() * grad_acc_steps

        # Step optimizer and scheduler only after accumulating enough gradients
        if (step + 1) % grad_acc_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return final_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    """
    Evaluates the model on a validation set.
    Returns average loss and AUC score.
    """
    model.eval()
    final_loss = 0
    fin_targets = []
    fin_outputs = []
    loss_fn = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for data in data_loader:
            input_ids = data["input_ids"].to(device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(device, dtype=torch.long)
            targets = data["target"].to(device, dtype=torch.float)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs.squeeze(), targets)
            final_loss += loss.item()

            # Store predictions (sigmoid applied later or implicitly by metric if needed,
            # but for AUC on logits vs probs, order is preserved. We use sigmoid for clarity.)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    avg_loss = final_loss / len(data_loader)

    # Handle edge case where only one class is present in batch
    try:
        auc = roc_auc_score(fin_targets, fin_outputs)
    except ValueError:
        auc = 0.5

    return avg_loss, auc


def predict(data_loader, model, device):
    """
    Generates probability predictions for the test set.
    """
    model.eval()
    fin_outputs = []

    with torch.no_grad():
        for data in data_loader:
            input_ids = data["input_ids"].to(device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(device, dtype=torch.long)

            outputs = model(input_ids, attention_mask)

            # Apply sigmoid to get probabilities in [0, 1]
            probs = torch.sigmoid(outputs).squeeze()

            # Handle batch size of 1 where squeeze might remove the batch dim entirely if not careful,
            # but usually for inference we want a list.
            if probs.ndim == 0:
                probs = probs.unsqueeze(0)

            fin_outputs.extend(probs.cpu().detach().numpy().tolist())

    return fin_outputs


def run_training(df_train, df_val, model_config, seed, save_name):
    """
    Orchestrates the training process for a specific model configuration and seed.

    Args:
        df_train (pd.DataFrame): Training data.
        df_val (pd.DataFrame or None): Validation data. If None, trains for fixed epochs without validation.
        model_config (dict): Configuration dictionary for the specific model architecture.
        seed (int): Random seed.
        save_name (str): Filename to save the trained model.
    """
    # 1. Set Seed
    set_seed(seed)
    device = Config.DEVICE

    print(f"Starting training for {model_config['model_name']} (Seed {seed})...")

    # 2. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_path"])

    # 3. Create DataLoaders
    train_loader = get_dataloader(
        df_train,
        tokenizer,
        batch_size=Config.TRAIN_BATCH_SIZE,
        is_test=False,
        shuffle=True,
    )

    val_loader = None
    if df_val is not None:
        val_loader = get_dataloader(
            df_val,
            tokenizer,
            batch_size=Config.VALID_BATCH_SIZE,
            is_test=False,
            shuffle=False,
        )

    # 4. Initialize Model
    model = TransformerModel(
        model_name=model_config["model_name"],
        dropout=model_config["dropout"],
        freeze_layers=model_config["freeze_layers"],
    )
    model.to(device)

    # 5. Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Calculate training steps
    # Effective Batch Size logic is handled in train_fn via accumulation,
    # but total optimizer steps depend on the number of updates.
    num_update_steps_per_epoch = len(train_loader) // Config.GRAD_ACC_STEPS
    max_train_steps = num_update_steps_per_epoch * model_config["epochs"]
    num_warmup_steps = int(max_train_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps
    )

    # 6. Training Loop
    best_auc = 0.0
    patience_counter = 0
    early_stopping_patience = (
        2  # Generic patience, though fixed epochs usually preferred for full data
    )

    for epoch in range(model_config["epochs"]):
        train_loss = train_fn(
            train_loader, model, optimizer, scheduler, device, Config.GRAD_ACC_STEPS
        )

        print(
            f"Epoch {epoch + 1}/{model_config['epochs']} | Train Loss: {train_loss:.16f}"
        )

        # If validation data is available, evaluate and check early stopping
        if val_loader is not None:
            val_loss, val_auc = eval_fn(val_loader, model, device)
            print(
                f"Epoch {epoch + 1}/{model_config['epochs']} | Val Loss: {val_loss:.16f} | Val AUC: {val_auc:.16f}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                # Save best model found so far
                torch.save(
                    model.state_dict(), os.path.join(Config.MODEL_DIR, save_name)
                )
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break
        else:
            # Full Data Training Mode: Save at the end of every epoch (or just overwrite)
            # Since we trust the fixed epoch count, we save the latest state.
            torch.save(model.state_dict(), os.path.join(Config.MODEL_DIR, save_name))

    print(f"Training complete for {save_name}.")
