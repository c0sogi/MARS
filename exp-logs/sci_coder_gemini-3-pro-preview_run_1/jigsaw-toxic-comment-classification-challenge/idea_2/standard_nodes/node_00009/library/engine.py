import os
import re
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from transformers import get_linear_schedule_with_warmup
from library.config import Config


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def loss_fn(outputs, targets):
    """Computes the Binary Cross Entropy Loss with Logits."""
    return nn.BCEWithLogitsLoss()(outputs, targets)


def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["labels"].to(device)

        batch_size = input_ids.size(0)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    return running_loss / dataset_size


def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and mean column-wise ROC AUC.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    fin_targets = []
    fin_outputs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)

            batch_size = input_ids.size(0)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Store targets and sigmoid probabilities for AUC calculation
            fin_targets.append(targets.cpu().numpy())
            fin_outputs.append(torch.sigmoid(outputs).cpu().numpy())

    avg_loss = running_loss / dataset_size

    fin_targets = np.vstack(fin_targets)
    fin_outputs = np.vstack(fin_outputs)

    # Calculate column-wise ROC AUC
    aucs = []
    for i in range(Config.NUM_LABELS):
        try:
            # Only calculate AUC if there is more than one class present
            if len(np.unique(fin_targets[:, i])) > 1:
                auc = roc_auc_score(fin_targets[:, i], fin_outputs[:, i])
                aucs.append(auc)
            else:
                # Fallback if a class is missing in the validation set (rare)
                aucs.append(0.5)
        except ValueError:
            aucs.append(0.5)

    mean_auc = np.mean(aucs)

    return avg_loss, mean_auc


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.
    Returns numpy array of probabilities.
    """
    model.eval()
    fin_outputs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)
            # Apply sigmoid to get probabilities
            fin_outputs.append(torch.sigmoid(outputs).cpu().numpy())

    return np.vstack(fin_outputs)


def run_training(model, train_loader, val_loader):
    """
    Orchestrates the training process including optimization,
    scheduling, and early stopping.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE
    model.to(device)

    # Define Optimizer with Layer-wise Learning Rate Decay (LLRD)
    # Cite Lesson 00007: Optimizing Transfer Learning via Transformers
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    layer_decay = 0.95

    optimizer_grouped_parameters = []
    param_groups = {}  # Key: (lr, weight_decay), Value: list of params

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        lr = Config.LEARNING_RATE

        # Apply decay based on layer depth
        if "embeddings" in name:
            lr *= layer_decay**12
        elif "encoder.layer" in name:
            # Extract layer index to determine depth
            # roberta.encoder.layer.11 is top (decay^1), layer.0 is bottom (decay^12)
            match = re.search(r"encoder\.layer\.(\d+)", name)
            if match:
                layer_idx = int(match.group(1))
                lr *= layer_decay ** (12 - layer_idx)
        # Head parameters (classifier, attention_pooling) keep base LR

        # Determine weight decay
        wd = Config.WEIGHT_DECAY
        if any(nd in name for nd in no_decay):
            wd = 0.0

        key = (lr, wd)
        if key not in param_groups:
            param_groups[key] = []
        param_groups[key].append(param)

    # Flatten groups for optimizer
    for (lr, wd), params in param_groups.items():
        optimizer_grouped_parameters.append(
            {"params": params, "weight_decay": wd, "lr": lr}
        )

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    # Define Scheduler
    num_train_steps = int(len(train_loader) * Config.EPOCHS)
    num_warmup_steps = int(num_train_steps * Config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    best_auc = -np.inf
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss}")

        # Validate
        val_loss, val_auc = evaluate(model, val_loader, device)
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")  # Printing full precision as requested

        # Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            print(f"New best model found! Saving to {Config.MODEL_SAVE_PATH}")
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val AUC: {best_auc}")
