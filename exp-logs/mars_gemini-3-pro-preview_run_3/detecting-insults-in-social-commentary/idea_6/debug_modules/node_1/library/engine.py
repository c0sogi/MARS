import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from library.config import Config
from library.utils import get_device


def train_fn(dataloader, model, optimizer, scheduler, device, epoch):
    """
    Executes one training epoch with gradient accumulation.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Define loss function
    criterion = nn.BCEWithLogitsLoss()

    optimizer.zero_grad()

    for step, data in enumerate(dataloader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["labels"].to(device).float()

        batch_size = input_ids.size(0)

        # Forward pass
        outputs = model(input_ids, attention_mask)

        # Calculate loss (outputs are logits, labels are 0/1)
        loss = criterion(outputs.view(-1), labels)

        # Normalize loss for gradient accumulation
        loss = loss / Config.gradient_accumulation_steps

        # Backward pass
        loss.backward()

        running_loss += (loss.item() * Config.gradient_accumulation_steps) * batch_size
        dataset_size += batch_size

        # Update weights after accumulation steps
        if (step + 1) % Config.gradient_accumulation_steps == 0 or (step + 1) == len(
            dataloader
        ):
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    avg_loss = running_loss / dataset_size
    return avg_loss


def inference_fn(dataloader, model, device):
    """
    Generates probability predictions for a given dataloader.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs).view(-1).cpu().numpy()
            preds.append(probs)

    predictions = np.concatenate(preds)
    return predictions


def evaluate_fn(dataloader, model, device):
    """
    Evaluates the model on the validation set, returning Loss and AUC.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0
    preds = []
    targets = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            labels = data["labels"].to(device).float()

            batch_size = input_ids.size(0)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1), labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            probs = torch.sigmoid(outputs).view(-1).cpu().numpy()
            preds.append(probs)
            targets.append(labels.cpu().numpy())

    avg_loss = running_loss / dataset_size
    predictions = np.concatenate(preds)
    true_labels = np.concatenate(targets)

    try:
        auc_score = roc_auc_score(true_labels, predictions)
    except ValueError:
        # Handle edge case where only one class is present in batch/subset
        auc_score = 0.5

    return avg_loss, auc_score


def train_runner(train_loader, val_loader, model, save_name="model_best.bin"):
    """
    Manages the full training loop, including optimization, scheduling,
    validation, and early stopping.
    """
    device = get_device()
    model.to(device)

    # Optimizer
    optimizer = AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    # Scheduler
    # Calculate total training steps
    num_update_steps_per_epoch = len(train_loader) / Config.gradient_accumulation_steps
    max_train_steps = int(num_update_steps_per_epoch * Config.num_epochs)
    num_warmup_steps = int(max_train_steps * Config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps
    )

    # Early Stopping variables
    best_auc = -1.0
    patience_counter = 0
    best_model_path = os.path.join(Config.output_dir, save_name)

    print(f"Starting training for {Config.num_epochs} epochs...")

    for epoch in range(Config.num_epochs):
        # Train
        train_loss = train_fn(train_loader, model, optimizer, scheduler, device, epoch)

        # Validate
        val_loss, val_auc = evaluate_fn(val_loader, model, device)

        # Print metrics (full precision)
        print(f"Epoch {epoch+1}/{Config.num_epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Early Stopping Logic
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), best_model_path)
            print(f"Validation AUC improved. Model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.patience}")

        if patience_counter >= Config.patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val AUC: {best_auc}")

    # Load best weights before returning
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model, best_auc
