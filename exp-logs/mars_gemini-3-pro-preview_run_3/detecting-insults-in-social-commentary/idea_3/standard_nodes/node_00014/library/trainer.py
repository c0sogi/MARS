import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import get_linear_schedule_with_warmup
from library.config import Config
from library.utils import seed_everything, get_score
from library.data import get_dataloaders
from library.model import InsultModel


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    """
    Constructs the parameter groups for the optimizer.
    Simplified to handle weight decay and standard fine-tuning (Simplicity Lesson).
    """
    # Identify named parameters that require gradients
    # Since we implemented freezing in the model, we filter by requires_grad
    param_optimizer = list(
        filter(lambda p: p[1].requires_grad, model.named_parameters())
    )

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
            "lr": encoder_lr,  # Using single LR for simplicity as per lessons
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": encoder_lr,
        },
    ]

    return optimizer_parameters


def train_fn(dataloader, model, criterion, optimizer, scheduler, device, epoch):
    model.train()

    running_loss = 0.0
    dataset_size = 0

    for step, data in enumerate(dataloader):
        input_ids = data["input_ids"].to(device, dtype=torch.long)
        attention_mask = data["attention_mask"].to(device, dtype=torch.long)
        targets = data["target"].to(device, dtype=torch.float)

        batch_size = input_ids.size(0)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        # Outputs are logits (batch_size, 1)
        # Targets are (batch_size,)

        loss = criterion(outputs.view(-1), targets)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def valid_fn(dataloader, model, criterion, device):
    model.eval()

    running_loss = 0.0
    dataset_size = 0

    preds = []
    valid_labels = []

    with torch.no_grad():
        for step, data in enumerate(dataloader):
            input_ids = data["input_ids"].to(device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(device, dtype=torch.long)
            targets = data["target"].to(device, dtype=torch.float)

            batch_size = input_ids.size(0)

            outputs = model(input_ids, attention_mask)

            loss = criterion(outputs.view(-1), targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for predictions
            batch_preds = torch.sigmoid(outputs.view(-1))

            preds.append(batch_preds.cpu().numpy())
            valid_labels.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size
    preds = np.concatenate(preds)
    valid_labels = np.concatenate(valid_labels)

    auc_score = get_score(valid_labels, preds)

    return epoch_loss, auc_score


def inference_fn(dataloader, model, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for step, data in enumerate(dataloader):
            input_ids = data["input_ids"].to(device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(device, dtype=torch.long)

            outputs = model(input_ids, attention_mask)

            # Apply sigmoid
            batch_preds = torch.sigmoid(outputs.view(-1))
            preds.append(batch_preds.cpu().numpy())

    preds = np.concatenate(preds)
    return preds


def run_training():
    seed_everything(Config.seed)

    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=Config.debug
    )

    print("Initializing Model...")
    device = Config.device
    model = InsultModel()
    model.to(device)

    # Define Loss
    criterion = nn.BCEWithLogitsLoss()

    # Define Optimizer with LLRD
    # Head gets Config.learning_rate, Backbone gets same or slightly less if desired.
    # Here we treat Config.learning_rate as the max LR (head).
    optimizer_parameters = get_optimizer_params(
        model,
        encoder_lr=Config.learning_rate,
        decoder_lr=Config.learning_rate,
        weight_decay=Config.weight_decay,
    )

    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=Config.learning_rate, eps=1e-6
    )

    # Define Scheduler
    num_train_steps = int(len(train_loader) * Config.epochs)
    num_warmup_steps = int(num_train_steps * Config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    best_auc = 0.0
    best_model_path = os.path.join(Config.output_dir, "best_model.bin")

    print(f"Starting training for {Config.epochs} epochs...")

    for epoch in range(Config.epochs):
        start_time = time.time()

        train_loss = train_fn(
            train_loader, model, criterion, optimizer, scheduler, device, epoch
        )
        val_loss, val_auc = valid_fn(val_loader, model, criterion, device)

        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{Config.epochs} | Time: {elapsed:.0f}s")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Val AUC:    {val_auc}")  # Full precision as requested

        if val_auc > best_auc:
            print(f"  AUC Improved ({best_auc} -> {val_auc}). Saving model...")
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
        else:
            print("  AUC did not improve.")

    print(f"Training complete. Best Val AUC: {best_auc}")

    # ==========================================
    # Inference & Submission
    # ==========================================
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)

    print("Generating predictions on test set...")
    predictions = inference_fn(test_loader, model, device)

    # Create Submission DataFrame
    # We need to load the test metadata to get Date and Comment columns
    test_df = pd.read_csv(Config.test_path)

    # If debug mode was on, test_loader only has a subset. We must ensure lengths match.
    if Config.debug:
        test_df = test_df.head(Config.debug_sample_size).reset_index(drop=True)

    if len(test_df) != len(predictions):
        print(
            f"Warning: Length mismatch. Test DF: {len(test_df)}, Preds: {len(predictions)}"
        )

    # Construct dataframe
    submission = pd.DataFrame()
    submission["Insult"] = predictions
    submission["Date"] = test_df["Date"]
    submission["Comment"] = test_df["Comment"]

    # Reorder columns to match sample: Insult, Date, Comment
    submission = submission[["Insult", "Date", "Comment"]]

    print(f"Saving submission to {Config.submission_path}...")
    submission.to_csv(Config.submission_path, index=False)
    print("Submission saved successfully.")
