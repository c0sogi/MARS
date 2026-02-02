import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint
from library.data_processing import load_data
from library.model import HybridDebertaModel


def train_fn(model, dataloader, optimizer, scheduler, device, criterion):
    """
    Executes one training epoch.
    """
    model.train()
    final_loss = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        structural_features = batch["structural_features"].to(device)
        targets = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask, structural_features)
        loss = criterion(logits.view(-1), targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        final_loss += loss.item()

    avg_loss = final_loss / len(dataloader)
    return avg_loss


def eval_fn(model, dataloader, device, criterion):
    """
    Evaluates the model on the validation set.
    Returns average loss and AUC score.
    """
    model.eval()
    final_loss = 0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            structural_features = batch["structural_features"].to(device)
            targets = batch["label"].to(device)

            logits = model(input_ids, attention_mask, structural_features)
            loss = criterion(logits.view(-1), targets)

            final_loss += loss.item()

            # Apply sigmoid to get probabilities for AUC
            preds = torch.sigmoid(logits).view(-1).cpu().numpy()
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds)

    avg_loss = final_loss / len(dataloader)

    # Calculate AUC
    # Handle edge case where only one class is present in batch (unlikely in full val set)
    try:
        auc_score = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc_score = 0.5

    return avg_loss, auc_score


def predict_fn(model, dataloader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            structural_features = batch["structural_features"].to(device)

            logits = model(input_ids, attention_mask, structural_features)
            preds = torch.sigmoid(logits).view(-1).cpu().numpy()
            all_preds.extend(preds)

    return np.array(all_preds)


def run_training(load_cached_data=True):
    """
    Main function to run the training pipeline, evaluation, and submission generation.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    print(f"Device: {device}")

    # 1. Load Data
    print("Loading data...")
    train_dataset, val_dataset, test_dataset = load_data(
        load_cached_data=load_cached_data
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Initialize Model
    print("Initializing model...")
    model = HybridDebertaModel(
        model_name=Config.MODEL_NAME,
        num_structural_features=Config.SVD_COMPONENTS,
        hidden_size=Config.HIDDEN_SIZE,
        dropout_rate=Config.DROPOUT_RATE,
    )
    model.to(device)

    # 3. Optimizer and Scheduler
    # Differential learning rates
    optimizer_parameters = [
        {
            "params": model.backbone.parameters(),
            "lr": Config.LR_BACKBONE,
            "weight_decay": Config.WEIGHT_DECAY,
        },
        {
            "params": model.fusion_head.parameters(),
            "lr": Config.LR_HEAD,
            "weight_decay": Config.WEIGHT_DECAY,
        },
    ]

    optimizer = AdamW(optimizer_parameters)

    num_train_steps = len(train_loader) * Config.NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_train_steps),
        num_training_steps=num_train_steps,
    )

    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.bin")

    print("Starting training...")
    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_fn(
            model, train_loader, optimizer, scheduler, device, criterion
        )
        val_loss, val_auc = eval_fn(model, val_loader, device, criterion)

        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Checkpoint and Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_auc, best_model_path)
            print("New best model saved.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 5. Generate Submission
    print("Generating submission...")

    # Load best model
    checkpoint = load_checkpoint(best_model_path, model, device=device)
    print(
        f"Loaded best model from epoch {checkpoint['epoch']} with AUC {checkpoint['metric_value']}"
    )

    predictions = predict_fn(model, test_loader, device)

    # Create submission DataFrame
    # We assume the sample submission format is just IDs (if any) and predictions,
    # but based on provided info, sample_submission_null.csv has columns like Insult, Date, Comment.
    # The task description says "Your predictions should be a number in the range [0,1]."
    # and "See 'sample_submissions_null.csv' for the correct format."
    # Usually, submission files for Kaggle-like tasks need an ID and a Prediction.
    # However, looking at the sample_submission_null.csv content in the prompt:
    # It has columns: Insult, Date, Comment.
    # The 'Insult' column seems to be the target.
    # We should likely output a CSV with the same structure but filled Insult column,
    # OR just the required columns.
    # Given standard practices and the prompt "Submission Format: Your predictions should be a number...",
    # we will reconstruct the dataframe from metadata/test.csv and fill the Insult column.

    # Load test metadata to preserve structure/order
    test_df = pd.read_csv(Config.TEST_DATA_PATH)

    # Assign predictions
    test_df["Insult"] = predictions

    # Ensure output directory exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Save
    # Note: The sample submission shows 'Insult', 'Date', 'Comment'.
    # We will save exactly these columns.
    submission_cols = ["Insult", "Date", "Comment"]

    # If Date/Comment are missing in test_df (unlikely given metadata generation), handle gracefully
    if "Date" not in test_df.columns:
        test_df["Date"] = ""
    if "Comment" not in test_df.columns:
        test_df["Comment"] = ""

    test_df[submission_cols].to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
