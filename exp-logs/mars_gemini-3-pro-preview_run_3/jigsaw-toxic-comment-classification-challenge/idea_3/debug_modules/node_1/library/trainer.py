import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, calculate_roc_auc
from library.dataset import load_data_splits, ToxicDataset
from library.neural_model import DebertaClassifier


def train_fn(model, data_loader, optimizer, scheduler, device, epoch):
    model.train()
    final_loss = 0

    # Using tqdm for progress tracking is standard, but requirements say
    # "Only print the required information. Do not print progress bars".
    # So we will iterate silently or with minimal logging.

    for batch_idx, data in enumerate(data_loader):
        input_ids = data["input_ids"].to(device, dtype=torch.long)
        attention_mask = data["attention_mask"].to(device, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
        labels = data["labels"].to(device, dtype=torch.float)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask, token_type_ids)
        loss = nn.BCEWithLogitsLoss()(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        final_loss += loss.item()

    avg_loss = final_loss / len(data_loader)
    return avg_loss


def inference_fn(model, data_loader, device):
    model.eval()
    final_preds = []

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            input_ids = data["input_ids"].to(device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)

            outputs = model(input_ids, attention_mask, token_type_ids)
            # Apply sigmoid to convert logits to probabilities
            preds = torch.sigmoid(outputs).cpu().numpy()
            final_preds.append(preds)

    return np.vstack(final_preds)


def run_training():
    seed_everything(Config.SEED)

    # --- Data Loading ---
    print("Loading data splits...")
    df_train, df_val, df_test = load_data_splits()

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Create Datasets
    train_dataset = ToxicDataset(df_train, tokenizer, Config.MAX_LEN)
    val_dataset = ToxicDataset(df_val, tokenizer, Config.MAX_LEN)
    test_dataset = ToxicDataset(df_test, tokenizer, Config.MAX_LEN, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Model Setup ---
    print("Initializing model...")
    device = Config.DEVICE
    model = DebertaClassifier(Config.MODEL_NAME, Config.NUM_LABELS)
    model.to(device)

    # --- Optimizer & Scheduler ---
    # Use LLRD (Layer-wise Learning Rate Decay)
    optimizer_grouped_parameters = model.get_optimizer_params(
        base_lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        llrd_decay=Config.LLRD_DECAY,
    )

    optimizer = AdamW(optimizer_grouped_parameters, lr=Config.LEARNING_RATE)

    num_train_steps = int(len(df_train) / Config.TRAIN_BATCH_SIZE * Config.EPOCHS)
    num_warmup_steps = int(num_train_steps * Config.WARMUP_RATIO)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # --- Training Loop ---
    best_score = -1.0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        avg_loss = train_fn(model, train_loader, optimizer, scheduler, device, epoch)

        # Validate
        val_preds = inference_fn(model, val_loader, device)
        val_labels = df_val[Config.LABEL_COLS].values

        val_score = calculate_roc_auc(val_labels, val_preds)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Loss: {avg_loss} - Val AUC: {val_score}"
        )

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)

    print(f"Training complete. Best Val AUC: {best_score}")

    # --- Final Inference ---
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT_PATH))
    model.to(device)

    print("Generating validation predictions...")
    val_preds = inference_fn(model, val_loader, device)

    print("Generating test predictions...")
    test_preds = inference_fn(model, test_loader, device)

    # --- Save Predictions ---
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Saving predictions to {Config.WORKING_DIR}...")
    np.save(Config.PRED_DEBERTA_VAL, val_preds)
    np.save(Config.PRED_DEBERTA_TEST, test_preds)

    return val_preds, test_preds
