import os
import gc
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from library.config import Config
from library.dataset import EssayDataset
from library.modeling import DebertaRegressor, AWP, train_one_epoch, valid_one_epoch
from library.utils import seed_everything


def train_fold(fold, train_df, val_df):
    """
    Trains the DeBERTa model for a single fold using the configuration settings.
    Handles AWP, scheduling, logging, and model checkpointing.

    Args:
        fold (int): The current fold index.
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.

    Returns:
        float: The best validation QWK score achieved.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # --- Data Preparation ---
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    train_dataset = EssayDataset(train_df, tokenizer)
    val_dataset = EssayDataset(val_df, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Model & Optimizer Setup ---
    model = DebertaRegressor(Config.MODEL_NAME, pretrained=True)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    num_train_steps = int(len(train_loader) * Config.EPOCHS)
    num_warmup_steps = int(num_train_steps * Config.WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # Adversarial Weight Perturbation
    awp = None
    if Config.USE_AWP:
        awp = AWP(
            model, optimizer, adv_lr=Config.AWP_ADV_LR, adv_eps=Config.AWP_ADV_EPS
        )

    # --- Training Loop ---
    best_qwk = -1.0
    patience_counter = 0
    early_stopping_patience = 2  # Stop if no improvement for 2 epochs

    print(f"--- Fold {fold} Training ---")

    for epoch in range(Config.EPOCHS):
        # Train Step
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, epoch, awp
        )

        # Validation Step
        val_loss, val_qwk, _ = valid_one_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val QWK: {val_qwk}"
        )

        # Checkpointing & Early Stopping
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            patience_counter = 0

            save_path = os.path.join(Config.OUTPUT_DIR, f"deberta_fold_{fold}.bin")
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # --- Cleanup ---
    del (
        model,
        optimizer,
        scheduler,
        awp,
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
    )
    torch.cuda.empty_cache()
    gc.collect()

    return best_qwk


def predict(test_df, checkpoint_path):
    """
    Generates predictions for the test set using a trained model checkpoint.

    Args:
        test_df (pd.DataFrame): Test data containing 'full_text'.
        checkpoint_path (str): Path to the saved model .bin file.

    Returns:
        np.ndarray: Array of continuous score predictions.
    """
    device = Config.DEVICE
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    dataset = EssayDataset(test_df, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Model
    model = DebertaRegressor(Config.MODEL_NAME, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    preds = []

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)
            # Flatten predictions
            preds.append(outputs.view(-1).cpu().numpy())

    preds = np.concatenate(preds)

    # Cleanup
    del model, dataloader, dataset
    torch.cuda.empty_cache()
    gc.collect()

    return preds
