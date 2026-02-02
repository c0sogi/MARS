import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    logging,
)
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, compute_log_loss
from library.data_loader import load_data, AuthorDataset

# Suppress transformer warnings
logging.set_verbosity_error()


class TransformerModel(nn.Module):
    """
    A PyTorch module wrapping a pre-trained Transformer with a classification head.
    """

    def __init__(self, model_name=Config.MODEL_NAME, num_classes=3):
        super(TransformerModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use the representation of the [CLS] token (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_output)
        logits = self.classifier(x)
        return logits


def predict_transformer(model, data_loader, device):
    """
    Generates probability predictions for the given data loader.

    Args:
        model: The trained TransformerModel.
        data_loader: DataLoader containing the input text.
        device: Torch device (cpu or cuda).

    Returns:
        numpy.ndarray: Predicted probabilities (n_samples, n_classes).
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    return np.vstack(all_probs)


def train_transformer(train_loader, val_loader, device, save_dir):
    """
    Trains the transformer model with AdamW, scheduler, and early stopping.

    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: Torch device.
        save_dir: Directory to save the best model.

    Returns:
        model: The best trained TransformerModel.
    """
    print("Initializing Transformer Model...")
    model = TransformerModel(Config.MODEL_NAME, num_classes=3)
    model.to(device)

    # Optimization setup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    criterion = nn.CrossEntropyLoss()

    # Early stopping tracking
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(save_dir, "best_model.pt")

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss_accum = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask)

                probs = torch.softmax(logits, dim=1)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # Compute metrics
        y_val_true = np.concatenate(all_labels)
        y_val_pred = np.vstack(all_preds)
        # We pass labels=[0, 1, 2] to ensure log_loss knows the class order/count
        epoch_log_loss = compute_log_loss(y_val_true, y_val_pred, labels=[0, 1, 2])

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {avg_train_loss:.6f} - Val Log Loss: {epoch_log_loss}"
        )

        # --- Early Stopping ---
        if epoch_log_loss < best_val_loss:
            best_val_loss = epoch_log_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  -> New best model saved.")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # Load best model for return
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model


from sklearn.model_selection import StratifiedKFold


def run_transformer_branch(load_cached_data=True):
    """
    Orchestrates the transformer branch with K-Fold Cross Validation.
    """
    print("--- Starting Transformer Branch Pipeline (K-Fold) ---")
    set_seed(Config.SEED)
    device = Config.get_device()

    # 1. Load Data
    df_train = load_data("train")
    df_val = load_data("val")
    df_test = load_data("test")

    y_train_full = df_train["author"].map(Config.LABEL2ID).values

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # 3. K-Fold Setup
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    oof_preds = np.zeros((len(df_train), 3))
    val_preds_accum = np.zeros((len(df_val), 3))
    test_preds_accum = np.zeros((len(df_test), 3))

    # Prepare shared datasets for inference
    val_dataset = AuthorDataset(
        df_val["text"], df_val["author"], tokenizer, Config.MAX_LENGTH
    )
    test_dataset = AuthorDataset(
        df_test["text"], labels=None, tokenizer=tokenizer, max_length=Config.MAX_LENGTH
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Training Transformer ({Config.MODEL_NAME}) with {Config.N_FOLDS} folds...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, y_train_full)):
        print(f"\n--- Fold {fold+1}/{Config.N_FOLDS} ---")

        # Create Fold Datasets
        train_sub = df_train.iloc[train_idx]
        val_sub = df_train.iloc[val_idx]

        train_ds = AuthorDataset(
            train_sub["text"], train_sub["author"], tokenizer, Config.MAX_LENGTH
        )
        # We use the fold-validation set for early stopping
        fold_val_ds = AuthorDataset(
            val_sub["text"], val_sub["author"], tokenizer, Config.MAX_LENGTH
        )

        train_dl = DataLoader(
            train_ds,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        fold_val_dl = DataLoader(
            fold_val_ds,
            batch_size=Config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Train
        fold_save_dir = os.path.join(Config.TRANSFORMER_MODEL_DIR, f"fold_{fold}")
        os.makedirs(fold_save_dir, exist_ok=True)

        # Check cache
        model_path = os.path.join(fold_save_dir, "best_model.pt")
        model = None
        if load_cached_data and os.path.exists(model_path):
            try:
                print("Loading cached fold model...")
                model = TransformerModel(Config.MODEL_NAME, num_classes=3)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
            except:
                model = None

        if model is None:
            model = train_transformer(train_dl, fold_val_dl, device, fold_save_dir)

        # OOF Inference
        oof_preds[val_idx] = predict_transformer(model, fold_val_dl, device)

        # Accumulate Inference
        val_preds_accum += predict_transformer(model, val_loader, device)
        test_preds_accum += predict_transformer(model, test_loader, device)

        # Clear memory
        del model
        torch.cuda.empty_cache()

    val_probs = val_preds_accum / Config.N_FOLDS
    test_probs = test_preds_accum / Config.N_FOLDS

    loss = compute_log_loss(y_train_full, oof_preds, labels=[0, 1, 2])
    print(f"Transformer Branch OOF Log Loss: {loss}")

    return oof_preds, val_probs, test_probs, y_train_full
