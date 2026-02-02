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


def run_transformer_branch(load_cached_data=True):
    """
    Orchestrates the transformer branch: data loading, training/loading, and prediction.

    Args:
        load_cached_data (bool): If True, attempts to load a pre-trained model from disk.

    Returns:
        tuple: (val_probs, test_probs, y_val)
    """
    print("--- Starting Transformer Branch Pipeline ---")
    set_seed(Config.SEED)
    device = Config.get_device()

    # Ensure directory exists
    os.makedirs(Config.TRANSFORMER_MODEL_DIR, exist_ok=True)
    model_path = os.path.join(Config.TRANSFORMER_MODEL_DIR, "best_model.pt")

    # 1. Load Data
    print("Loading raw metadata...")
    df_train = load_data("train")
    df_val = load_data("val")
    df_test = load_data("test")

    # 2. Initialize Tokenizer
    print(f"Initializing Tokenizer ({Config.MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # 3. Create Datasets
    print("Creating Datasets...")
    train_dataset = AuthorDataset(
        df_train["text"], df_train["author"], tokenizer, Config.MAX_LENGTH
    )
    val_dataset = AuthorDataset(
        df_val["text"], df_val["author"], tokenizer, Config.MAX_LENGTH
    )
    test_dataset = AuthorDataset(
        df_test["text"], labels=None, tokenizer=tokenizer, max_length=Config.MAX_LENGTH
    )

    # 4. Create DataLoaders
    # Pin memory for faster transfer to GPU
    use_pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=use_pin_memory,
    )

    # 5. Train or Load Model
    model = None
    if load_cached_data and os.path.exists(model_path):
        print(f"Loading cached model from {model_path}...")
        try:
            model = TransformerModel(Config.MODEL_NAME, num_classes=3)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
        except Exception as e:
            print(f"Failed to load cached model: {e}. Proceeding to retrain.")
            model = None

    if model is None:
        model = train_transformer(
            train_loader, val_loader, device, Config.TRANSFORMER_MODEL_DIR
        )

    # 6. Generate Predictions
    print("Generating predictions...")
    val_probs = predict_transformer(model, val_loader, device)
    test_probs = predict_transformer(model, test_loader, device)

    # Get ground truth for validation
    y_val = df_val["author"].map(Config.LABEL2ID).values

    print("--- Transformer Branch Pipeline Complete ---")
    return val_probs, test_probs, y_val
