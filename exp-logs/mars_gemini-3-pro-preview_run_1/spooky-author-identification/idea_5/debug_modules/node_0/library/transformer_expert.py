import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm.auto import tqdm

# Import library utilities
from library.utils import seed_everything, calculate_log_loss, ensure_directory
from library.data_loader import AuthorDataset, create_stratified_folds, LABEL_MAP

# Constants
WORKING_DIR = "./working/idea_5/"
MODEL_DIR = os.path.join(WORKING_DIR, "transformer_models")
CACHE_DIR = WORKING_DIR
MODEL_NAME = "microsoft/deberta-v3-large"
NUM_CLASSES = 3


class DebertaClassifier(nn.Module):
    """
    Transformer-based classifier using DeBERTa-v3-large backbone.
    """

    def __init__(self, model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)

        # DeBERTa V3 hidden size is usually 1024 for large
        self.fc = nn.Linear(self.config.hidden_size, num_classes)

        # Initialize weights for the head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        # Forward pass through backbone
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Use the representation of the [CLS] token (first token)
        # DeBERTa v3 outputs last_hidden_state
        last_hidden_state = outputs.last_hidden_state
        cls_embeddings = last_hidden_state[:, 0, :]

        # Pass through classifier
        logits = self.fc(cls_embeddings)
        return logits


def inference_fn(model, dataloader, device):
    """
    Generates probabilities for a dataloader using the given model.
    """
    model.eval()
    preds = []

    # Disable gradients for inference
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            preds.append(probs.cpu().numpy())

    return np.concatenate(preds, axis=0)


def train_transformer_fold(fold, df_train, df_val, tokenizer, cfg):
    """
    Trains the transformer model for a single fold.

    Args:
        fold (int): Fold number.
        df_train (pd.DataFrame): Training data for this fold.
        df_val (pd.DataFrame): Validation data for this fold.
        tokenizer: HuggingFace tokenizer.
        cfg (dict): Configuration dictionary.

    Returns:
        tuple: (best_val_probs, path_to_best_model)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_directory(MODEL_DIR)

    # Datasets
    train_dataset = AuthorDataset(df_train, tokenizer, max_len=cfg["max_len"])
    val_dataset = AuthorDataset(df_val, tokenizer, max_len=cfg["max_len"])

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    model = DebertaClassifier(model_name=cfg["model_name"])
    model.to(device)

    # Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    num_train_steps = int(len(df_train) / cfg["batch_size"] * cfg["epochs"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * 0.1),
        num_training_steps=num_train_steps,
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Mixed Precision
    scaler = torch.amp.GradScaler("cuda")

    best_loss = float("inf")
    best_model_path = os.path.join(MODEL_DIR, f"transformer_fold_{fold}.pt")
    best_val_probs = None
    patience_counter = 0

    print(f"Starting training for Fold {fold}...")

    for epoch in range(cfg["epochs"]):
        # --- Training ---
        model.train()
        train_loss_accum = 0

        # Use simple loop without progress bar to reduce log noise, or minimal logging
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device)

            with torch.amp.autocast("cuda"):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()

            # Gradient Accumulation handling if needed, but here we assume batch_size is sufficient
            # or handled via cfg['batch_size']. If accumulation is needed, logic goes here.
            # Assuming standard step for simplicity unless memory is tight.

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- Validation ---
        val_probs = inference_fn(model, val_loader, device)
        val_labels = df_val["author"].map(LABEL_MAP).values
        val_loss = calculate_log_loss(val_labels, val_probs)

        print(
            f"Fold {fold} | Epoch {epoch+1}/{cfg['epochs']} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.10f}"
        )

        # --- Checkpointing ---
        if val_loss < best_loss:
            best_loss = val_loss
            best_val_probs = val_probs
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg["patience"]:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Cleanup
    del model, optimizer, scheduler, scaler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    return best_val_probs, best_model_path


def run_transformer_expert(
    n_folds=5,
    seed=42,
    debug=False,
    load_cached_data=True,
    epochs=4,
    batch_size=8,
    lr=1e-5,
):
    """
    Main function to run the Transformer Expert (DeBERTa).

    Args:
        n_folds (int): Number of CV folds.
        seed (int): Random seed.
        debug (bool): Debug mode (smaller data, fewer epochs).
        load_cached_data (bool): Whether to load OOF/Test preds from cache.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size per GPU.
        lr (float): Learning rate.

    Returns:
        tuple: (oof_preds, test_preds)
    """
    seed_everything(seed)
    ensure_directory(WORKING_DIR)
    ensure_directory(MODEL_DIR)

    oof_cache_path = os.path.join(WORKING_DIR, "oof_transformer.npy")
    test_cache_path = os.path.join(WORKING_DIR, "test_transformer.npy")

    # 1. Check Cache
    if (
        load_cached_data
        and os.path.exists(oof_cache_path)
        and os.path.exists(test_cache_path)
    ):
        print(f"Loading cached Transformer Expert predictions from {WORKING_DIR}...")
        try:
            oof_preds = np.load(oof_cache_path)
            test_preds = np.load(test_cache_path)
            return oof_preds, test_preds
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print(f"Running Transformer Expert ({MODEL_NAME})...")

    # Configuration
    cfg = {
        "model_name": MODEL_NAME,
        "max_len": 256,
        "batch_size": batch_size,
        "epochs": 2 if debug else epochs,
        "lr": lr,
        "weight_decay": 0.01,
        "patience": 2,
    }

    # 2. Load Data
    df_folds = create_stratified_folds(
        data_path="./metadata/train.csv",
        n_folds=n_folds,
        seed=seed,
        load_cached_data=load_cached_data,
        debug=debug,
    )

    df_test = pd.read_csv("./metadata/test.csv")
    if debug:
        df_test = df_test.head(100)
        print(f"Debug mode: Sampled {len(df_test)} test rows.")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    # Prepare containers
    oof_preds = np.zeros((len(df_folds), NUM_CLASSES))
    test_preds_accum = np.zeros((len(df_test), NUM_CLASSES))

    scores = []

    # 3. Training Loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold in range(n_folds):
        print(f"\n--- Transformer Fold {fold + 1}/{n_folds} ---")

        # Split Data
        train_idx = df_folds["fold"] != fold
        val_idx = df_folds["fold"] == fold

        df_train_fold = df_folds[train_idx].reset_index(drop=True)
        df_val_fold = df_folds[val_idx].reset_index(drop=True)

        # Train
        val_probs, model_path = train_transformer_fold(
            fold, df_train_fold, df_val_fold, tokenizer, cfg
        )

        # Store OOF
        # Map validation indices back to original dataframe
        val_indices = df_folds.index[val_idx]
        oof_preds[val_indices] = val_probs

        # Record Score
        val_labels = df_val_fold["author"].map(LABEL_MAP).values
        fold_loss = calculate_log_loss(val_labels, val_probs)
        scores.append(fold_loss)
        print(f"Fold {fold + 1} Best Log Loss: {fold_loss}")

        # Inference on Test
        print(f"Generating test predictions for Fold {fold + 1}...")

        # Load best model for this fold
        model = DebertaClassifier(model_name=cfg["model_name"], pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        test_dataset = AuthorDataset(
            df_test, tokenizer, max_len=cfg["max_len"], is_test=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg["batch_size"] * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        fold_test_probs = inference_fn(model, test_loader, device)
        test_preds_accum += fold_test_probs

        # Cleanup
        del model, test_loader, test_dataset
        torch.cuda.empty_cache()
        gc.collect()

    # 4. Aggregate
    test_preds = test_preds_accum / n_folds

    # Calculate Overall CV Score
    # Need labels for all data
    all_labels = df_folds["author"].map(LABEL_MAP).values
    overall_loss = calculate_log_loss(all_labels, oof_preds)

    print(f"\nTransformer Overall CV Log Loss: {overall_loss}")
    print(f"Average Fold Log Loss: {np.mean(scores)}")

    # 5. Save Predictions
    try:
        np.save(oof_cache_path, oof_preds)
        np.save(test_cache_path, test_preds)
        print(f"Saved OOF predictions to {oof_cache_path}")
        print(f"Saved Test predictions to {test_cache_path}")
    except Exception as e:
        print(f"Error saving predictions: {e}")

    return oof_preds, test_preds
