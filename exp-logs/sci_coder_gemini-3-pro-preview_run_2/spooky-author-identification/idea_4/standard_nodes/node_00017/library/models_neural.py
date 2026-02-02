import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from library.config import (
    WORKING_DIR,
    SEED,
    N_FOLDS,
    DEVICE,
    DEBERTA_MODEL,
    MAX_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    PATIENCE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    GRADIENT_ACCUMULATION_STEPS,
    NUM_WORKERS,
)
from library.utils import seed_everything, calculate_log_loss
from library.data_loader import load_data, TextDataset


class DebertaClassifier(nn.Module):
    """
    Branch B: Disentangled Attention Model using [CLS] token pooling.
    """

    def __init__(self, model_name, num_classes):
        super(DebertaClassifier, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        self.fc = nn.Linear(self.config.hidden_size, num_classes)

        # Initialize weights for the head
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.normal_(self.fc.bias, std=1e-3)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token (index 0)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        logits = self.fc(cls_rep)
        return logits


def train_fn(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast("cuda"):
            logits = model(input_ids, mask)
            loss = criterion(logits, labels)
            loss = loss / GRADIENT_ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

    return total_loss / len(dataloader)


def eval_fn(model, dataloader, device):
    model.eval()
    preds = []
    labels_list = []
    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            if "label" in batch:
                labels = batch["label"].to(device)
                labels_list.append(labels.cpu().numpy())

            with torch.amp.autocast("cuda"):
                logits = model(input_ids, mask)
                if "label" in batch:
                    loss = criterion(logits, labels)
                    total_loss += loss.item()

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds.append(probs)

    preds = np.concatenate(preds, axis=0)
    avg_loss = total_loss / len(dataloader) if labels_list else 0

    return preds, avg_loss


def run_single_model_cv(
    model_name, model_class, tokenizer, X_text, y_full, test_text, n_classes, cache_key
):
    """
    Helper function to run CV for a single neural architecture.
    """
    n_samples = len(X_text)
    n_test = len(test_text)

    oof_preds = np.zeros((n_samples, n_classes))
    test_preds = np.zeros((n_test, n_classes))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    print(f"Starting CV for {cache_key} ({model_name})...")

    # Prepare Test Loader once
    test_dataset = TextDataset(
        test_text, labels=None, tokenizer=tokenizer, max_length=MAX_LENGTH
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n_samples), y_full)):
        print(f"\n--- {cache_key} Fold {fold + 1}/{N_FOLDS} ---")

        # Split Data
        train_txt, val_txt = X_text[train_idx], X_text[val_idx]
        y_train, y_val = y_full[train_idx], y_full[val_idx]

        # Datasets
        train_dataset = TextDataset(
            train_txt, labels=y_train, tokenizer=tokenizer, max_length=MAX_LENGTH
        )
        val_dataset = TextDataset(
            val_txt, labels=y_val, tokenizer=tokenizer, max_length=MAX_LENGTH
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = model_class(model_name, n_classes)
        model.to(DEVICE)

        # Optimizer & Scheduler
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_parameters, lr=LEARNING_RATE)
        num_train_steps = int(len(train_loader) * EPOCHS / GRADIENT_ACCUMULATION_STEPS)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * 0.1),
            num_training_steps=num_train_steps,
        )

        # Training Loop with Early Stopping
        best_loss = float("inf")
        patience_counter = 0
        best_weights = None

        for epoch in range(EPOCHS):
            train_loss = train_fn(model, train_loader, optimizer, scheduler, DEVICE)
            val_preds_fold, val_loss = eval_fn(model, val_loader, DEVICE)

            print(
                f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.15f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

        # Load Best Weights
        if best_weights is not None:
            model.load_state_dict(best_weights)

        # Generate Final Predictions for Fold
        val_preds_final, _ = eval_fn(model, val_loader, DEVICE)
        oof_preds[val_idx] = val_preds_final

        test_preds_fold, _ = eval_fn(model, test_loader, DEVICE)
        test_preds += test_preds_fold / N_FOLDS

        # Cleanup
        del (
            model,
            optimizer,
            scheduler,
            train_loader,
            val_loader,
            train_dataset,
            val_dataset,
        )
        torch.cuda.empty_cache()
        gc.collect()

    return oof_preds, test_preds


def run_neural_cv(load_cached_preds=True):
    """
    Orchestrates the training and prediction of neural models (DeBERTa, RoBERTa)
    using Stratified K-Fold Cross-Validation.

    Args:
        load_cached_preds (bool): If True, attempts to load predictions from disk.

    Returns:
        dict: A dictionary containing OOF and Test predictions for each model.
    """
    seed_everything(SEED)

    # Define paths for caching
    cache_files = {
        "deberta_oof": os.path.join(WORKING_DIR, "oof_deberta.npy"),
        "deberta_test": os.path.join(WORKING_DIR, "pred_test_deberta.npy"),
    }

    # Check cache
    if load_cached_preds and all(os.path.exists(p) for p in cache_files.values()):
        print("Loading cached neural model predictions...")
        results = {}
        for k, v in cache_files.items():
            results[k] = np.load(v)
        return results

    print("Running Neural Models CV...")

    # Load Data
    train_df, val_df, test_df = load_data()

    # Prepare Labels (Combine Train+Val)
    le = LabelEncoder()
    le.fit(train_df["author"])
    y_train = le.transform(train_df["author"])
    y_val = le.transform(val_df["author"])
    y_full = np.concatenate([y_train, y_val])

    # Prepare Text
    X_text = pd.concat([train_df["text"], val_df["text"]], axis=0).values
    test_text = test_df["text"].values

    n_classes = 3
    results = {}

    # ---------------------------
    # Branch B: DeBERTa (CLS Token)
    # ---------------------------
    print(f"\n=== Training Branch B: {DEBERTA_MODEL} ===")
    tokenizer_deb = AutoTokenizer.from_pretrained(DEBERTA_MODEL)

    oof_deb, test_deb = run_single_model_cv(
        DEBERTA_MODEL,
        DebertaClassifier,
        tokenizer_deb,
        X_text,
        y_full,
        test_text,
        n_classes,
        "DeBERTa",
    )

    results["deberta_oof"] = oof_deb
    results["deberta_test"] = test_deb

    print(f"DeBERTa OOF Log Loss: {calculate_log_loss(y_full, oof_deb)}")

    # Save DeBERTa results immediately
    np.save(cache_files["deberta_oof"], oof_deb)
    np.save(cache_files["deberta_test"], test_deb)

    # Cleanup DeBERTa specific objects
    del tokenizer_deb, oof_deb, test_deb
    gc.collect()
    torch.cuda.empty_cache()

    return results
