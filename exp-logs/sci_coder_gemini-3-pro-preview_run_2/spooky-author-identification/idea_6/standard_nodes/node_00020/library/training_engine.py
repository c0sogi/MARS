import os
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data_factory import AuthorDataset, get_tokenizer, LABEL_MAP
from library.neural_model import DebertaMultiView


def _evaluate_neural(model, loader, device):
    """
    Helper function to evaluate the neural model on a dataloader.
    Returns average loss (if labels exist) and probability predictions.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            labels = None
            if "label" in batch:
                labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, labels=labels)

            if labels is not None:
                total_loss += outputs["loss"].item()

            # Apply Softmax to logits to get probabilities
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_preds.append(probs)

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    predictions = np.concatenate(all_preds, axis=0)

    return avg_loss, predictions


def train_classical_fold(
    fold_idx,
    X_tfidf_train,
    y_train,
    X_tfidf_val,
    y_val,
    X_tfidf_test,
    X_svd_train,
    X_svd_val,
    X_svd_test,
):
    """
    Trains classical models (LR, NB, XGB) for a single fold.

    Args:
        fold_idx (int): Current fold number.
        X_tfidf_*: Sparse TF-IDF matrices for LR and NB.
        X_svd_*: Dense SVD arrays for XGBoost.
        y_*: Label arrays.

    Returns:
        dict: Dictionary containing 'val' (OOF) and 'test' predictions for each model.
    """
    seed_everything(Config.SEED)
    results = {}

    # --- 1. Logistic Regression ---
    print(f"[Fold {fold_idx}] Training Logistic Regression...")
    lr_model = LogisticRegression(
        C=1.0,
        solver="sag",
        multi_class="multinomial",
        n_jobs=Config.NUM_WORKERS,
        random_state=Config.SEED,
        max_iter=1000,
    )
    lr_model.fit(X_tfidf_train, y_train)

    val_preds_lr = lr_model.predict_proba(X_tfidf_val)
    test_preds_lr = lr_model.predict_proba(X_tfidf_test)

    score_lr = calculate_metric(y_val, val_preds_lr)
    print(f"[Fold {fold_idx}] LR Validation Loss: {score_lr}")
    results["lr"] = {"val": val_preds_lr, "test": test_preds_lr}

    # --- 2. Multinomial Naive Bayes ---
    print(f"[Fold {fold_idx}] Training Naive Bayes...")
    nb_model = MultinomialNB(alpha=0.01)
    nb_model.fit(X_tfidf_train, y_train)

    val_preds_nb = nb_model.predict_proba(X_tfidf_val)
    test_preds_nb = nb_model.predict_proba(X_tfidf_test)

    score_nb = calculate_metric(y_val, val_preds_nb)
    print(f"[Fold {fold_idx}] NB Validation Loss: {score_nb}")
    results["nb"] = {"val": val_preds_nb, "test": test_preds_nb}

    # --- 3. XGBoost ---
    print(f"[Fold {fold_idx}] Training XGBoost...")
    # Using CPU for XGBoost to avoid resource contention with PyTorch if run in parallel,
    # and because dataset size is small enough for CPU to be efficient.
    xgb_model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=Config.NUM_LABELS,
        n_jobs=Config.NUM_WORKERS,
        random_state=Config.SEED,
        tree_method="hist",
        device="cpu",
        early_stopping_rounds=50,
    )

    xgb_model.fit(X_svd_train, y_train, eval_set=[(X_svd_val, y_val)], verbose=False)

    val_preds_xgb = xgb_model.predict_proba(X_svd_val)
    test_preds_xgb = xgb_model.predict_proba(X_svd_test)

    score_xgb = calculate_metric(y_val, val_preds_xgb)
    print(f"[Fold {fold_idx}] XGB Validation Loss: {score_xgb}")
    results["xgb"] = {"val": val_preds_xgb, "test": test_preds_xgb}

    return results


def train_neural_fold(fold_idx, train_df, val_df, test_df):
    """
    Trains the DebertaMultiView model for a single fold.

    Args:
        fold_idx (int): Current fold number.
        train_df, val_df, test_df (pd.DataFrame): Data splits.

    Returns:
        tuple: (oof_predictions, test_predictions)
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # 1. Data Preparation
    tokenizer = get_tokenizer()

    train_dataset = AuthorDataset(train_df, tokenizer, Config.MAX_LENGTH, is_test=False)
    val_dataset = AuthorDataset(val_df, tokenizer, Config.MAX_LENGTH, is_test=False)
    test_dataset = AuthorDataset(test_df, tokenizer, Config.MAX_LENGTH, is_test=True)

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

    # 2. Model Setup
    model = DebertaMultiView()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01)

    # Calculate training steps for scheduler
    num_update_steps_per_epoch = len(train_loader) // Config.GRADIENT_ACCUMULATION_STEPS
    num_training_steps = num_update_steps_per_epoch * Config.EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    # 3. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, f"deberta_fold_{fold_idx}.pth")

    print(f"[Fold {fold_idx}] Starting Neural Training...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs["loss"]

            # Normalize loss for gradient accumulation
            loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            if (step + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * Config.GRADIENT_ACCUMULATION_STEPS

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        val_loss, val_preds = _evaluate_neural(model, val_loader, device)
        val_labels = val_df["author"].map(LABEL_MAP).values
        metric_score = calculate_metric(val_labels, val_preds)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss} | Val Loss: {val_loss} | Metric: {metric_score}"
        )

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # 4. Final Inference
    print(f"[Fold {fold_idx}] Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    _, oof_preds = _evaluate_neural(model, val_loader, device)
    _, test_preds = _evaluate_neural(model, test_loader, device)

    # Cleanup to free GPU memory
    del model, optimizer, scheduler, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()

    return oof_preds, test_preds
