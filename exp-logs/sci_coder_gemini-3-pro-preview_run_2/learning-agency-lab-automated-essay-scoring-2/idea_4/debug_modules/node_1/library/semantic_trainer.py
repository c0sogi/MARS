import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoTokenizer

from library.config import Config
from library.utils import seed_everything, compute_qwk
from library.data import load_dataset, get_folds, EssayDataset
from library.model_semantic import build_model, get_optimizer, get_loss_fn


def train_one_epoch(model, optimizer, scheduler, dataloader, device, scaler):
    """
    Trains the model for one epoch using mixed precision and gradient accumulation.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Set gradients to zero initially
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["labels"].to(device)

        batch_size = input_ids.size(0)

        # Mixed precision forward pass
        with torch.cuda.amp.autocast(enabled=Config.USE_FP16):
            outputs = model(input_ids, attention_mask)
            # Flatten outputs and targets for loss calculation
            loss = get_loss_fn()(outputs.view(-1), targets.view(-1))
            # Scale loss for gradient accumulation
            loss = loss / Config.GRADIENT_ACCUMULATION_STEPS

        # Backward pass with scaling
        scaler.scale(loss).backward()

        # Optimizer step (only after accumulation steps)
        if (step + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        # Track loss (scale back up for reporting)
        running_loss += loss.item() * Config.GRADIENT_ACCUMULATION_STEPS * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def valid_one_epoch(model, dataloader, device):
    """
    Validates the model on the validation set.
    Returns average loss, QWK score, and raw predictions.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    preds = []
    targets_list = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Store targets if available
            if "labels" in batch:
                targets = batch["labels"].to(device)
                targets_list.extend(targets.cpu().numpy())

            # Mixed precision inference
            with torch.cuda.amp.autocast(enabled=Config.USE_FP16):
                outputs = model(input_ids, attention_mask)

                if "labels" in batch:
                    loss = get_loss_fn()(outputs.view(-1), targets.view(-1))
                    running_loss += loss.item() * input_ids.size(0)
                    dataset_size += input_ids.size(0)

            # Collect predictions
            preds.extend(outputs.view(-1).cpu().numpy())

    # Calculate metrics
    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    # Calculate QWK if targets are available
    qwk = 0.0
    if targets_list:
        # Clip predictions to valid range and round for QWK
        preds_arr = np.array(preds)
        preds_clipped = np.clip(preds_arr, 1, 6)
        preds_rounded = np.round(preds_clipped).astype(int)
        targets_arr = np.array(targets_list).astype(int)
        qwk = compute_qwk(targets_arr, preds_rounded)

    return epoch_loss, qwk, np.array(preds)


def predict(model, df, tokenizer, device, batch_size=Config.VALID_BATCH_SIZE):
    """
    Helper function to generate predictions for a dataframe.
    """
    dataset = EssayDataset(df, is_test=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    _, _, preds = valid_one_epoch(model, dataloader, device)
    return preds


def run_fold(fold, df_train_fold, df_val_fold, tokenizer, device, debug=False):
    """
    Trains the model for a single fold.
    """
    print(f"\n=== Training Fold {fold} ===")

    # Create Datasets
    train_dataset = EssayDataset(df_train_fold, is_test=False)
    val_dataset = EssayDataset(df_val_fold, is_test=False)

    # Create DataLoaders
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

    # Initialize Model
    model = build_model()
    model.to(device)

    # Optimizer and Scheduler
    optimizer = get_optimizer(model)

    num_epochs = Config.NUM_EPOCHS
    if debug:
        num_epochs = 2

    num_training_steps = (
        len(train_loader) * num_epochs
    ) // Config.GRADIENT_ACCUMULATION_STEPS
    num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=Config.USE_FP16)

    # Training Loop
    best_loss = float("inf")
    best_qwk = -1.0
    patience_counter = 0

    model_save_path = os.path.join(Config.MODEL_OUTPUT_DIR, f"deberta_fold_{fold}.bin")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, scaler
        )
        val_loss, val_qwk, _ = valid_one_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val QWK: {val_qwk:.6f}"
        )

        # Save best model based on Loss (Regression)
        if val_loss < best_loss:
            best_loss = val_loss
            best_qwk = val_qwk
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Saved best model (Loss: {best_loss:.6f})")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Load best model for inference
    print(f"Loading best model from {model_save_path}...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    # Generate OOF predictions for this fold
    _, _, oof_preds = valid_one_epoch(model, val_loader, device)

    return model, oof_preds


def run_semantic_training(debug=False, load_cached_data=True):
    """
    Main function to run the semantic branch training pipeline.
    """
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Load Data
    print("Loading datasets...")
    df_train_full = load_dataset("train", tokenizer, load_cached_data, debug)
    df_val_meta = load_dataset("val", tokenizer, load_cached_data, debug)
    df_test_meta = load_dataset("test", tokenizer, load_cached_data, debug)

    # Generate Folds on the Training set
    df_train_full = get_folds(df_train_full, n_folds=Config.N_FOLDS)

    # Prepare arrays for results
    oof_preds = np.zeros(len(df_train_full))
    val_meta_preds = np.zeros((len(df_val_meta), Config.N_FOLDS))
    test_meta_preds = np.zeros((len(df_test_meta), Config.N_FOLDS))

    # Loop over folds
    for fold in range(Config.N_FOLDS):
        # Split Data
        train_idx = df_train_full["fold"] != fold
        val_idx = df_train_full["fold"] == fold

        df_train_fold = df_train_full[train_idx].reset_index(drop=True)
        df_val_fold = df_train_full[val_idx].reset_index(drop=True)

        # Train
        model, fold_oof_preds = run_fold(
            fold, df_train_fold, df_val_fold, tokenizer, device, debug
        )

        # Store OOF predictions
        oof_preds[val_idx] = fold_oof_preds

        # Predict on Meta Validation Set (Holdout)
        print(f"Predicting on holdout validation set (Fold {fold})...")
        val_meta_preds[:, fold] = predict(model, df_val_meta, tokenizer, device)

        # Predict on Test Set
        print(f"Predicting on test set (Fold {fold})...")
        test_meta_preds[:, fold] = predict(model, df_test_meta, tokenizer, device)

        # Cleanup to save VRAM
        del model, df_train_fold, df_val_fold
        torch.cuda.empty_cache()
        gc.collect()

    # Aggregate Predictions
    df_train_full["semantic_pred"] = oof_preds
    df_val_meta["semantic_pred"] = val_meta_preds.mean(axis=1)
    df_test_meta["semantic_pred"] = test_meta_preds.mean(axis=1)

    # Clip predictions to valid range
    df_train_full["semantic_pred"] = np.clip(df_train_full["semantic_pred"], 1, 6)
    df_val_meta["semantic_pred"] = np.clip(df_val_meta["semantic_pred"], 1, 6)
    df_test_meta["semantic_pred"] = np.clip(df_test_meta["semantic_pred"], 1, 6)

    # Calculate Overall OOF Score
    oof_qwk = compute_qwk(
        df_train_full["score"].values,
        np.round(df_train_full["semantic_pred"]).astype(int),
    )
    print(f"\nOverall OOF QWK: {oof_qwk:.6f}")

    # Calculate Holdout Validation Score
    val_qwk = compute_qwk(
        df_val_meta["score"].values, np.round(df_val_meta["semantic_pred"]).astype(int)
    )
    print(f"Holdout Validation QWK: {val_qwk:.6f}")

    # Save predictions to disk for stacking
    print("Saving predictions to disk...")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    df_train_full.to_parquet(
        os.path.join(Config.WORKING_DIR, "train_semantic_preds.parquet"), index=False
    )
    df_val_meta.to_parquet(
        os.path.join(Config.WORKING_DIR, "val_semantic_preds.parquet"), index=False
    )
    df_test_meta.to_parquet(
        os.path.join(Config.WORKING_DIR, "test_semantic_preds.parquet"), index=False
    )

    return df_train_full, df_val_meta, df_test_meta
