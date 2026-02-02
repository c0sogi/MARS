import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import seed_everything, compute_qwk, get_optimizer_params
from library.data import EssayDataset, Collate, get_test_data
from library.modeling import EssayScorer


def train_one_fold(fold, train_loader, val_loader, tokenizer, model_path=None):
    """
    Executes the training and validation loop for a single fold.

    Args:
        fold (int): The current fold index.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        tokenizer (PreTrainedTokenizer): Tokenizer for the model.
        model_path (str, optional): Path to a pre-trained model checkpoint (e.g., from MLM).

    Returns:
        float: The best QWK score achieved on the validation set.
    """
    print(f"\n=== Training Fold {fold} ===")

    # 1. Initialize Model
    # Use the provided model path (e.g., MLM adapted) or the base configuration
    backbone_path = model_path if model_path else Config.model_name
    model = EssayScorer(model_name_or_path=backbone_path, pretrained=True)
    model.to(Config.device)

    # 2. Optimizer with LLRD
    optimizer_parameters = get_optimizer_params(
        model,
        encoder_lr=Config.learning_rate,
        decoder_lr=Config.head_lr,
        weight_decay=Config.weight_decay,
    )
    optimizer = torch.optim.AdamW(optimizer_parameters)

    # 3. Scheduler
    # Calculate total training steps
    num_train_steps = int(
        len(train_loader) / Config.gradient_accumulation_steps * Config.epochs
    )
    num_warmup_steps = int(num_train_steps * Config.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps,
    )

    # 4. Loss and Scaler
    criterion = nn.MSELoss()
    scaler = GradScaler(enabled=Config.use_fp16)

    # Tracking
    best_qwk = -1.0
    save_path = os.path.join(Config.output_dir, f"model_fold_{fold}.pth")

    for epoch in range(Config.epochs):
        # --- Training Loop ---
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)
            labels = batch["labels"].to(Config.device)

            with autocast(enabled=Config.use_fp16):
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                if Config.gradient_accumulation_steps > 1:
                    loss = loss / Config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % Config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * Config.gradient_accumulation_steps

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        preds = []
        true_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(Config.device)
                attention_mask = batch["attention_mask"].to(Config.device)
                labels = batch["labels"].to(Config.device)

                with autocast(enabled=Config.use_fp16):
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds.append(outputs.detach().cpu().numpy())
                true_labels.append(labels.detach().cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        preds = np.concatenate(preds)
        true_labels = np.concatenate(true_labels)

        # Compute QWK
        qwk = compute_qwk(true_labels, preds)

        print(
            f"Epoch {epoch + 1}/{Config.epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"Val QWK: {qwk:.6f}"
        )

        # Save Best Model
        if qwk > best_qwk:
            best_qwk = qwk
            print(f"New Best QWK! Saving model to {save_path}")
            torch.save(model.state_dict(), save_path)

    # Cleanup
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    gc.collect()

    return best_qwk


def run_cross_validation(mlm_model_path=None):
    """
    Orchestrates the Stratified K-Fold Cross-Validation training process.

    Args:
        mlm_model_path (str, optional): Path to the domain-adapted model checkpoint.
    """
    seed_everything(Config.seed)
    print("Starting Cross-Validation Training...")

    # 1. Load and Combine Metadata
    if not os.path.exists(Config.train_path) or not os.path.exists(Config.val_path):
        raise FileNotFoundError("Metadata files not found in ./metadata/")

    df_train_meta = pd.read_csv(Config.train_path)
    df_val_meta = pd.read_csv(Config.val_path)

    # Concatenate to perform a fresh Stratified K-Fold split
    df_full = pd.concat([df_train_meta, df_val_meta], axis=0).reset_index(drop=True)

    # 2. Setup Folds
    skf = StratifiedKFold(
        n_splits=Config.num_folds, shuffle=True, random_state=Config.seed
    )
    df_full["fold"] = -1
    # Stratify by score
    for fold, (_, val_idx) in enumerate(skf.split(df_full, df_full["score"])):
        df_full.loc[val_idx, "fold"] = fold

    # 3. Initialize Tokenizer
    tokenizer_path = mlm_model_path if mlm_model_path else Config.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    collate_fn = Collate(tokenizer)

    cv_scores = []
    model_paths = []

    # 4. Iterate Folds
    for fold in range(Config.num_folds):
        # Subset Data
        df_train = df_full[df_full["fold"] != fold].reset_index(drop=True)
        df_val = df_full[df_full["fold"] == fold].reset_index(drop=True)

        if Config.debug:
            print("Debug mode: using subset of data")
            df_train = df_train.iloc[:100]
            df_val = df_val.iloc[:100]

        # Create Datasets
        train_dataset = EssayDataset(
            df_train, tokenizer, max_length=Config.max_length, include_labels=True
        )
        val_dataset = EssayDataset(
            df_val, tokenizer, max_length=Config.max_length, include_labels=True
        )

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=Config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Train
        best_qwk = train_one_fold(
            fold, train_loader, val_loader, tokenizer, model_path=mlm_model_path
        )
        cv_scores.append(best_qwk)
        model_paths.append(os.path.join(Config.output_dir, f"model_fold_{fold}.pth"))

    print(f"\nCV QWK Scores: {cv_scores}")
    print(f"Mean CV QWK: {np.mean(cv_scores)}")

    return model_paths


def inference(model_paths):
    """
    Generates predictions for the test set using an ensemble of trained models.

    Args:
        model_paths (list): List of paths to the trained model checkpoints.
    """
    print("\n=== Starting Inference ===")
    seed_everything(Config.seed)

    # 1. Load Test Data
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    test_dataset = get_test_data(tokenizer, load_cached_data=True)
    collate_fn = Collate(tokenizer)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # 2. Ensemble Prediction
    # Initialize array to store sum of predictions
    final_preds = np.zeros(len(test_dataset))

    for fold, path in enumerate(model_paths):
        print(f"Predicting with model fold {fold}: {path}")

        # Load Model
        model = EssayScorer(model_name_or_path=Config.model_name, pretrained=False)
        state_dict = torch.load(path, map_location=Config.device)
        model.load_state_dict(state_dict)
        model.to(Config.device)
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(Config.device)
                attention_mask = batch["attention_mask"].to(Config.device)

                with autocast(enabled=Config.use_fp16):
                    outputs = model(input_ids, attention_mask)

                fold_preds.append(outputs.detach().cpu().numpy())

        fold_preds = np.concatenate(fold_preds)
        final_preds += fold_preds

        # Cleanup
        del model, state_dict
        torch.cuda.empty_cache()
        gc.collect()

    # 3. Average Predictions
    avg_preds = final_preds / len(model_paths)

    # 4. Post-processing (Clip and Round)
    # Map continuous regression output to 1-6 integer scale
    final_scores = np.rint(np.clip(avg_preds, 1, 6)).astype(int)

    # 5. Create Submission File
    df_test = pd.read_csv(Config.test_path)
    submission = pd.DataFrame({"essay_id": df_test["essay_id"], "score": final_scores})

    # Save
    sub_path = os.path.join(Config.output_dir, "submission.csv")
    # Also save to the root directory if needed by the competition platform
    # but strictly following the prompt, we save to output_dir or specified location.
    # The prompt mentions sample_submission.csv in input, but we should output to working.
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")

    # Also save to ./submission.csv as a fallback for some environments
    submission.to_csv("submission.csv", index=False)
