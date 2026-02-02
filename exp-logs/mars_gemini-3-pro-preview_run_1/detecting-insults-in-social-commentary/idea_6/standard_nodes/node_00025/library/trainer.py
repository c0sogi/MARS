import os
import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    get_linear_schedule_with_warmup,
    logging as transformers_logging,
)
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import seed_everything, get_score, AverageMeter
from library.awp import AWP
from library.data_processing import (
    load_data,
    get_structural_features,
    MLMDataset,
    InsultDataset,
)
from library.model import HybridDeberta

# Suppress transformer warnings
transformers_logging.set_verbosity_error()


def run_mlm(train_texts, val_texts, test_texts):
    """
    Stage 1: Domain Adaptive Pre-training using Masked Language Modeling.
    Fine-tunes the backbone on the combined corpus.
    """
    print(f"\n{'='*20} Stage 1: MLM Domain Adaptation {'='*20}")

    # Combine all texts
    all_texts = list(train_texts) + list(val_texts) + list(test_texts)
    print(f"Total MLM corpus size: {len(all_texts)} samples")

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Create Dataset and DataLoader
    dataset = MLMDataset(
        all_texts, tokenizer, max_len=Config.MAX_LEN, mask_prob=Config.MLM_MASK_PROB
    )
    dataloader = DataLoader(
        dataset,
        batch_size=Config.MLM_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model = AutoModelForMaskedLM.from_pretrained(Config.MODEL_NAME)
    model.to(Config.DEVICE)

    optimizer = AdamW(
        model.parameters(), lr=Config.MLM_LR, weight_decay=Config.WEIGHT_DECAY
    )

    model.train()

    for epoch in range(Config.MLM_EPOCHS):
        start_time = time.time()
        losses = AverageMeter()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(Config.DEVICE)
            attention_mask = batch["attention_mask"].to(Config.DEVICE)
            labels = batch["labels"].to(Config.DEVICE)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

            losses.update(loss.item(), input_ids.size(0))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.MLM_EPOCHS} | Loss: {losses.avg:.4f} | Time: {elapsed:.0f}s"
        )

    # Save the fine-tuned backbone
    print(f"Saving MLM fine-tuned model to {Config.MLM_MODEL_PATH}...")
    os.makedirs(Config.MLM_MODEL_PATH, exist_ok=True)
    # We save the underlying transformer so AutoModel.from_pretrained can load it later
    if hasattr(model, "deberta"):
        model.deberta.save_pretrained(Config.MLM_MODEL_PATH)
    elif hasattr(model, "roberta"):
        model.roberta.save_pretrained(Config.MLM_MODEL_PATH)
    elif hasattr(model, "bert"):
        model.bert.save_pretrained(Config.MLM_MODEL_PATH)
    else:
        # Fallback for generic saving (might require careful loading)
        model.save_pretrained(Config.MLM_MODEL_PATH)

    tokenizer.save_pretrained(Config.MLM_MODEL_PATH)

    del model, optimizer, dataloader, dataset
    torch.cuda.empty_cache()
    gc.collect()


def train_fn(
    dataloader,
    model,
    criterion,
    optimizer,
    epoch,
    scheduler,
    device,
    awp=None,
    scaler=None,
):
    model.train()
    losses = AverageMeter()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        svd_features = batch["svd_features"].to(device)
        labels = batch["label"].to(device)

        batch_size = input_ids.size(0)

        # Mixed Precision Training
        with torch.amp.autocast("cuda", enabled=True):
            y_preds = model(input_ids, attention_mask, svd_features)
            loss = criterion(y_preds, labels)

        losses.update(loss.item(), batch_size)

        scaler.scale(loss).backward()

        # AWP Attack
        if Config.AWP_ENABLED and awp is not None:
            # AWP requires a dict of inputs matching model signature
            awp_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "svd_features": svd_features,
            }
            awp.attack_backward(awp_inputs, labels, criterion, epoch)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

    return losses.avg


def valid_fn(dataloader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            svd_features = batch["svd_features"].to(device)
            labels = batch["label"].to(device)

            batch_size = input_ids.size(0)

            with torch.amp.autocast("cuda", enabled=True):
                y_preds = model(input_ids, attention_mask, svd_features)
                loss = criterion(y_preds, labels)

            losses.update(loss.item(), batch_size)

            # Apply sigmoid for probabilities
            preds.append(torch.sigmoid(y_preds).cpu().numpy())
            targets.append(labels.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    return losses.avg, preds, targets


def predict(model, dataloader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            svd_features = batch["svd_features"].to(device)

            with torch.amp.autocast("cuda", enabled=True):
                y_preds = model(input_ids, attention_mask, svd_features)

            preds.append(torch.sigmoid(y_preds).cpu().numpy())

    return np.concatenate(preds)


def run_fold(
    fold,
    train_idx,
    val_idx,
    df,
    svd_features,
    tokenizer,
    test_df,
    test_svd,
    model_path=None,
):
    print(f"\n{'='*10} Fold {fold+1}/{Config.N_FOLDS} {'='*10}")

    # Prepare Data
    train_data = df.iloc[train_idx].reset_index(drop=True)
    val_data = df.iloc[val_idx].reset_index(drop=True)

    train_svd = svd_features[train_idx]
    val_svd = svd_features[val_idx]

    train_dataset = InsultDataset(
        train_data["Comment"].values,
        train_svd,
        tokenizer,
        labels=train_data["Insult"].values,
    )
    val_dataset = InsultDataset(
        val_data["Comment"].values, val_svd, tokenizer, labels=val_data["Insult"].values
    )

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

    # Initialize Model
    if model_path is None:
        # Use MLM weights if available, otherwise default
        model_path = (
            Config.MLM_MODEL_PATH
            if os.path.exists(Config.MLM_MODEL_PATH)
            else Config.MODEL_NAME
        )
    print(f"Initializing model from: {model_path}")
    model = HybridDeberta(pretrained_model_name_or_path=model_path)
    model.to(Config.DEVICE)

    # Optimizer with Differential Learning Rates
    optimizer_parameters = [
        {
            "params": [p for n, p in model.backbone.named_parameters()],
            "lr": Config.LR_BACKBONE,
            "weight_decay": Config.WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "lr": Config.LR_HEAD,
            "weight_decay": Config.WEIGHT_DECAY,
        },
    ]

    optimizer = AdamW(optimizer_parameters)

    # Scheduler
    num_train_steps = int(len(train_data) / Config.TRAIN_BATCH_SIZE * Config.CLS_EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
        num_training_steps=num_train_steps,
    )

    # Loss & Scaler
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda")

    # AWP
    awp = AWP(
        model,
        optimizer,
        adv_lr=Config.AWP_LR,
        adv_eps=Config.AWP_EPS,
        start_epoch=Config.AWP_START_EPOCH,
        scaler=scaler,
    )

    best_score = -np.inf
    best_model_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.bin")

    # Training Loop
    for epoch in range(Config.CLS_EPOCHS):
        start_time = time.time()

        train_loss = train_fn(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            scheduler,
            Config.DEVICE,
            awp,
            scaler,
        )
        val_loss, val_preds, val_labels = valid_fn(
            val_loader, model, criterion, Config.DEVICE
        )

        val_score = get_score(val_labels, val_preds)
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val AUC: {val_score:.10f} | Time: {elapsed:.0f}s"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New Best Score! Model Saved.")

    # Inference on Test Set with Best Model
    model.load_state_dict(torch.load(best_model_path))
    test_dataset = InsultDataset(test_df["Comment"].values, test_svd, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    test_preds = predict(model, test_loader, Config.DEVICE)

    del model, optimizer, scheduler, scaler, awp, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    return test_preds, best_score


def train_and_predict():
    seed_everything(Config.SEED)

    # 1. Load Data
    print("Loading Data...")
    train_df, val_df, test_df = load_data()

    # 2. Structural Features (SVD)
    # Fit on train, transform all
    print("Generating Structural Features...")
    train_svd, val_svd, test_svd = get_structural_features(
        train_df["Comment"].tolist(),
        val_df["Comment"].tolist(),
        test_df["Comment"].tolist(),
        load_cached_data=True,
    )

    # 3. Stage 1: MLM
    # Check if MLM model already exists to skip re-training if possible (optional, but good for restart)
    if not os.path.exists(Config.MLM_MODEL_PATH):
        run_mlm(train_df["Comment"], val_df["Comment"], test_df["Comment"])
    else:
        print(f"MLM model found at {Config.MLM_MODEL_PATH}, skipping Stage 1.")

    # 4. Stage 2: Stratified K-Fold CV
    print(f"\n{'='*20} Stage 2: Supervised Training (5-Fold CV) {'='*20}")

    # Combine Train + Val for CV
    full_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    full_svd = np.concatenate([train_svd, val_svd], axis=0)

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Tokenizer (load from MLM path if available, else base)
    tokenizer_path = (
        Config.MLM_MODEL_PATH
        if os.path.exists(Config.MLM_MODEL_PATH)
        else Config.MODEL_NAME
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    oof_preds = np.zeros(len(full_df))
    test_preds_accum = np.zeros(len(test_df))
    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, full_df["Insult"])):
        fold_test_preds, fold_score = run_fold(
            fold, train_idx, val_idx, full_df, full_svd, tokenizer, test_df, test_svd
        )
        scores.append(fold_score)
        test_preds_accum += fold_test_preds

    avg_score = np.mean(scores)
    print(f"\nAverage CV AUC: {avg_score:.10f}")

    # 5. Submission
    print("Generating Submission...")
    final_test_preds = test_preds_accum / Config.N_FOLDS

    submission_df = pd.DataFrame(
        {
            "id": range(
                len(test_df)
            ),  # Assuming implicit ID based on row index if not provided
            "Insult": final_test_preds,
        }
    )

    # Match sample submission format
    # The sample submission usually has specific columns.
    # Based on task description, we just need predictions in range [0,1].
    # We will save strictly as requested.
    # Check sample submission format if available, but here we construct based on test.csv length.
    # The provided sample_submission_null.csv has columns: |    |   Insult | Date | Comment |
    # Usually submission files for such tasks require an ID or just the target column aligned with test.csv.
    # Given the instructions "Your predictions should be a number in the range [0,1]. See 'sample_submissions_null.csv' for the correct format."
    # And sample_submission_null.csv has the same columns as test.csv plus 'Insult'.
    # We will create a dataframe with the same structure as sample_submission_null.csv

    # Re-read sample submission to be sure of format
    sample_sub_path = os.path.join("./input", "sample_submission_null.csv")
    if os.path.exists(sample_sub_path):
        sample_df = pd.read_csv(sample_sub_path)
        if "Insult" in sample_df.columns:
            sample_df["Insult"] = final_test_preds
            submission_df = sample_df

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
