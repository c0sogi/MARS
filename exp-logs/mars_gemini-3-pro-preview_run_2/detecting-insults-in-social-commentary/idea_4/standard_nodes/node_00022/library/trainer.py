import os
import gc
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import seed_everything, get_score
from library.dataset import InsultDataset, get_tokenizer, load_dataset_dataframe
from library.model import InsultModel
from library.awp import AWP


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    """
    Configures layer-wise learning rate decay (LLRD) for the optimizer.
    Assigns higher LR to the head and top layers, and lower LR to bottom layers.
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = []

    # 1. Head Parameters (Classifier, Pooler, etc.) - Use decoder_lr
    head_params = []
    for name, param in model.named_parameters():
        if "backbone" not in name:
            head_params.append((name, param))

    if head_params:
        optimizer_parameters.append(
            {
                "params": [
                    p for n, p in head_params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
                "lr": decoder_lr,
            }
        )
        optimizer_parameters.append(
            {
                "params": [
                    p for n, p in head_params if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": decoder_lr,
            }
        )

    # 2. Backbone Layers - Use decaying encoder_lr
    # We assume a standard structure with 'backbone.encoder.layer.{i}'
    # Large models typically have 24 layers. We'll iterate backwards.
    num_layers = 24

    for layer_i in range(num_layers - 1, -1, -1):
        # Calculate decayed LR: lr * (decay ^ depth_from_top)
        layer_lr = encoder_lr * (Config.llrd_decay ** (num_layers - 1 - layer_i))

        layer_params = []
        for name, param in model.named_parameters():
            if f"encoder.layer.{layer_i}." in name:
                layer_params.append((name, param))

        if layer_params:
            optimizer_parameters.append(
                {
                    "params": [
                        p
                        for n, p in layer_params
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": weight_decay,
                    "lr": layer_lr,
                }
            )
            optimizer_parameters.append(
                {
                    "params": [
                        p for n, p in layer_params if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                    "lr": layer_lr,
                }
            )

    # 3. Embeddings - Lowest LR
    embed_lr = encoder_lr * (Config.llrd_decay**num_layers)
    embed_params = []
    for name, param in model.named_parameters():
        if "embeddings" in name and "backbone" in name:
            embed_params.append((name, param))

    if embed_params:
        optimizer_parameters.append(
            {
                "params": [
                    p for n, p in embed_params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
                "lr": embed_lr,
            }
        )
        optimizer_parameters.append(
            {
                "params": [
                    p for n, p in embed_params if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": embed_lr,
            }
        )

    return optimizer_parameters


def train_one_epoch(epoch, model, train_loader, optimizer, scheduler, device, awp=None):
    """
    Trains the model for one epoch, integrating Adversarial Weight Perturbation (AWP).
    """
    model.train()
    scaler = GradScaler(enabled=Config.use_fp16)

    dataset_size = 0
    running_loss = 0.0
    start = time.time()

    for step, data in enumerate(train_loader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        targets = data["target"].to(device)

        batch_size = input_ids.size(0)

        # 1. Standard Forward Pass
        with autocast(enabled=Config.use_fp16):
            outputs = model(input_ids, attention_mask)
            loss = nn.BCEWithLogitsLoss()(outputs.view(-1), targets.view(-1))

        # 2. Standard Backward
        scaler.scale(loss).backward()

        # 3. AWP Attack (if enabled and active for this epoch)
        if awp is not None and epoch >= Config.awp_start_epoch:
            awp.attack()
            with autocast(enabled=Config.use_fp16):
                # Re-compute forward pass with perturbed weights
                outputs_adv = model(input_ids, attention_mask)
                loss_adv = nn.BCEWithLogitsLoss()(
                    outputs_adv.view(-1), targets.view(-1)
                )

            # Backward on adversarial loss
            scaler.scale(loss_adv).backward()

            # Restore original weights
            awp.restore()

        # 4. Optimization Step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    print(
        f"Train Epoch {epoch} | Loss: {epoch_loss:.5f} | Time: {time.time() - start:.1f}s"
    )
    return epoch_loss


def valid_one_epoch(model, val_loader, device):
    """
    Validates the model and calculates AUC.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    preds = []
    targets_list = []
    start = time.time()

    with torch.no_grad():
        for step, data in enumerate(val_loader):
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            targets = data["target"].to(device)

            batch_size = input_ids.size(0)

            with autocast(enabled=Config.use_fp16):
                outputs = model(input_ids, attention_mask)
                loss = nn.BCEWithLogitsLoss()(outputs.view(-1), targets.view(-1))

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            preds.append(torch.sigmoid(outputs).view(-1).cpu().numpy())
            targets_list.append(targets.view(-1).cpu().numpy())

    epoch_loss = running_loss / dataset_size
    all_preds = np.concatenate(preds)
    all_targets = np.concatenate(targets_list)

    score = get_score(all_targets, all_preds)
    print(
        f"Valid Loss: {epoch_loss:.5f} | AUC: {score} | Time: {time.time() - start:.1f}s"
    )

    return epoch_loss, score


def run_fold(fold, df, train_idx, val_idx, model_name):
    """
    Orchestrates training for a specific fold and model architecture.
    """
    # 1. Prepare Data
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    tokenizer = get_tokenizer(model_name)

    train_dataset = InsultDataset(train_df, tokenizer, Config.max_len)
    val_dataset = InsultDataset(val_df, tokenizer, Config.max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size * 2,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # 2. Initialize Model
    device = Config.device
    model = InsultModel(model_name, Config)
    model.to(device)

    # 3. Optimizer & Scheduler with LLRD
    optimizer_grouped_parameters = get_optimizer_params(
        model,
        encoder_lr=Config.lr,
        decoder_lr=Config.lr,
        weight_decay=Config.weight_decay,
    )
    optimizer = AdamW(optimizer_grouped_parameters, lr=Config.lr, eps=1e-6)

    num_train_steps = int(len(train_df) / Config.batch_size * Config.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
        num_training_steps=num_train_steps,
    )

    # 4. Initialize AWP
    awp = None
    if Config.use_awp:
        awp = AWP(
            model,
            optimizer,
            adv_lr=Config.awp_lr,
            adv_eps=Config.awp_eps,
            start_epoch=Config.awp_start_epoch,
        )

    # 5. Training Loop
    best_score = -np.inf
    model_name_safe = model_name.replace("/", "_")
    best_model_path = os.path.join(
        Config.working_dir, f"{model_name_safe}_fold_{fold}.pth"
    )

    for epoch in range(1, Config.epochs + 1):
        print(f"\nEpoch {epoch}/{Config.epochs} (Fold {fold}, Model {model_name})")

        train_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, scheduler, device, awp
        )
        val_loss, val_score = valid_one_epoch(model, val_loader, device)

        # Save best model based on AUC
        if val_score > best_score:
            print(f"Score Improved: {best_score} -> {val_score}. Saving model...")
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)

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
    gc.collect()
    torch.cuda.empty_cache()

    return best_score


def predict_test(model_name, test_df, fold):
    """
    Generates predictions for the test set using a specific trained model.
    """
    device = Config.device
    tokenizer = get_tokenizer(model_name)

    test_dataset = InsultDataset(test_df, tokenizer, Config.max_len, is_test=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size * 2,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    model = InsultModel(model_name, Config)
    model_name_safe = model_name.replace("/", "_")
    weight_path = os.path.join(Config.working_dir, f"{model_name_safe}_fold_{fold}.pth")

    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for data in test_loader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)

            with autocast(enabled=Config.use_fp16):
                outputs = model(input_ids, attention_mask)

            preds.append(torch.sigmoid(outputs).view(-1).cpu().numpy())

    del model, test_loader, test_dataset
    gc.collect()
    torch.cuda.empty_cache()

    return np.concatenate(preds)


def train_and_predict():
    """
    Main function to execute the full training and inference pipeline.
    """
    seed_everything(Config.seed)

    # 1. Load Data
    # Combine train and val metadata to perform custom K-Fold
    train_df_part = load_dataset_dataframe(Config.train_path, "train_cleaned")
    val_df_part = load_dataset_dataframe(Config.val_path, "val_cleaned")
    full_train_df = pd.concat([train_df_part, val_df_part]).reset_index(drop=True)

    test_df = load_dataset_dataframe(Config.test_path, "test_cleaned")

    # 2. Configure Cross-Validation
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    test_preds_accum = np.zeros(len(test_df))
    total_models = len(Config.model_backbones) * Config.n_folds

    # 3. Train Heterogeneous Ensemble
    for model_name in Config.model_backbones:
        print(f"\n{'='*40}\nTraining Backbone: {model_name}\n{'='*40}")

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(full_train_df, full_train_df["Insult"])
        ):
            print(f"\n--- Fold {fold} ---")

            # Train
            run_fold(fold, full_train_df, train_idx, val_idx, model_name)

            # Predict
            fold_preds = predict_test(model_name, test_df, fold)
            test_preds_accum += fold_preds

    # 4. Average Predictions
    avg_preds = test_preds_accum / total_models

    # 5. Generate Submission
    sample_sub_path = os.path.join(Config.input_dir, "sample_submission_null.csv")
    if os.path.exists(sample_sub_path):
        submission = pd.read_csv(sample_sub_path)
        if "Unnamed: 0" in submission.columns:
            submission = submission.drop(columns=["Unnamed: 0"])
    else:
        submission = pd.DataFrame()

    # Assign predictions
    submission["Insult"] = avg_preds

    # Save
    submission.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
