import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
import gc

from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data_processing import load_data, InsultDataset
from library.model import InsultDetector
from library.awp import AWP


class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()

        # Differential Learning Rates
        self.optimizer = self._get_optimizer()

        # Scheduler
        num_train_steps = int(len(self.train_loader) * Config.epochs)
        num_warmup_steps = int(num_train_steps * Config.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )

        # AWP
        self.awp = AWP(
            self.model, self.optimizer, adv_lr=Config.awp_lr, adv_eps=Config.awp_eps
        )
        self.use_awp = Config.use_awp

    def _get_optimizer(self):
        # Separate backbone and head parameters
        backbone_params = list(self.model.backbone.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in backbone_params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": Config.weight_decay,
                "lr": Config.lr_backbone,
            },
            {
                "params": [
                    p for n, p in backbone_params if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": Config.lr_backbone,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "backbone" not in n and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": Config.weight_decay,
                "lr": Config.lr_head,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "backbone" not in n and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": Config.lr_head,
            },
        ]
        return AdamW(optimizer_grouped_parameters)

    def train_epoch(self, epoch):
        self.model.train()
        losses = []

        # Enable AWP if epoch condition met
        do_awp = self.use_awp and (epoch >= Config.awp_start_epoch)

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            svd_features = batch["svd_features"].to(self.device)
            labels = batch["label"].to(self.device).unsqueeze(1)

            # 1. Standard Forward Pass
            outputs = self.model(input_ids, attention_mask, svd_features)
            loss = self.criterion(outputs, labels)

            # 2. Standard Backward
            loss.backward()

            # 3. AWP Attack & Backward
            if do_awp:
                self.awp.attack()
                # Forward pass with perturbed weights
                adv_outputs = self.model(input_ids, attention_mask, svd_features)
                adv_loss = self.criterion(adv_outputs, labels)
                adv_loss.backward()
                self.awp.restore()

            # 4. Optimization
            nn.utils.clip_grad_norm_(self.model.parameters(), Config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            losses.append(loss.item())

        return np.mean(losses)

    def valid_epoch(self):
        self.model.eval()
        losses = []
        preds = []
        targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                svd_features = batch["svd_features"].to(self.device)
                labels = batch["label"].to(self.device).unsqueeze(1)

                outputs = self.model(input_ids, attention_mask, svd_features)
                loss = self.criterion(outputs, labels)

                losses.append(loss.item())
                preds.append(torch.sigmoid(outputs).cpu().numpy())
                targets.append(labels.cpu().numpy())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        auc = calculate_metric(targets, preds)

        return np.mean(losses), auc

    def fit(self, fold_idx):
        best_auc = 0.0
        patience_counter = 0
        best_model_path = os.path.join(Config.working_dir, f"model_fold_{fold_idx}.bin")

        print(f"Starting training for Fold {fold_idx}...")

        for epoch in range(Config.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_auc = self.valid_epoch()

            print(
                f"Epoch {epoch+1}/{Config.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(self.model.state_dict(), best_model_path)
                patience_counter = 0
                print(f"  -> New best model saved! AUC: {val_auc}")
            else:
                patience_counter += 1

            if patience_counter >= Config.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        return best_model_path


def predict(model, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = (
                batch["input_ids"].to(self.device)
                if hasattr(self, "device")
                else batch["input_ids"].to(device)
            )
            attention_mask = (
                batch["attention_mask"].to(self.device)
                if hasattr(self, "device")
                else batch["attention_mask"].to(device)
            )
            svd_features = (
                batch["svd_features"].to(self.device)
                if hasattr(self, "device")
                else batch["svd_features"].to(device)
            )

            outputs = model(input_ids, attention_mask, svd_features)
            preds.append(torch.sigmoid(outputs).cpu().numpy())

    return np.concatenate(preds)


def run_training():
    seed_everything(Config.seed)

    # 1. Load Data
    # We load the split metadata but combine them for full CV
    train_ds_part, val_ds_part, test_ds = load_data(load_cached_data=True)

    # Combine train and val for StratifiedKFold
    all_texts = np.concatenate([train_ds_part.texts, val_ds_part.texts])
    all_svd = np.concatenate([train_ds_part.svd_features, val_ds_part.svd_features])
    all_labels = np.concatenate([train_ds_part.labels, val_ds_part.labels])

    print(f"Combined Training Data Shape: {all_texts.shape}")

    # 2. Prepare Test Loader
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # 3. Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    # Array to store test predictions from each fold
    test_preds_accumulator = np.zeros((len(test_ds), 1))

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_texts, all_labels)):
        print(f"\n{'='*20} Fold {fold} {'='*20}")

        # Create Datasets for this fold
        train_fold_ds = InsultDataset(
            texts=all_texts[train_idx],
            svd_features=all_svd[train_idx],
            labels=all_labels[train_idx],
            tokenizer=train_ds_part.tokenizer,
            max_len=Config.max_len,
        )

        val_fold_ds = InsultDataset(
            texts=all_texts[val_idx],
            svd_features=all_svd[val_idx],
            labels=all_labels[val_idx],
            tokenizer=train_ds_part.tokenizer,
            max_len=Config.max_len,
        )

        # Create Loaders
        train_loader = DataLoader(
            train_fold_ds,
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_fold_ds,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Initialize Model
        model = InsultDetector()
        model.to(Config.device)

        # Initialize Trainer
        trainer = Trainer(model, train_loader, val_loader, Config.device)

        # Train
        best_model_path = trainer.fit(fold)

        # Load Best Model for Inference
        print(f"Loading best model for Fold {fold} inference...")
        model.load_state_dict(torch.load(best_model_path, map_location=Config.device))
        model.to(Config.device)

        # Predict on Test
        fold_preds = predict(model, test_loader, Config.device)
        test_preds_accumulator += fold_preds

        # Cleanup
        del model, trainer, train_loader, val_loader, train_fold_ds, val_fold_ds
        torch.cuda.empty_cache()
        gc.collect()

    # 4. Average Predictions
    avg_preds = test_preds_accumulator / Config.n_folds

    # 5. Save Submission
    print("Saving submission...")
    # Load sample submission to preserve format
    try:
        sub_df = pd.read_csv(Config.sample_submission_path)
    except FileNotFoundError:
        # Fallback if sample submission not found, create from test.csv
        test_df = pd.read_csv(Config.raw_test_path)
        sub_df = pd.DataFrame()
        # Assuming sample submission has 'Insult' column and maybe ID/Date/Comment
        # Based on task description, sample_submission_null.csv has 3 columns: Insult, Date, Comment
        # We need to fill 'Insult' column.
        sub_df = test_df.copy()
        if "Insult" not in sub_df.columns:
            sub_df["Insult"] = 0

    # Ensure lengths match
    if len(sub_df) != len(avg_preds):
        print(
            f"Warning: Submission length {len(sub_df)} != Prediction length {len(avg_preds)}"
        )

    sub_df["Insult"] = avg_preds

    # Save
    sub_df.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
