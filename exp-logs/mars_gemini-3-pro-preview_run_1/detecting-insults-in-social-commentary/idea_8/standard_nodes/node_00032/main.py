import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, logging

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_device
from library.feature_engineering import StructuralFeatureGenerator
from library.dataset import InsultDataset
from library.model import HybridDebertaModel
from library.engine import train_fn, valid_fn, inference_fn
import library.engine  # To monkeypatch tqdm


# =============================================================================
# Monkeypatch tqdm to suppress progress bars as per requirements
# =============================================================================
class DummyTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        if self.iterable is None:
            return iter([])
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass


library.engine.tqdm = DummyTqdm

# Suppress transformers logging
logging.set_verbosity_error()


# =============================================================================
# Main Orchestration
# =============================================================================
def main():
    # 1. Setup
    seed_everything(Config.seed)
    device = get_device()
    os.makedirs(Config.working_dir, exist_ok=True)

    # 2. Load Data & Generate Structural Features
    print("Loading data and generating features...")
    df_train = pd.read_csv(Config.train_path)
    df_val_holdout = pd.read_csv(Config.val_path)
    df_test = pd.read_csv(Config.test_path)

    # Generate SVD features (Fit on Train, Transform Val/Test)
    feature_gen = StructuralFeatureGenerator()
    train_feats, val_feats, test_feats = feature_gen.generate_features(
        load_cached_data=True
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # =========================================================================
    # Stage 1: Teacher Training
    # =========================================================================
    print("Starting Stage 1: Teacher Training...")

    # We will store teacher predictions on test set here
    teacher_test_preds = np.zeros(len(df_test))

    # Stratified K-Fold on Training Data
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    # Store teacher models to clear memory later if needed, but we need them for inference immediately
    # Actually, we can predict on test set fold-by-fold to save memory

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_train, df_train["Insult"])
    ):
        print(f"  Teacher Fold {fold + 1}/{Config.n_folds}")

        # Split Data
        train_sub = df_train.iloc[train_idx].reset_index(drop=True)
        val_sub = df_train.iloc[val_idx].reset_index(drop=True)

        train_sub_feats = train_feats[train_idx]
        val_sub_feats = train_feats[val_idx]

        # Datasets
        train_ds = InsultDataset(
            train_sub["Comment"].values,
            train_sub_feats,
            tokenizer,
            labels=train_sub["Insult"].values,
        )
        val_ds = InsultDataset(
            val_sub["Comment"].values,
            val_sub_feats,
            tokenizer,
            labels=val_sub["Insult"].values,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Model
        model = HybridDebertaModel(pretrained=True).to(device)

        # Optimizer
        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in model.backbone.named_parameters()
                    if not any(nd in n for nd in ["bias", "LayerNorm.weight"])
                ],
                "lr": Config.lr_backbone,
                "weight_decay": Config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.backbone.named_parameters()
                    if any(nd in n for nd in ["bias", "LayerNorm.weight"])
                ],
                "lr": Config.lr_backbone,
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if "backbone" not in n
                ],
                "lr": Config.lr_head,
                "weight_decay": Config.weight_decay,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters)

        # Scheduler
        num_train_steps = int(len(train_sub) / Config.batch_size * Config.epochs)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
            num_training_steps=num_train_steps,
        )

        # Loss
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop
        best_auc = 0
        best_model_state = None

        for epoch in range(Config.epochs):
            train_loss = train_fn(
                train_loader,
                model,
                criterion,
                optimizer,
                epoch,
                scheduler,
                device,
                Config,
            )
            val_loss, val_auc = valid_fn(val_loader, model, criterion, device)

            if val_auc > best_auc:
                best_auc = val_auc
                best_model_state = model.state_dict()

        # Load best model for inference
        model.load_state_dict(best_model_state)

        # Predict on Test Set (Accumulate)
        test_ds = InsultDataset(
            df_test["Comment"].values, test_feats, tokenizer, labels=None
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.batch_size * 2,
            shuffle=False,
            num_workers=Config.num_workers,
        )

        fold_preds = inference_fn(test_loader, model, device)
        teacher_test_preds += fold_preds / Config.n_folds

        # Cleanup
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # =========================================================================
    # Stage 2 & 3: Student Training (Self-Distillation)
    # =========================================================================
    print("Starting Stage 3: Student Training (Distillation)...")

    # Prepare Soft Labels for Test Data
    soft_labels = teacher_test_preds  # These are probabilities [0, 1]

    # We will store Student models for final inference
    student_models = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_train, df_train["Insult"])
    ):
        print(f"  Student Fold {fold + 1}/{Config.n_folds}")

        # 1. Prepare Augmented Training Data (Train Fold + Soft Test)
        train_sub = df_train.iloc[train_idx].reset_index(drop=True)
        train_sub_feats = train_feats[train_idx]
        train_sub_labels = train_sub["Insult"].values.astype(np.float32)

        # Combine with Test Data
        aug_texts = np.concatenate(
            [train_sub["Comment"].values, df_test["Comment"].values]
        )
        aug_feats = np.concatenate([train_sub_feats, test_feats])
        aug_labels = np.concatenate([train_sub_labels, soft_labels])

        # Validation Data (Same as Teacher Fold Val)
        val_sub = df_train.iloc[val_idx].reset_index(drop=True)
        val_sub_feats = train_feats[val_idx]
        val_sub_labels = val_sub["Insult"].values

        # Datasets
        train_ds = InsultDataset(aug_texts, aug_feats, tokenizer, labels=aug_labels)
        val_ds = InsultDataset(
            val_sub["Comment"].values, val_sub_feats, tokenizer, labels=val_sub_labels
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        # Model
        model = HybridDebertaModel(pretrained=True).to(device)

        # Optimizer
        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in model.backbone.named_parameters()
                    if not any(nd in n for nd in ["bias", "LayerNorm.weight"])
                ],
                "lr": Config.lr_backbone,
                "weight_decay": Config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.backbone.named_parameters()
                    if any(nd in n for nd in ["bias", "LayerNorm.weight"])
                ],
                "lr": Config.lr_backbone,
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if "backbone" not in n
                ],
                "lr": Config.lr_head,
                "weight_decay": Config.weight_decay,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters)

        # Scheduler
        num_train_steps = int(len(aug_texts) / Config.batch_size * Config.epochs)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
            num_training_steps=num_train_steps,
        )

        # Loss (BCEWithLogitsLoss handles soft targets perfectly)
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop
        best_auc = 0
        best_model_state = None

        for epoch in range(Config.epochs):
            train_loss = train_fn(
                train_loader,
                model,
                criterion,
                optimizer,
                epoch,
                scheduler,
                device,
                Config,
            )
            val_loss, val_auc = valid_fn(val_loader, model, criterion, device)

            if val_auc > best_auc:
                best_auc = val_auc
                best_model_state = model.state_dict()

        # Save best model to disk to save RAM, reload later
        save_path = os.path.join(Config.working_dir, f"student_fold_{fold}.bin")
        torch.save(best_model_state, save_path)

        # Cleanup
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # =========================================================================
    # Final Validation on Hold-out Set
    # =========================================================================
    print("Performing Final Validation on Hold-out Set...")

    val_holdout_ds = InsultDataset(
        df_val_holdout["Comment"].values, val_feats, tokenizer, labels=None
    )
    val_holdout_loader = DataLoader(
        val_holdout_ds,
        batch_size=Config.batch_size * 2,
        shuffle=False,
        num_workers=Config.num_workers,
    )

    final_val_preds = np.zeros(len(df_val_holdout))

    for fold in range(Config.n_folds):
        model = HybridDebertaModel(pretrained=False)  # Config loaded internally
        model.to(device)
        model.load_state_dict(
            torch.load(os.path.join(Config.working_dir, f"student_fold_{fold}.bin"))
        )

        fold_preds = inference_fn(val_holdout_loader, model, device)
        final_val_preds += fold_preds / Config.n_folds

        del model
        torch.cuda.empty_cache()

    final_auc = roc_auc_score(df_val_holdout["Insult"].values, final_val_preds)
    print(f"Final Validation Metric: {final_auc}")

    # =========================================================================
    # Failure Analysis
    # =========================================================================
    print("Performing Failure Analysis...")
    residuals = np.abs(df_val_holdout["Insult"].values - final_val_preds)

    # Metadata features
    df_val_holdout["char_len"] = df_val_holdout["Comment"].fillna("").apply(len)
    df_val_holdout["word_count"] = (
        df_val_holdout["Comment"].fillna("").apply(lambda x: len(str(x).split()))
    )

    # SVD Norm
    svd_norms = np.linalg.norm(val_feats, axis=1)

    corr_len = np.corrcoef(residuals, df_val_holdout["char_len"])[0, 1]
    corr_word = np.corrcoef(residuals, df_val_holdout["word_count"])[0, 1]
    corr_svd = np.corrcoef(residuals, svd_norms)[0, 1]

    print(f"Correlation (Error vs Char Length): {corr_len}")
    print(f"Correlation (Error vs Word Count): {corr_word}")
    print(f"Correlation (Error vs Structural Feature Norm): {corr_svd}")

    # =========================================================================
    # Submission
    # =========================================================================
    threshold = 0.9586453201970443
    if final_auc > threshold:
        print("Metric check passed. Generating submission...")

        test_ds = InsultDataset(
            df_test["Comment"].values, test_feats, tokenizer, labels=None
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.batch_size * 2,
            shuffle=False,
            num_workers=Config.num_workers,
        )

        final_test_preds = np.zeros(len(df_test))

        for fold in range(Config.n_folds):
            model = HybridDebertaModel(pretrained=False)
            model.to(device)
            model.load_state_dict(
                torch.load(os.path.join(Config.working_dir, f"student_fold_{fold}.bin"))
            )

            fold_preds = inference_fn(test_loader, model, device)
            final_test_preds += fold_preds / Config.n_folds

            del model
            torch.cuda.empty_cache()

        submission_df = pd.DataFrame(
            {
                "id": df_test.index,  # Assuming index matches, sample sub has no ID col usually just rows
                "Insult": final_test_preds,
            }
        )

        # Match sample submission format
        # Sample submission has 3 columns usually in these old datasets, but prompt says "Your predictions should be a number in the range [0,1]"
        # and "See 'sample_submissions_null.csv' for the correct format."
        # The sample provided in description has: | Insult | Date | Comment |
        # Usually we just replace the Insult column.

        sample_sub = pd.read_csv("./input/sample_submission_null.csv")
        sample_sub["Insult"] = final_test_preds

        sample_sub.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")
    else:
        print(
            f"Metric {final_auc} did not beat threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
