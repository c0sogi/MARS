import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import StratifiedGroupKFold

from library.config import Config
from library.dataset import PhraseDataset, load_and_preprocess_data
from library.utils import seed_everything, compute_pearson

# =============================================================================
# Model Definition
# =============================================================================


class PhraseSimilarityModel(nn.Module):
    def __init__(self, model_name=Config.model_name, pretrained=True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            model_name, num_labels=Config.num_classes
        )

        # Configure dropout
        self.config.classifier_dropout = Config.dropout
        self.config.attention_probs_dropout_prob = 0.0
        self.config.hidden_dropout_prob = 0.0

        if pretrained:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, config=self.config
            )
        else:
            self.model = AutoModelForSequenceClassification.from_config(self.config)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        # Forward pass through the HF model
        # Returns SequenceClassifierOutput (loss, logits, hidden_states, etc.)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        return output


# =============================================================================
# Training & Evaluation Helpers
# =============================================================================


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

        loss = outputs.loss
        logits = outputs.logits.squeeze(-1)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * input_ids.size(0)

        all_preds.extend(logits.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_pearson = compute_pearson(all_preds, all_labels)

    return epoch_loss, epoch_pearson


def validate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )

            loss = outputs.loss
            logits = outputs.logits.squeeze(-1)

            running_loss += loss.item() * input_ids.size(0)
            all_preds.extend(logits.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_pearson = compute_pearson(all_preds, all_labels)

    return epoch_loss, epoch_pearson


def inference(model, dataloader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=None,
            )

            logits = outputs.logits.squeeze(-1)
            all_preds.extend(logits.detach().cpu().numpy())

    return np.array(all_preds)


# =============================================================================
# Main Pipeline
# =============================================================================


def run_kfold_training():
    seed_everything(Config.seed)
    device = Config.device

    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading data...")
    df_train_part = load_and_preprocess_data("train")
    df_val_part = load_and_preprocess_data("val")
    df_test = load_and_preprocess_data("test")

    # Combine for K-Fold
    df_full = pd.concat([df_train_part, df_val_part]).reset_index(drop=True)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Prepare Test Dataset & Loader
    test_dataset = PhraseDataset(df_test, tokenizer, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory,
    )

    # 2. K-Fold Setup
    skf = StratifiedGroupKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    # Stratification targets and groups
    y_strata = df_full["score"].astype(str)
    groups = df_full["anchor"]

    fold_preds = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_full, y_strata, groups=groups)
    ):
        print(f"\n{'='*20} Fold {fold+1}/{Config.n_folds} {'='*20}")

        df_train_fold = df_full.iloc[train_idx].reset_index(drop=True)
        df_val_fold = df_full.iloc[val_idx].reset_index(drop=True)

        # Datasets
        train_dataset = PhraseDataset(df_train_fold, tokenizer, mode="train")
        val_dataset = PhraseDataset(df_val_fold, tokenizer, mode="val")

        # Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.train_batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=Config.pin_memory,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=Config.pin_memory,
        )

        # Model
        model = PhraseSimilarityModel(pretrained=True)
        model.to(device)

        # Optimizer & Scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay,
        )

        num_training_steps = len(train_loader) * Config.epochs
        num_warmup_steps = int(num_training_steps * Config.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Training Loop with Early Stopping
        best_pearson = -1.0
        patience_counter = 0
        best_model_path = os.path.join(
            Config.model_output_dir, f"model_fold_{fold}.bin"
        )

        for epoch in range(Config.epochs):
            train_loss, train_pearson = train_one_epoch(
                model, train_loader, optimizer, scheduler, device, epoch
            )
            val_loss, val_pearson = validate(model, val_loader, device)

            print(
                f"Epoch {epoch+1}/{Config.epochs} | "
                f"Train Loss: {train_loss:.4f} Pearson: {train_pearson:.4f} | "
                f"Val Loss: {val_loss:.4f} Pearson: {val_pearson:.10f}"
            )

            if val_pearson > best_pearson:
                best_pearson = val_pearson
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> New Best Model Saved (Pearson: {best_pearson:.10f})")
            else:
                patience_counter += 1
                print(f"  -> Patience: {patience_counter}/{Config.patience}")

            if patience_counter >= Config.patience:
                print("Early stopping triggered.")
                break

        # Load Best Model for Inference
        print(f"Loading best model for Fold {fold+1} inference...")
        model.load_state_dict(torch.load(best_model_path))
        model.to(device)

        # Predict on Test
        preds = inference(model, test_loader, device)
        fold_preds.append(preds)

        # Clean up to save memory
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # 3. Ensemble Predictions
    print("\nEnsembling predictions...")
    avg_preds = np.mean(fold_preds, axis=0)

    # 4. Save Submission
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    submission_df = pd.DataFrame({"id": df_test["id"], "score": avg_preds})

    # Clip scores to valid range [0, 1]
    submission_df["score"] = submission_df["score"].clip(0.0, 1.0)

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    # Run the pipeline
    run_kfold_training()
