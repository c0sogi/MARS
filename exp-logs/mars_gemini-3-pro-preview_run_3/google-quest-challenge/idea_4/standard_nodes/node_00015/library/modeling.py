import os
import numpy as np
import torch
import torch.nn as nn
import joblib
import pandas as pd
from torch.optim import AdamW
from transformers import AutoModel, AutoConfig, get_linear_schedule_with_warmup
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import MinMaxScaler

from library.config import Config
from library.utils import compute_metric, seed_everything, TrainingLogger


class SegmentAwareCrossEncoder(nn.Module):
    """
    A Cross-Encoder that wraps a Transformer backbone and extracts
    segment-specific features (CLS, Question, Answer, Difference).
    """

    def __init__(self, model_name, num_labels=30):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        self.hidden_size = self.config.hidden_size

        # Temporary linear head for fine-tuning
        self.classifier = nn.Linear(self.hidden_size, num_labels)

        # Determine separator token ID for segmentation
        # RoBERTa uses 2, BERT/MPNet use 102
        self.sep_token_id = (
            self.config.sep_token_id if self.config.sep_token_id is not None else 102
        )
        self.model_type = self.config.model_type

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass for Fine-Tuning. Returns logits.
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Use CLS token (or pooler output) for stability during fine-tuning
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits

    def get_segment_features(self, input_ids, attention_mask, token_type_ids=None):
        """
        Extracts concatenated features: [CLS, Q_mean, A_mean, Q_mean - A_mean].
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last_hidden_state = outputs.last_hidden_state  # (B, Seq, H)

        # 1. CLS Embedding
        h_cls = last_hidden_state[:, 0, :]

        # 2. Identify Segments
        # Find the first separator token index for each sequence in the batch
        # input_ids: (B, Seq)
        is_sep = (input_ids == self.sep_token_id).long()
        # argmax gives the index of the first '1' (first SEP)
        sep_indices = torch.argmax(is_sep, dim=1)

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Create range matrix [0, 1, ..., seq_len-1] repeated for batch
        seq_range = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )

        # Question Mask: positions > 0 and < first_sep
        # We start at 1 to skip CLS (or <s>)
        q_mask = (seq_range > 0) & (seq_range < sep_indices.unsqueeze(1))

        # Answer Mask: positions > first_sep and valid (attention_mask=1)
        # For RoBERTa, the format is <s> Q </s> </s> A </s>.
        # The first SEP is at end of Q. The start of A is sep_indices + 1 (or +2).
        # We handle this by checking if the token after first SEP is also a SEP.

        # Default start of A
        a_start = sep_indices.unsqueeze(1) + 1

        # Refine for RoBERTa double-sep case if needed, but simple masking usually works.
        # We exclude the final SEP by ensuring we are within the valid length.
        # Valid length can be approximated by attention_mask sum.
        valid_lens = attention_mask.sum(dim=1).unsqueeze(1)
        # Exclude last token (usually SEP)
        a_mask = (seq_range >= a_start) & (seq_range < (valid_lens - 1))

        # Apply attention mask to be safe (ignore padding)
        q_mask = q_mask & (attention_mask.bool())
        a_mask = a_mask & (attention_mask.bool())

        # 3. Mean Pooling
        # Expand masks to (B, Seq, H)
        q_mask_expanded = q_mask.unsqueeze(-1).float()
        a_mask_expanded = a_mask.unsqueeze(-1).float()

        # Compute sums
        q_sum = (last_hidden_state * q_mask_expanded).sum(dim=1)
        a_sum = (last_hidden_state * a_mask_expanded).sum(dim=1)

        # Compute counts (clamp to avoid div by zero)
        q_counts = q_mask_expanded.sum(dim=1).clamp(min=1e-9)
        a_counts = a_mask_expanded.sum(dim=1).clamp(min=1e-9)

        h_q = q_sum / q_counts
        h_a = a_sum / a_counts

        # 4. Difference Feature
        h_diff = h_q - h_a

        # Concatenate: (B, 4*H)
        features = torch.cat([h_cls, h_q, h_a, h_diff], dim=1)
        return features


def train_backbone(model, train_loader, val_loader, epochs, lr, device, save_path):
    """
    Fine-tunes the backbone model using a temporary linear head.
    Implements Early Stopping based on Validation Loss.
    """
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY)

    # Scheduler
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    criterion = nn.BCEWithLogitsLoss()
    logger = TrainingLogger()

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        train_loss_accum = 0.0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, targets)

            # Gradient Accumulation
            loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            if (batch_idx + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss_accum += loss.item() * Config.GRADIENT_ACCUMULATION_STEPS

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss_accum = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["targets"].to(device)
                token_type_ids = batch.get("token_type_ids", None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)

                outputs = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(outputs, targets)
                val_loss_accum += loss.item()

                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(targets.cpu().numpy())

        avg_val_loss = val_loss_accum / len(val_loader)

        # Compute Metric
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_targets)
        val_score = compute_metric(y_true, y_pred)

        logger.log_epoch(epoch, avg_train_loss, avg_val_loss, val_score)

        # --- Early Stopping & Saving ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    # Load best model before returning
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


def extract_features(model, loader, device, cache_path, load_cached_data=True):
    """
    Extracts features using the trained backbone.
    Handles caching to disk (.npy).
    """
    # 1. Check Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        return np.load(cache_path)

    print(f"Extracting features (Cache miss: {cache_path})...")
    model.to(device)
    model.eval()

    all_features = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            # Use the feature extraction method
            feats = model.get_segment_features(
                input_ids, attention_mask, token_type_ids
            )
            all_features.append(feats.cpu().numpy())

    features_arr = np.vstack(all_features)

    # 2. Save Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, features_arr)
    print(f"Features saved to {cache_path}")

    return features_arr


def train_ridge_ensemble(X_train, y_train, X_val, y_val, save_path):
    """
    Trains a RidgeCV multi-output regressor on the concatenated features.
    """
    print("Training Ridge Regression Ensemble...")

    # RidgeCV automatically performs LOOCV to select alpha
    # Cite debug_lesson_2: Pass Mutable Lists, Not Tuples, to Scikit-Learn Array-Like Parameters
    ridge = RidgeCV(
        alphas=list(Config.RIDGE_ALPHAS), scoring=None
    )  # Default scoring is r2, which is fine for regression optimization

    # Fit
    ridge.fit(X_train, y_train)

    # Evaluate
    print("Evaluating Ridge Model on Validation Set...")
    val_preds = ridge.predict(X_val)

    # Clip predictions to [0, 1]
    val_preds = np.clip(val_preds, 0, 1)

    score = compute_metric(y_val, val_preds)
    print(f"Ridge Ensemble Validation Spearman Score: {score}")

    # Save model
    joblib.dump(ridge, save_path)
    print(f"Ridge model saved to {save_path}")

    return ridge


def predict_and_submit(ridge_model, X_test, test_df, submission_path):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating predictions for test set...")

    test_preds = ridge_model.predict(X_test)
    test_preds = np.clip(test_preds, 0, 1)

    # Create DataFrame
    sub_df = pd.DataFrame(test_preds, columns=Config.TARGET_COLS)
    sub_df.insert(0, Config.ID_COL, test_df[Config.ID_COL])

    # Save
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    sub_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
