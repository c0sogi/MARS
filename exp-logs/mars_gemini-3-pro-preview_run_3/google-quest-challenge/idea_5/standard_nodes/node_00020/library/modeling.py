import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.linear_model import RidgeCV
import joblib

from library.config import GlobalConfig, ModelConfig, MPNET_CONFIG, DEBERTA_CONFIG
from library.dataset import get_dataloader
from library.utils import seed_everything, compute_spearman_metric

# --------------------------------------------------------------------------
# Model Definition
# --------------------------------------------------------------------------


class SegmentAwareCrossEncoder(nn.Module):
    """
    A Cross-Encoder that computes segment-aware embeddings for Question and Answer.

    Architecture:
    1. Transformer Backbone (e.g., MPNet, RoBERTa)
    2. Segment-Aware Pooling:
       - h_cls: [CLS] token embedding
       - h_q: Mean of Question tokens (segment_mask == 1)
       - h_a: Mean of Answer tokens (segment_mask == 2)
       - h_diff: h_q - h_a
    3. Feature Vector: Concat([h_cls, h_q, h_a, h_diff])
    4. Linear Head: For fine-tuning targets.
    """

    def __init__(self, model_name, num_labels=30, dropout_prob=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.config = self.backbone.config
        self.hidden_size = self.config.hidden_size

        # Feature dimension: 4 * hidden_size (CLS, Q, A, Diff)
        self.feature_dim = 4 * self.hidden_size

        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.feature_dim, num_labels)

        # Initialize weights of the head
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, segment_mask):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            segment_mask: (batch, seq_len) - 0:Pad/Special, 1:Question, 2:Answer

        Returns:
            logits: (batch, num_labels)
            features: (batch, feature_dim)
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # 1. CLS Embedding
        h_cls = last_hidden_state[:, 0, :]

        # 2. Segment Masking
        # Expand masks to (batch, seq_len, 1) for broadcasting
        q_mask = (segment_mask == 1).unsqueeze(-1).float()
        a_mask = (segment_mask == 2).unsqueeze(-1).float()

        # 3. Mean Pooling for Q and A
        # Sum embeddings
        sum_q = torch.sum(last_hidden_state * q_mask, dim=1)
        sum_a = torch.sum(last_hidden_state * a_mask, dim=1)

        # Count tokens (clamp to avoid div by zero)
        cnt_q = torch.clamp(q_mask.sum(dim=1), min=1e-9)
        cnt_a = torch.clamp(a_mask.sum(dim=1), min=1e-9)

        h_q = sum_q / cnt_q
        h_a = sum_a / cnt_a

        # 4. Difference Feature
        h_diff = h_q - h_a

        # 5. Concatenate
        features = torch.cat([h_cls, h_q, h_a, h_diff], dim=1)

        # 6. Classification Head
        pooled_output = self.dropout(features)
        logits = self.classifier(pooled_output)

        return logits, features


# --------------------------------------------------------------------------
# Training & Processing Functions
# --------------------------------------------------------------------------


def train_backbone(model_config, tokenizer):
    """
    Fine-tunes the backbone model using the training set.
    """
    seed_everything(GlobalConfig.SEED)
    device = torch.device(GlobalConfig.DEVICE)

    print(f"\n[Stream: {model_config.name_tag}] Starting Backbone Fine-tuning...")

    # 1. Prepare Data
    train_loader = get_dataloader(
        GlobalConfig.TRAIN_METADATA_PATH,
        tokenizer,
        batch_size=model_config.train_batch_size,
        max_length=model_config.max_length,
        shuffle=True,
    )
    val_loader = get_dataloader(
        GlobalConfig.VAL_METADATA_PATH,
        tokenizer,
        batch_size=model_config.valid_batch_size,
        max_length=model_config.max_length,
        shuffle=False,
    )

    # 2. Initialize Model
    model = SegmentAwareCrossEncoder(model_config.model_name)
    model.to(device)

    # 3. Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=model_config.learning_rate,
        weight_decay=model_config.weight_decay,
    )
    total_steps = len(train_loader) * model_config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_val_loss = float("inf")
    patience = 2
    patience_counter = 0

    for epoch in range(model_config.epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            segment_mask = batch["segment_mask"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask, segment_mask)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                segment_mask = batch["segment_mask"].to(device)
                targets = batch["targets"].to(device)

                logits, _ = model(input_ids, attention_mask, segment_mask)
                loss = criterion(logits, targets)
                val_loss += loss.item()

                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_targets.append(targets.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        val_score = compute_spearman_metric(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{model_config.epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"Val Spearman: {val_score:.6f}"
        )

        # Early Stopping & Saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_config.model_save_path)
            print(f"  -> Model Saved to {model_config.model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("  -> Early stopping triggered.")
                break

    # Cleanup
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()


def extract_features(
    model_config, tokenizer, data_path, output_path, load_cached_data=True
):
    """
    Extracts features using the fine-tuned backbone. Caches results to disk.
    """
    if load_cached_data and os.path.exists(output_path):
        print(f"Loading cached features from {output_path}")
        return np.load(output_path)

    print(f"Extracting features for {os.path.basename(data_path)}...")

    device = torch.device(GlobalConfig.DEVICE)

    # Load Model
    model = SegmentAwareCrossEncoder(model_config.model_name)
    if os.path.exists(model_config.model_save_path):
        model.load_state_dict(
            torch.load(model_config.model_save_path, map_location=device)
        )
    else:
        print("Warning: Fine-tuned weights not found. Using pre-trained weights.")

    model.to(device)
    model.eval()

    # DataLoader
    # Determine if it's test set by filename convention or path
    is_test = "test" in os.path.basename(data_path)
    dataloader = get_dataloader(
        data_path,
        tokenizer,
        batch_size=model_config.valid_batch_size,
        max_length=model_config.max_length,
        shuffle=False,
        is_test=is_test,
    )

    features_list = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            segment_mask = batch["segment_mask"].to(device)

            _, features = model(input_ids, attention_mask, segment_mask)
            features_list.append(features.cpu().numpy().astype(np.float32))

    all_features = np.vstack(features_list)

    # Save
    np.save(output_path, all_features)
    print(f"Features saved to {output_path} (Shape: {all_features.shape})")

    del model, dataloader
    torch.cuda.empty_cache()
    gc.collect()

    return all_features


def train_ridge_head(model_config, train_features, train_targets):
    """
    Trains a Ridge Regression head on the extracted features.
    """
    print(f"\n[Stream: {model_config.name_tag}] Training Ridge Head...")

    # RidgeCV handles multi-output regression natively
    # Alphas to search
    alphas = [0.1, 1.0, 10.0, 100.0]

    ridge = RidgeCV(alphas=alphas, scoring=None)  # Default scoring is R^2
    ridge.fit(train_features, train_targets)

    print(f"  -> Best Alpha: {ridge.alpha_}")

    # Save model
    joblib.dump(ridge, model_config.ridge_path)
    print(f"  -> Ridge Model saved to {model_config.ridge_path}")
    return ridge


def predict_stream(model_config, test_features):
    """
    Predicts using the Ridge Head.
    """
    if not os.path.exists(model_config.ridge_path):
        raise FileNotFoundError(f"Ridge model not found at {model_config.ridge_path}")

    ridge = joblib.load(model_config.ridge_path)
    preds = ridge.predict(test_features)

    # Ridge can output values outside [0,1], clip them
    preds = np.clip(preds, 0.0, 1.0)
    return preds


def run_pipeline():
    """
    Orchestrates the full pipeline:
    1. Fine-tune MPNet & RoBERTa
    2. Extract Features
    3. Train Ridge Heads
    4. Predict & Fuse
    """
    seed_everything(GlobalConfig.SEED)

    # Load Targets
    train_df = pd.read_csv(GlobalConfig.TRAIN_METADATA_PATH)
    train_targets = train_df[GlobalConfig.TARGET_COLS].values.astype(np.float32)

    # ----------------------------------------------------------------------
    # Stream 1: MPNet
    # ----------------------------------------------------------------------
    tokenizer_mpnet = AutoTokenizer.from_pretrained(MPNET_CONFIG.model_name)

    # 1. Fine-tune
    if not os.path.exists(MPNET_CONFIG.model_save_path):
        train_backbone(MPNET_CONFIG, tokenizer_mpnet)

    # 2. Extract Features
    mpnet_train_feats = extract_features(
        MPNET_CONFIG,
        tokenizer_mpnet,
        GlobalConfig.TRAIN_METADATA_PATH,
        MPNET_CONFIG.train_features_path,
    )
    mpnet_test_feats = extract_features(
        MPNET_CONFIG,
        tokenizer_mpnet,
        GlobalConfig.TEST_METADATA_PATH,
        MPNET_CONFIG.test_features_path,
    )

    # 3. Train Ridge
    train_ridge_head(MPNET_CONFIG, mpnet_train_feats, train_targets)

    # 4. Predict
    mpnet_preds = predict_stream(MPNET_CONFIG, mpnet_test_feats)

    # Clear memory
    del tokenizer_mpnet, mpnet_train_feats, mpnet_test_feats
    gc.collect()

    # ----------------------------------------------------------------------
    # Stream 2: DeBERTa
    # ----------------------------------------------------------------------
    tokenizer_deberta = AutoTokenizer.from_pretrained(DEBERTA_CONFIG.model_name)

    # 1. Fine-tune
    if not os.path.exists(DEBERTA_CONFIG.model_save_path):
        train_backbone(DEBERTA_CONFIG, tokenizer_deberta)

    # 2. Extract Features
    deberta_train_feats = extract_features(
        DEBERTA_CONFIG,
        tokenizer_deberta,
        GlobalConfig.TRAIN_METADATA_PATH,
        DEBERTA_CONFIG.train_features_path,
    )
    deberta_test_feats = extract_features(
        DEBERTA_CONFIG,
        tokenizer_deberta,
        GlobalConfig.TEST_METADATA_PATH,
        DEBERTA_CONFIG.test_features_path,
    )

    # 3. Train Ridge
    train_ridge_head(DEBERTA_CONFIG, deberta_train_feats, train_targets)

    # 4. Predict
    deberta_preds = predict_stream(DEBERTA_CONFIG, deberta_test_feats)

    # Clear memory
    del tokenizer_deberta, deberta_train_feats, deberta_test_feats
    gc.collect()

    # ----------------------------------------------------------------------
    # Late Fusion & Submission
    # ----------------------------------------------------------------------
    print("\nGenerating Final Submission (Late Fusion)...")

    # Average Probabilities
    final_preds = 0.5 * mpnet_preds + 0.5 * deberta_preds

    # Create Submission DataFrame
    test_df = pd.read_csv(GlobalConfig.TEST_METADATA_PATH)
    sub_df = pd.DataFrame(final_preds, columns=GlobalConfig.TARGET_COLS)
    sub_df.insert(0, "qa_id", test_df["qa_id"])

    # Save
    sub_df.to_csv(GlobalConfig.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {GlobalConfig.SUBMISSION_PATH}")
    print(sub_df.head())


if __name__ == "__main__":
    # This block is for testing purposes only, not part of the module definition
    pass
