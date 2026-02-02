import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import time
from transformers import AutoModel, AutoConfig, AutoTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from library.config import Config
from library.utils import seed_everything, get_score
from library.data import load_dataset, make_loader

# =============================================================================
# Pooling Layer
# =============================================================================


class LinearAttentionPooling(nn.Module):
    """
    Linear Attention Pooling:
    Computes a weighted average of hidden states using a learned attention vector.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention_vector = nn.Linear(hidden_size, 1)

    def forward(self, last_hidden_state, attention_mask):
        # last_hidden_state: (batch, seq_len, hidden_size)
        # attention_mask: (batch, seq_len)

        # Calculate raw attention scores: (batch, seq_len)
        attn_scores = self.attention_vector(last_hidden_state).squeeze(-1)

        # Mask padding tokens (set to -1e9)
        attn_scores.masked_fill_(attention_mask == 0, -1e9)

        # Softmax to get weights: (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(-1)

        # Weighted sum of hidden states: (batch, hidden_size)
        weighted_avg = torch.sum(last_hidden_state * attn_weights, dim=1)

        return weighted_avg


# =============================================================================
# Model Architecture
# =============================================================================


class ToxicityModel(nn.Module):
    """
    DeBERTa-v3 based model with Hybrid Pooling (Max + Linear Attention)
    and Multi-Sample Dropout.
    """

    def __init__(self):
        super().__init__()

        # Load Backbone
        self.config = AutoConfig.from_pretrained(Config.model_name)
        self.backbone = AutoModel.from_pretrained(Config.model_name, config=self.config)

        # Feature sizes
        self.hidden_size = Config.hidden_size

        # Pooling Layers
        self.attention_pooling = LinearAttentionPooling(self.hidden_size)

        # Multi-Sample Dropout
        # We create multiple dropout layers that will be applied in parallel
        self.dropouts = nn.ModuleList(
            [nn.Dropout(Config.dropout_rate) for _ in range(Config.dropout_samples)]
        )

        # Final Classification Layer
        # Input dim is hidden_size * 2 because we concatenate Max Pooling and Attention Pooling
        self.fc = nn.Linear(self.hidden_size * 2, Config.num_classes)

        # Initialize weights for the new head layers
        self._init_weights(self.attention_pooling)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def feature_extraction(self, input_ids, attention_mask):
        # Get backbone outputs
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # 1. Linear Attention Pooling
        attn_pool_out = self.attention_pooling(last_hidden_state, attention_mask)

        # 2. Global Max Pooling
        # Mask padding for max pooling to avoid selecting padding values
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        # Clone to avoid modifying original tensor in place
        last_hidden_state_masked = last_hidden_state.clone()
        # Set padding tokens to a large negative value
        last_hidden_state_masked[input_mask_expanded == 0] = -1e9
        max_pool_out = torch.max(last_hidden_state_masked, dim=1)[0]

        # Concatenate pooling results
        features = torch.cat([max_pool_out, attn_pool_out], dim=1)
        return features

    def forward(self, input_ids, attention_mask, labels=None):
        # Extract features
        features = self.feature_extraction(input_ids, attention_mask)

        # Multi-Sample Dropout
        logits_list = []
        for dropout in self.dropouts:
            # Apply dropout
            dropped_features = dropout(features)
            # Pass through the shared classification layer
            logits = self.fc(dropped_features)
            logits_list.append(logits)

        # Average the logits from all dropout samples
        # Stack: (num_samples, batch, num_classes) -> Mean dim 0 -> (batch, num_classes)
        final_logits = torch.mean(torch.stack(logits_list), dim=0)

        return final_logits


# =============================================================================
# Training & Execution Logic
# =============================================================================


def train_fn(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def valid_fn(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            # Apply sigmoid for predictions
            probs = torch.sigmoid(logits)
            preds_list.append(probs.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    final_preds = np.concatenate(preds_list)
    final_labels = np.concatenate(labels_list)

    avg_loss = total_loss / len(loader)
    score = get_score(final_labels, final_preds)

    return avg_loss, score


def inference_fn(model, loader, device):
    model.eval()
    preds_list = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            preds_list.append(probs.cpu().numpy())

    return np.concatenate(preds_list)


def run():
    """
    Main driver function to execute the training and inference pipeline.
    """
    seed_everything(Config.seed)

    print("Loading Data...")
    train_df = load_dataset("train")
    val_df = load_dataset("val")
    test_df = load_dataset("test")

    print(f"Train Size: {len(train_df)}")
    print(f"Val Size: {len(val_df)}")
    print(f"Test Size: {len(test_df)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # DataLoaders
    train_loader = make_loader(
        train_df, tokenizer, is_train=True, batch_size=Config.train_batch_size
    )
    val_loader = make_loader(
        val_df, tokenizer, is_train=False, batch_size=Config.valid_batch_size
    )
    test_loader = make_loader(
        test_df, tokenizer, is_train=False, batch_size=Config.valid_batch_size
    )

    # Model
    model = ToxicityModel()
    model.to(Config.device)

    # Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    # OneCycleLR
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=Config.epochs,
        pct_start=Config.pct_start,
        div_factor=Config.div_factor,
        final_div_factor=Config.final_div_factor,
    )

    # Loss
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    best_score = -np.inf

    print("Starting Training...")
    for epoch in range(Config.epochs):
        start_time = time.time()

        train_loss = train_fn(
            model, train_loader, optimizer, scheduler, criterion, Config.device
        )
        val_loss, val_score = valid_fn(model, val_loader, criterion, Config.device)

        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{Config.epochs} | Time: {elapsed:.0f}s")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Val AUC:    {val_score:.6f}")

        if val_score > best_score:
            best_score = val_score
            print(f"  New Best Score! Saving model to {Config.model_save_path}")
            torch.save(model.state_dict(), Config.model_save_path)

    # Inference
    print("Starting Inference on Test Set...")
    # Load best model
    model.load_state_dict(torch.load(Config.model_save_path))

    test_preds = inference_fn(model, test_loader, Config.device)

    # Create Submission
    submission = pd.DataFrame(test_preds, columns=Config.target_cols)
    submission["id"] = test_df["id"]

    # Reorder columns to match requirement: id, toxic, severe_toxic, ...
    cols = ["id"] + Config.target_cols
    submission = submission[cols]

    submission.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")
