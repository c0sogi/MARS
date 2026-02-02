import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoConfig, get_linear_schedule_with_warmup
from library.config import Config
from library.utils import compute_score
from library.data import get_dataloaders, get_test_dataloader

# ==========================================
# Modeling Components
# ==========================================


class AttentionPooling(nn.Module):
    """
    Learned weighted averaging of token embeddings.
    """

    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

    def forward(self, last_hidden_state, attention_mask):
        """
        Args:
            last_hidden_state: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] (1 for valid, 0 for pad)
        """
        # Calculate attention scores
        # w: [batch_size, seq_len, 1]
        w = self.attention(last_hidden_state)

        # Mask padding tokens
        # We use a large negative number to effectively zero out the softmax probability for padding
        mask_expanded = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
        w = w + (1.0 - mask_expanded) * -1e9

        # Compute weights
        scores = torch.softmax(w, dim=1)  # [batch, seq_len, 1]

        # Weighted sum
        # context_vector: [batch_size, hidden_size]
        context_vector = torch.sum(last_hidden_state * scores, dim=1)
        return context_vector


class SiameseHybridModel(nn.Module):
    """
    Siamese Contextual Transformer with Hybrid Head.
    Uses a shared encoder for two branches, attention pooling,
    interaction features, and explicit scalar features.
    """

    def __init__(
        self, model_name=Config.MODEL_NAME, num_labels=Config.NUM_LABELS, num_scalars=3
    ):
        super(SiameseHybridModel, self).__init__()

        # 1. Backbone
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        # Enable gradient checkpointing for memory efficiency if needed
        # self.backbone.gradient_checkpointing_enable()

        hidden_size = self.config.hidden_size

        # 2. Pooling
        self.pooling = AttentionPooling(hidden_size)

        # 3. Hybrid Head
        # Interaction features: u, v, |u-v|, u*v -> 4 vectors
        interaction_dim = 4 * hidden_size

        # Total input dimension for MLP = Interaction vectors + Scalar features
        combined_dim = interaction_dim + num_scalars

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(
        self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, scalars
    ):
        """
        Args:
            input_ids_a, attention_mask_a: Inputs for Branch A (Prompt + Resp A)
            input_ids_b, attention_mask_b: Inputs for Branch B (Prompt + Resp B)
            scalars: [batch_size, num_scalars] Explicit features
        """
        # Branch A Forward
        out_a = self.backbone(input_ids=input_ids_a, attention_mask=attention_mask_a)
        u = self.pooling(out_a.last_hidden_state, attention_mask_a)

        # Branch B Forward
        out_b = self.backbone(input_ids=input_ids_b, attention_mask=attention_mask_b)
        v = self.pooling(out_b.last_hidden_state, attention_mask_b)

        # Interaction Features
        diff = torch.abs(u - v)
        prod = u * v

        # Concatenate: [u, v, |u-v|, u*v]
        interaction = torch.cat([u, v, diff, prod], dim=1)

        # Hybrid Fusion: [interaction, scalars]
        combined = torch.cat([interaction, scalars], dim=1)

        # Classification
        logits = self.classifier(combined)

        return logits


# ==========================================
# Training & Evaluation Logic
# ==========================================


def train_one_epoch(model, dataloader, optimizer, scheduler, device, criterion, scaler):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        # Move inputs to device
        input_ids_a = batch["input_ids_a"].to(device)
        attention_mask_a = batch["attention_mask_a"].to(device)
        input_ids_b = batch["input_ids_b"].to(device)
        attention_mask_b = batch["attention_mask_b"].to(device)
        scalars = batch["scalars"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Mixed Precision Forward
        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            logits = model(
                input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, scalars
            )
            loss = criterion(logits, labels)

        # Backward
        scaler.scale(loss).backward()

        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIPPING)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            scalars = batch["scalars"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = model(
                    input_ids_a,
                    attention_mask_a,
                    input_ids_b,
                    attention_mask_b,
                    scalars,
                )
                loss = criterion(logits, labels)

            total_loss += loss.item()

            # Apply softmax for predictions
            probs = torch.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Compute metrics
    metrics = compute_score(all_labels, all_preds)
    metrics["log_loss"] = avg_loss  # Use the calculated average loss directly

    return metrics


def train_model(tokenizer):
    """
    Main training loop with Early Stopping.
    """
    # Setup
    device = Config.DEVICE
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader = get_dataloaders(tokenizer)

    # Model
    model = SiameseHybridModel()
    model.to(device)

    # Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_training_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    # Tracking
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, criterion, scaler
        )
        val_metrics = validate(model, val_loader, device, criterion)

        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_metrics['log_loss']}")
        print(f"Val Accuracy: {val_metrics['accuracy']}")

        # Early Stopping & Checkpointing
        if val_metrics["log_loss"] < best_val_loss:
            best_val_loss = val_metrics["log_loss"]
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved to {Config.MODEL_PATH}")
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    return model


def predict(tokenizer):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    device = Config.DEVICE

    # Load Data
    test_loader = get_test_dataloader(tokenizer)

    # Load Model
    model = SiameseHybridModel()
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print(f"Loaded model from {Config.MODEL_PATH}")
    else:
        print(
            "Warning: Model checkpoint not found. Using initialized weights (random)."
        )

    model.to(device)
    model.eval()

    all_preds = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            scalars = batch["scalars"].to(device)

            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = model(
                    input_ids_a,
                    attention_mask_a,
                    input_ids_b,
                    attention_mask_b,
                    scalars,
                )

            probs = torch.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)

    # Create Submission DataFrame
    # We need the IDs from the test file to match the rows
    test_df = pd.read_csv(Config.TEST_DATA_PATH)
    submission = pd.DataFrame(
        {
            "id": test_df["id"],
            "winner_model_a": all_preds[:, 0],
            "winner_model_b": all_preds[:, 1],
            "winner_tie": all_preds[:, 2],
        }
    )

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
