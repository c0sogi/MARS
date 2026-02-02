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


class CrossEncoderHybridModel(nn.Module):
    """
    Cross-Encoder Transformer with Hybrid Head.
    Processes [CLS] Prompt [SEP] Resp A [SEP] Resp B [SEP] in a single pass.
    Cite solution_lesson_node_00006: Cross-Encoders allow deep interaction between prompt and responses.
    """

    def __init__(
        self, model_name=Config.MODEL_NAME, num_labels=Config.NUM_LABELS, num_scalars=3
    ):
        super(CrossEncoderHybridModel, self).__init__()

        # 1. Backbone
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        hidden_size = self.config.hidden_size

        # 2. Hybrid Head (Cite solution_lesson_node_00004)
        # We take the [CLS] embedding and concatenate scalars
        combined_dim = hidden_size + num_scalars

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),  # Increased dropout for regularization
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids, attention_mask, scalars):
        """
        Args:
            input_ids, attention_mask: Concatenated sequence inputs
            scalars: [batch_size, num_scalars] Explicit features
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Take [CLS] token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Hybrid Fusion
        combined = torch.cat([cls_embedding, scalars], dim=1)

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
            probs = torch.softmax(logits.float(), dim=1)
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

            probs = torch.softmax(logits.float(), dim=1)
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

    submission.to_csv(Config.SUBMISSION_PATH, index=False, float_format="%.8f")
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
