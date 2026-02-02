import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModel, get_linear_schedule_with_warmup

from library.config import Config
from library.utils import set_seed, calculate_log_loss


class TransformerModel(nn.Module):
    """
    A PyTorch module wrapping a Hugging Face Transformer backbone for sequence classification.
    """

    def __init__(self, model_name=Config.MODEL_NAME, num_classes=Config.NUM_CLASSES):
        super(TransformerModel, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.config = self.backbone.config

        # Dropout for regularization
        self.drop = nn.Dropout(p=0.1)

        # Classification head
        self.fc = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # Get backbone outputs
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the last hidden state
        last_hidden_state = outputs.last_hidden_state

        # Use the [CLS] token embedding (first token) for classification
        cls_embedding = last_hidden_state[:, 0, :]

        # Apply dropout and classification layer
        x = self.drop(cls_embedding)
        logits = self.fc(x)

        return logits


def predict_transformer(model, data_loader, device=Config.DEVICE):
    """
    Generates probability predictions for a given DataLoader using the trained model.

    Args:
        model (nn.Module): The trained Transformer model.
        data_loader (DataLoader): DataLoader containing the data to predict.
        device (torch.device): Device to run inference on.

    Returns:
        np.ndarray: Predicted probabilities of shape (n_samples, n_classes).
    """
    model.eval()
    model.to(device)

    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


def train_transformer(
    train_loader,
    val_loader,
    epochs=Config.EPOCHS,
    lr=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    warmup_ratio=Config.WARMUP_RATIO,
    patience=Config.EARLY_STOPPING_PATIENCE,
    save_path=Config.MODEL_SAVE_PATH,
    device=Config.DEVICE,
):
    """
    Trains the Transformer model with Early Stopping and saves the best version.

    Args:
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        epochs (int): Maximum number of epochs.
        lr (float): Learning rate.
        weight_decay (float): Weight decay for optimizer.
        warmup_ratio (float): Ratio of total steps for warmup.
        patience (int): Early stopping patience.
        save_path (str): Path to save the best model.
        device (torch.device): Compute device.

    Returns:
        model (nn.Module): The trained model (loaded with best weights).
    """
    set_seed(Config.SEED)

    # Initialize model
    print(f"Initializing Transformer model: {Config.MODEL_NAME}")
    model = TransformerModel(
        model_name=Config.MODEL_NAME, num_classes=Config.NUM_CLASSES
    )
    model.to(device)

    # Optimizer
    # We group parameters to apply weight decay only to non-bias/LayerNorm weights
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    # Scheduler
    total_steps = len(train_loader) * epochs
    num_warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    # Tracking variables
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss_accum = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            optimizer.step()
            scheduler.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss_accum = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                val_loss_accum += loss.item()

                probs = torch.softmax(logits, dim=1)
                val_preds.append(probs.cpu().numpy())
                val_targets.append(labels.cpu().numpy())

        avg_val_loss = val_loss_accum / len(val_loader)

        # Calculate Log Loss metric on the whole validation set
        val_preds_concat = np.concatenate(val_preds, axis=0)
        val_targets_concat = np.concatenate(val_targets, axis=0)
        val_log_loss = calculate_log_loss(val_targets_concat, val_preds_concat)

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss} | Val Loss (CE): {avg_val_loss} | Val Log Loss: {val_log_loss}"
        )

        # --- Early Stopping & Saving ---
        if val_log_loss < best_val_loss:
            best_val_loss = val_log_loss
            patience_counter = 0

            # Save best model
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            # print(f"New best model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    # Load best model before returning
    if os.path.exists(save_path):
        print(f"Loading best model from {save_path} with Val Log Loss: {best_val_loss}")
        model.load_state_dict(torch.load(save_path, map_location=device))
    else:
        print("Warning: No model file saved. Returning current model.")

    return model
