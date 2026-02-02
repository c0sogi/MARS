import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoConfig, get_linear_schedule_with_warmup
from library.config import Config
from library.utils import get_logger, seed_everything

logger = get_logger("model")


class SiameseDualEncoder(nn.Module):
    """
    Siamese Dual-Encoder model architecture.

    This model uses a shared pre-trained Transformer backbone to encode
    (Prompt, Response A) and (Prompt, Response B) pairs independently.
    The resulting [CLS] embeddings are combined using concatenation and
    interaction terms before being passed to a classification head.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        dropout_prob=Config.DROPOUT,
    ):
        super(SiameseDualEncoder, self).__init__()

        # Load Transformer Backbone
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        # Calculate input dimension for the classifier
        # We concatenate: u, v, |u-v|, u*v
        # Each vector has size hidden_size
        combined_dim = self.config.hidden_size * 4

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(combined_dim, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(self.config.hidden_size, num_classes),
        )

    def mean_pooling(self, last_hidden_state, attention_mask):
        """
        Performs mean pooling on the token embeddings, ignoring padding.
        """
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b):
        """
        Forward pass for the Siamese Network.

        Args:
            input_ids_a, attention_mask_a: Inputs for (Prompt + Response A)
            input_ids_b, attention_mask_b: Inputs for (Prompt + Response B)

        Returns:
            logits: Unnormalized scores for [Winner A, Winner B, Tie]
        """
        # Pass A through backbone
        outputs_a = self.backbone(
            input_ids=input_ids_a, attention_mask=attention_mask_a
        )
        # Mean Pooling for A
        u = self.mean_pooling(outputs_a.last_hidden_state, attention_mask_a)

        # Pass B through backbone
        outputs_b = self.backbone(
            input_ids=input_ids_b, attention_mask=attention_mask_b
        )
        # Mean Pooling for B
        v = self.mean_pooling(outputs_b.last_hidden_state, attention_mask_b)

        # Feature Interaction
        diff = torch.abs(u - v)
        prod = u * v

        # Concatenate features
        features = torch.cat([u, v, diff, prod], dim=1)

        # Predict
        logits = self.classifier(features)
        return logits


def train_one_epoch(model, dataloader, optimizer, scheduler, device, criterion):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    correct_preds = 0
    total_samples = 0

    for batch in dataloader:
        # Move batch to device
        input_ids_a = batch["input_ids_a"].to(device)
        attention_mask_a = batch["attention_mask_a"].to(device)
        input_ids_b = batch["input_ids_b"].to(device)
        attention_mask_b = batch["attention_mask_b"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()
        if scheduler:
            scheduler.step()

        # Metrics
        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct_preds += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct_preds / total_samples
    return avg_loss, accuracy


def evaluate(model, dataloader, device, criterion):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_preds += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct_preds / total_samples
    return avg_loss, accuracy


def run_training(train_loader, val_loader):
    """
    Main training loop with Early Stopping and Model Checkpointing.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE
    logger.info(f"Initializing model on device: {device}")

    model = SiameseDualEncoder().to(device)

    # Optimization setup
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

    # Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    logger.info("Starting training...")

    for epoch in range(Config.EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, criterion
        )
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        logger.info(f"Epoch {epoch+1}/{Config.EPOCHS}")
        logger.info(f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f}")
        logger.info(f"Val Loss:   {val_loss:.6f} | Val Acc:   {val_acc:.6f}")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Ensure directory exists
            os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            logger.info(f"New best model saved to {Config.MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            logger.info("Early stopping triggered.")
            break

    return model


def generate_submission(test_loader):
    """
    Loads the best model, performs inference on the test set, and saves predictions.
    """
    device = Config.DEVICE
    model = SiameseDualEncoder().to(device)

    # Load best weights
    if os.path.exists(Config.MODEL_SAVE_PATH):
        logger.info(f"Loading best model from {Config.MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        logger.warning(
            "No checkpoint found. Using random weights (Warning: Predictions will be random)."
        )

    model.eval()

    all_ids = []
    all_probs = []

    logger.info("Generating predictions on test set...")

    with torch.no_grad():
        for batch in test_loader:
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)

            # Retrieve IDs
            ids = batch["id"]
            if isinstance(ids, torch.Tensor):
                ids = ids.cpu().numpy()

            logits = model(input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            all_ids.extend(ids)
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)

    # Create Submission DataFrame
    # Mapping: 0 -> winner_model_a, 1 -> winner_model_b, 2 -> winner_tie
    df_sub = pd.DataFrame(
        {
            "id": all_ids,
            "winner_model_a": all_probs[:, 0],
            "winner_model_b": all_probs[:, 1],
            "winner_tie": all_probs[:, 2],
        }
    )

    # Ensure correct column order
    df_sub = df_sub[["id", "winner_model_a", "winner_model_b", "winner_tie"]]

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission file saved to {Config.SUBMISSION_PATH}")
