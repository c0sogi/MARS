import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModel, AutoConfig, get_linear_schedule_with_warmup
from library.config import Config
from library.data_processing import create_dataloaders
from library.metrics import calculate_final_score


class MultiTaskTransformer(nn.Module):
    """
    Transformer-based model with Multi-Task Learning heads.

    Backbone: Pre-trained Transformer (e.g., DistilBERT)
    Head 1: Toxicity Classification (Scalar)
    Head 2: Identity Attribute Prediction (Vector)
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        num_identity_labels=Config.NUM_IDENTITY_LABELS,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

        # Primary Head: Toxicity (Scalar output)
        self.toxicity_head = nn.Linear(self.config.hidden_size, 1)

        # Auxiliary Head: Identity Attributes (Vector output)
        self.identity_head = nn.Linear(self.config.hidden_size, num_identity_labels)

    def forward(self, input_ids, attention_mask):
        # Pass through backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Extract representation
        # DistilBERT usually returns last_hidden_state.
        # We take the embedding of the [CLS] token (first token).
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]

        pooled_output = self.dropout(pooled_output)

        # Compute logits for both heads
        toxicity_logits = self.toxicity_head(pooled_output)
        identity_logits = self.identity_head(pooled_output)

        return toxicity_logits, identity_logits


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0

    # Loss functions
    # Toxicity: Weighted BCE (reduction='none' to apply sample weights)
    tox_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    # Identity: Standard BCE
    ident_loss_fn = nn.BCEWithLogitsLoss()

    for batch in dataloader:
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device)
        aux_targets = batch["aux_target"].to(device)
        weights = batch["weight"].to(device)

        optimizer.zero_grad()

        # Forward pass
        tox_logits, ident_logits = model(input_ids, mask)

        # Calculate Toxicity Loss
        # Squeeze logits to match target shape [batch_size]
        loss_tox = tox_loss_fn(tox_logits.squeeze(-1), targets)
        # Apply sample weights (bias mitigation)
        loss_tox = (loss_tox * weights).mean()

        # Calculate Identity Loss
        loss_ident = ident_loss_fn(ident_logits, aux_targets)

        # Total Loss (Multi-task objective)
        loss = loss_tox + (Config.AUX_LOSS_WEIGHT * loss_ident)

        # Backward pass
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """
    Evaluates the model and returns predictions and targets.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_identities = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            tox_logits, _ = model(input_ids, mask)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(tox_logits).squeeze(-1).cpu().numpy()
            all_preds.extend(probs)

            if "target" in batch:
                all_targets.extend(batch["target"].numpy())
                all_identities.extend(batch["aux_target"].numpy())

    return np.array(all_preds), np.array(all_targets), np.array(all_identities)


def run_training(load_cached_data=True, data_limit=None):
    """
    Main training pipeline:
    1. Loads data
    2. Initializes model/optimizer
    3. Runs training loop with Early Stopping
    4. Saves best model
    5. Generates submission
    """
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Load Data
    train_loader, val_loader, test_loader = create_dataloaders(
        load_cached_data=load_cached_data, data_limit=data_limit
    )

    # 2. Initialize Model
    model = MultiTaskTransformer().to(device)

    # Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_training_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=Config.WARMUP_STEPS,
        num_training_steps=num_training_steps,
    )

    # 3. Training Loop
    best_score = -float("inf")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.6f}")

        # Validate
        val_preds, val_targets, val_identities = evaluate(model, val_loader, device)

        # Reconstruct Validation DataFrame for Metrics
        val_eval_df = pd.DataFrame({"target": val_targets, "prediction": val_preds})
        # Add identity columns back for bias metric calculation
        # Note: val_identities is shape [N, num_identities]
        for idx, col in enumerate(Config.IDENTITY_COLUMNS):
            val_eval_df[col] = val_identities[:, idx]

        # Calculate Metrics
        metrics = calculate_final_score(val_eval_df)
        score = metrics["score"]

        print(f"Validation Score: {score:.6f}")
        print(f"Overall AUC: {metrics['overall_auc']:.6f}")
        print(f"Mean Subgroup AUC: {metrics['mean_subgroup_auc']:.6f}")
        print(f"Mean BPSN AUC: {metrics['mean_bpsn_auc']:.6f}")
        print(f"Mean BNSP AUC: {metrics['mean_bnsp_auc']:.6f}")

        # Early Stopping & Checkpointing
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved to {Config.MODEL_SAVE_PATH}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # 4. Generate Submission
    print("\nGenerating submission for Test set...")
    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    test_preds, _, _ = evaluate(model, test_loader, device)

    # Load sample submission to ensure correct ID alignment
    submission = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # If debugging with data_limit, truncate submission df
    if len(test_preds) != len(submission):
        print(
            f"Note: Prediction count ({len(test_preds)}) matches subset, truncating submission DataFrame."
        )
        submission = submission.iloc[: len(test_preds)]

    submission["prediction"] = test_preds
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
