import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoConfig
from library.config import Config
from library.utils import compute_pearson_correlation


class CrossEncoderModel(nn.Module):
    """
    Cross-Encoder model using a Transformer backbone for regression.
    """

    def __init__(
        self,
        model_name=Config.model_name,
        num_labels=Config.num_labels,
        dropout=Config.dropout,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(
        self, input_ids, attention_mask, token_type_ids=None, labels=None, **kwargs
    ):
        # Pass inputs through the backbone
        # Note: DeBERTa V3 might not use token_type_ids, but we accept them if provided
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Extract [CLS] token representation (index 0 of last_hidden_state)
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]

        # Apply dropout and regression head
        features = self.dropout(cls_embedding)
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            # Flatten to ensure shapes match: [batch_size] vs [batch_size]
            loss = loss_fct(logits.view(-1), labels.view(-1))

        return {"loss": loss, "logits": logits}


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs=Config.epochs,
    patience=Config.patience,
    save_path=Config.model_save_path,
):
    """
    Training loop with validation, early stopping, and model checkpointing.
    """
    model.to(device)
    best_pearson = -1.0
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        # ==========================
        # Training Phase
        # ==========================
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Handle optional token_type_ids
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )

            loss = outputs["loss"]
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # ==========================
        # Validation Phase
        # ==========================
        model.eval()
        val_preds = []
        val_targets = []
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                )

                loss = outputs["loss"]
                total_val_loss += loss.item()

                # Collect predictions and targets
                logits = outputs["logits"].detach().cpu().numpy().flatten()
                label_ids = labels.detach().cpu().numpy().flatten()

                val_preds.extend(logits)
                val_targets.extend(label_ids)

        avg_val_loss = total_val_loss / len(val_loader)

        # Compute Metric
        current_pearson = compute_pearson_correlation(val_preds, val_targets)

        # Print Metrics (Full Precision)
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss}")
        print(f"Val Loss: {avg_val_loss}")
        print(f"Val Pearson: {current_pearson}")

        # ==========================
        # Early Stopping & Saving
        # ==========================
        if current_pearson > best_pearson:
            best_pearson = current_pearson
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Load the best model weights before returning
    if os.path.exists(save_path):
        print(f"Loading best model from {save_path}...")
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model


def generate_submission(
    model, test_loader, device, submission_path=Config.submission_path
):
    """
    Generates predictions for the test set and saves them to a CSV file.
    """
    model.to(device)
    model.eval()

    predictions = []
    ids = []

    print("Generating predictions for test set...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            # Extract IDs (preserved by CustomCollator)
            batch_ids = batch["id"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            logits = outputs["logits"].detach().cpu().numpy().flatten()

            # Clip predictions to valid range [0, 1]
            logits = np.clip(logits, 0.0, 1.0)

            predictions.extend(logits)
            ids.extend(batch_ids)

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"id": ids, "score": predictions})

    # Save to CSV
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
