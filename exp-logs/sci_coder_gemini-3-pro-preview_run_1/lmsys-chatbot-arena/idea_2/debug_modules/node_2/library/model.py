import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from library.config import Config
from library.utils import setup_logger, compute_metric

# Initialize logger
logger = setup_logger("model_module")


class SiameseDeberta(nn.Module):
    """
    Siamese Transformer with Hybrid Feature Fusion.

    Architecture:
    1. Shared DeBERTa backbone encodes (Prompt + Response A) and (Prompt + Response B).
    2. Extracts [CLS] tokens as representations u and v.
    3. Computes interaction features: |u - v| and u * v.
    4. Fuses [u, v, |u-v|, u*v] with explicit scalar features.
    5. MLP Head predicts probabilities for [Winner A, Winner B, Tie].
    """

    def __init__(self):
        super().__init__()
        # Load Configuration and Backbone
        self.config = AutoConfig.from_pretrained(Config.model_name)
        self.backbone = AutoModel.from_pretrained(Config.model_name)

        # Dimensions
        self.hidden_size = self.config.hidden_size
        self.num_scalar = Config.num_scalar_features
        self.num_classes = Config.num_classes

        # Interaction: u, v, |u-v|, u*v -> 4 vectors
        # Input dimension for the fusion layer
        self.fusion_input_dim = (4 * self.hidden_size) + self.num_scalar

        # MLP Classification Head
        self.head = nn.Sequential(
            nn.Linear(self.fusion_input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.num_classes),
        )

        # Initialize head weights
        self._init_weights(self.head)

    def _init_weights(self, module):
        """Initialize weights for the custom MLP head."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids_a,
        attention_mask_a,
        input_ids_b,
        attention_mask_b,
        scalar_features,
    ):
        """
        Forward pass for the Siamese Network.

        Args:
            input_ids_a, attention_mask_a: Inputs for Branch A
            input_ids_b, attention_mask_b: Inputs for Branch B
            scalar_features: Tensor of auxiliary scalar features (batch, num_scalar)

        Returns:
            logits: Unnormalized scores (batch, num_classes)
        """
        # --- Branch A ---
        out_a = self.backbone(input_ids=input_ids_a, attention_mask=attention_mask_a)
        u = out_a.last_hidden_state[:, 0, :]  # Extract CLS token

        # --- Branch B ---
        out_b = self.backbone(input_ids=input_ids_b, attention_mask=attention_mask_b)
        v = out_b.last_hidden_state[:, 0, :]  # Extract CLS token

        # --- Semantic Interaction ---
        diff = torch.abs(u - v)
        prod = u * v

        # --- Hybrid Fusion ---
        # Concatenate: [u, v, |u-v|, u*v, scalar_features]
        # Ensure scalar_features is on the same device and dtype
        fused = torch.cat([u, v, diff, prod, scalar_features], dim=1)

        # --- Classification ---
        logits = self.head(fused)

        return logits


def train_model(
    model, train_loader, val_loader, epochs=Config.epochs, device=Config.device
):
    """
    Executes the training pipeline with differential learning rates and early stopping.
    """
    model.to(device)

    # Differential Learning Rates: Lower for backbone, higher for head
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n],
            "lr": Config.lr_backbone,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "lr": Config.lr_head,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, weight_decay=Config.weight_decay
    )

    # CrossEntropyLoss works with soft targets (probabilities) which matches our data
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        logger.info(f"Starting Epoch {epoch + 1}/{epochs}")

        # --- Training Phase ---
        model.train()
        train_losses = []

        for batch in train_loader:
            # Move data to device
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            scalar_features = batch["scalar_features"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            logits = model(
                input_ids_a,
                attention_mask_a,
                input_ids_b,
                attention_mask_b,
                scalar_features,
            )
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # --- Validation Phase ---
        model.eval()
        val_losses = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids_a = batch["input_ids_a"].to(device)
                attention_mask_a = batch["attention_mask_a"].to(device)
                input_ids_b = batch["input_ids_b"].to(device)
                attention_mask_b = batch["attention_mask_b"].to(device)
                scalar_features = batch["scalar_features"].to(device)
                labels = batch["label"].to(device)

                logits = model(
                    input_ids_a,
                    attention_mask_a,
                    input_ids_b,
                    attention_mask_b,
                    scalar_features,
                )
                loss = criterion(logits, labels)
                val_losses.append(loss.item())

                # Store predictions for metric calculation
                probs = F.softmax(logits, dim=1)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_val_loss = np.mean(val_losses)

        # Compute Validation Metric (Log Loss)
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_labels)
        metric_score = compute_metric(y_true, y_pred)

        logger.info(
            f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val Metric: {metric_score:.6f}"
        )

        # --- Early Stopping & Checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            logger.info(
                f"Validation loss improved. Saving model to {Config.model_save_path}"
            )
            torch.save(model.state_dict(), Config.model_save_path)
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: {patience_counter}/{Config.patience}"
            )

        if patience_counter >= Config.patience:
            logger.info("Early stopping triggered.")
            break

    logger.info("Training complete.")


def inference(model, test_loader, device=Config.device):
    """
    Runs inference on the test set and saves the submission file.
    """
    # Load best weights
    if os.path.exists(Config.model_save_path):
        logger.info(f"Loading best model weights from {Config.model_save_path}")
        model.load_state_dict(torch.load(Config.model_save_path, map_location=device))
    else:
        logger.warning("No saved model found. Using current weights for inference.")

    model.to(device)
    model.eval()

    all_probs = []

    logger.info("Starting inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            scalar_features = batch["scalar_features"].to(device)

            logits = model(
                input_ids_a,
                attention_mask_a,
                input_ids_b,
                attention_mask_b,
                scalar_features,
            )
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    # Concatenate all predictions
    final_probs = np.vstack(all_probs)

    # --- Generate Submission File ---
    # Load test metadata to retrieve IDs
    df_test = pd.read_csv(Config.test_path)

    # Handle Debug Mode: Ensure IDs match the subset used in Dataset
    if hasattr(Config, "debug") and Config.debug:
        logger.info("Debug mode: Subsetting test IDs to match inference.")
        df_test = df_test.head(50)

    if len(df_test) != len(final_probs):
        logger.error(
            f"Mismatch: Test IDs ({len(df_test)}) vs Predictions ({len(final_probs)})"
        )
        # Proceeding assuming the dataloader was sequential and deterministic

    submission = pd.DataFrame(
        {
            "id": df_test["id"],
            "winner_model_a": final_probs[:, 0],
            "winner_model_b": final_probs[:, 1],
            "winner_tie": final_probs[:, 2],
        }
    )

    logger.info(f"Saving submission file to {Config.submission_path}")
    submission.to_csv(Config.submission_path, index=False)
    logger.info("Submission generation complete.")
