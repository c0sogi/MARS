import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from library.config import ModelConfig
from library.utils import save_checkpoint, load_checkpoint
from library.dataset import get_dataloaders


class GatedBlock(nn.Module):
    """
    Residual Gated Block with GLU activation.
    Structure: x_out = x_in + Dropout(BatchNorm(GLU(Linear(x_in))))
    """

    def __init__(self, input_dim, dropout_rate):
        super(GatedBlock, self).__init__()
        # Linear layer maps to 2x dim for GLU splitting
        self.linear = nn.Linear(input_dim, input_dim * 2)
        self.norm = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x

        # Projection
        x = self.linear(x)

        # GLU: Split into value and gate branches
        val, gate = x.chunk(2, dim=1)
        # Sigmoid gating as per description
        x = val * torch.sigmoid(gate)

        # Normalization & Regularization
        x = self.norm(x)
        x = self.dropout(x)

        # Residual Connection
        return x + residual


class ResFunnelGLU(nn.Module):
    """
    Residual Funnel Gated Network.

    Features:
    - Categorical Embeddings
    - Funnel Architecture (decreasing width)
    - Residual Gated Blocks
    """

    def __init__(self):
        super(ResFunnelGLU, self).__init__()

        # Configuration
        self.vocab_size = ModelConfig.VOCAB_SIZE
        self.embed_dim = ModelConfig.EMBED_DIM
        self.seq_len = ModelConfig.SEQUENCE_LENGTH
        self.num_cont = ModelConfig.NUM_CONT_FEATURES
        self.hidden_dims = ModelConfig.HIDDEN_DIMS
        self.dropout_rate = ModelConfig.DROPOUT_RATE

        # 1. Input Processing
        # Embedding layer for character tokens
        self.embedding = nn.Embedding(self.vocab_size + 1, self.embed_dim)

        # Calculate dimension of the raw concatenated input
        # (10 chars * 32 dim) + 31 continuous features
        self.raw_input_dim = (self.seq_len * self.embed_dim) + self.num_cont

        # 2. Backbone Stages
        self.downsamples = nn.ModuleList()
        self.stages = nn.ModuleList()

        # Track input dimension for the next layer
        prev_dim = self.raw_input_dim

        for i, dim in enumerate(self.hidden_dims):
            # Downsampling / Projection Layer
            # For Stage 0, this projects Raw Input -> Hidden[0]
            # For Stage > 0, this projects Hidden[i-1] -> Hidden[i]
            self.downsamples.append(nn.Linear(prev_dim, dim))

            # Stacked Gated Blocks
            # Using 3 blocks per stage for depth
            blocks = nn.Sequential(
                GatedBlock(dim, self.dropout_rate),
                GatedBlock(dim, self.dropout_rate),
                GatedBlock(dim, self.dropout_rate),
            )
            self.stages.append(blocks)

            prev_dim = dim

        # 3. Output Head
        self.head = nn.Linear(self.hidden_dims[-1], 1)

    def forward(self, cont_data, cat_data):
        # Embed Categorical Features
        # cat_data: (B, 10) -> (B, 10, 32)
        embs = self.embedding(cat_data)
        # Flatten embeddings: (B, 320)
        embs_flat = embs.view(embs.size(0), -1)

        # Concatenate with Continuous Features to form Raw Input
        # (B, 320) + (B, 31) -> (B, 351)
        x = torch.cat([embs_flat, cont_data], dim=1)

        # Iterate through stages
        for i in range(len(self.hidden_dims)):
            # 1. Downsample / Project
            x = self.downsamples[i](x)

            # 2. Process through Gated Blocks
            x = self.stages[i](x)

        # Output Head
        logits = self.head(x)
        return logits


# ------------------------------------------------------------------------------
# Training & Inference Pipeline
# ------------------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch in loader:
        cont = batch["cont"].to(device)
        cat = batch["cat"].to(device)
        target = batch["target"].to(device).view(-1, 1)

        optimizer.zero_grad()

        logits = model(cont, cat)
        loss = criterion(logits, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            cont = batch["cont"].to(device)
            cat = batch["cat"].to(device)
            target = batch["target"].to(device).view(-1, 1)

            logits = model(cont, cat)
            loss = criterion(logits, target)

            running_loss += loss.item()

            # Apply sigmoid for AUC calculation
            probs = torch.sigmoid(logits)

            all_targets.append(target.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    auc = roc_auc_score(all_targets, all_preds)
    return running_loss / len(loader), auc


def predict(model, loader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            cont = batch["cont"].to(device)
            cat = batch["cat"].to(device)

            logits = model(cont, cat)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds).flatten()


def run_pipeline():
    """
    Executes the full training and submission pipeline.
    """
    # 1. Setup
    config = ModelConfig
    device = torch.device(config.DEVICE)
    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True, debug=config.DEBUG
    )

    # 3. Model Initialization
    model = IIResFunnelGLU().to(device)

    # 4. Optimization
    # Using AdamW with high weight decay as per Idea
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    # BCEWithLogitsLoss combines Sigmoid + BCE for numerical stability
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    print("Starting training...")
    best_auc = 0.0
    patience = 0

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val AUC: {val_auc}"
        )

        # Checkpoint based on AUC (decoupled from loss)
        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0
            save_checkpoint(
                model, optimizer, None, epoch, val_auc, config.MODEL_SAVE_PATH
            )
            print(f"New best model saved! AUC: {best_auc}")
        else:
            patience += 1
            if patience >= config.EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping triggered after {patience} epochs without improvement."
                )
                break

    # 6. Submission Generation
    print("Generating submission...")
    # Load best model
    checkpoint = load_checkpoint(config.MODEL_SAVE_PATH, model, device=device)
    print(
        f"Loaded best model from epoch {checkpoint['epoch']} with AUC {checkpoint['metric']}"
    )

    predictions = predict(model, test_loader, device)

    submission = pd.DataFrame({"id": test_ids, "target": predictions})

    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
