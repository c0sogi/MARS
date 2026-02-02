import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from library.config import Config

# =========================================================================
# Model Definition
# =========================================================================


class SingleStream(nn.Module):
    """
    A single stream of the IPPFE architecture.
    Features:
    - Independent Embeddings for categorical variables.
    - Independent Linear Projection for continuous variables (The Innovation).
    - Funnel MLP Backbone with ReLU and Dropout.
    """

    def __init__(self, vocab_sizes, num_cont, embed_dim, hidden_dims, dropout):
        super().__init__()

        # Independent Embeddings
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=v, embedding_dim=embed_dim)
                for v in vocab_sizes
            ]
        )

        # Independent Feature Transformation Layer (FTL)
        # Projects continuous features: Input Dim -> Input Dim
        self.cont_proj = nn.Linear(num_cont, num_cont)

        # Calculate concatenated input dimension
        # (Num Categorical * Embed Dim) + Num Continuous
        input_dim = (len(vocab_sizes) * embed_dim) + num_cont

        # Build Funnel MLP
        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if Config.ACTIVATION == "ReLU":
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        # Output Layer (Binary Logit)
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, cat_x, cont_x):
        # cat_x: (Batch, Num_Cat_Features)
        # cont_x: (Batch, Num_Cont_Features)

        # 1. Process Embeddings
        emb_list = []
        for i, emb_layer in enumerate(self.embeddings):
            # cat_x[:, i] is the i-th categorical feature column
            emb_list.append(emb_layer(cat_x[:, i]))

        # Concatenate embeddings: (Batch, Num_Cat * Embed_Dim)
        x_cat = torch.cat(emb_list, dim=1)

        # 2. Project Continuous Features
        x_cont = self.cont_proj(cont_x)

        # 3. Early Fusion
        x = torch.cat([x_cat, x_cont], dim=1)

        # 4. Backbone
        logit = self.mlp(x)

        return logit


class IPPFEModel(nn.Module):
    """
    Independent-Projection Parallel Funnel Ensemble.
    Contains 5 independent streams within a single graph.
    """

    def __init__(self, vocab_sizes, num_cont):
        super().__init__()
        self.streams = nn.ModuleList()

        # Instantiate heterogeneous streams based on Config
        for stream_cfg in Config.STREAMS:
            self.streams.append(
                SingleStream(
                    vocab_sizes=vocab_sizes,
                    num_cont=num_cont,
                    embed_dim=Config.EMBEDDING_DIM,
                    hidden_dims=stream_cfg["hidden_dims"],
                    dropout=stream_cfg["dropout"],
                )
            )

    def forward(self, cat_x, cont_x):
        # Collect logits from all streams
        logits_list = []
        for stream in self.streams:
            logits_list.append(stream(cat_x, cont_x))

        # Concatenate to shape (Batch, 5)
        return torch.cat(logits_list, dim=1)


# =========================================================================
# Training & Evaluation Logic
# =========================================================================


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        cat_x = batch["cat_features"].to(device)
        cont_x = batch["cont_features"].to(device)
        targets = batch["target"].to(device)  # Shape: (Batch,)

        optimizer.zero_grad()

        # Forward pass: (Batch, 5)
        logits = model(cat_x, cont_x)

        # Calculate loss
        # We sum the BCE loss for each stream independently
        # Expand targets to (Batch, 5) to match logits
        targets_expanded = targets.unsqueeze(1).repeat(1, 5)
        loss = criterion(logits, targets_expanded)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * cat_x.size(0)

    return running_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            cat_x = batch["cat_features"].to(device)
            cont_x = batch["cont_features"].to(device)
            targets = batch["target"].to(device)

            logits = model(cat_x, cont_x)

            # Loss calculation
            targets_expanded = targets.unsqueeze(1).repeat(1, 5)
            loss = criterion(logits, targets_expanded)
            running_loss += loss.item() * cat_x.size(0)

            # Prediction for AUC: Average of probabilities
            probs = torch.sigmoid(logits)  # (Batch, 5)
            avg_probs = torch.mean(probs, dim=1)

            all_preds.append(avg_probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    epoch_auc = roc_auc_score(all_targets, all_preds)
    epoch_loss = running_loss / len(dataloader.dataset)

    return epoch_loss, epoch_auc


def train_model(train_loader, val_loader, metadata):
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Initialize Model
    model = IPPFEModel(
        vocab_sizes=metadata["vocab_sizes"], num_cont=metadata["num_cont_features"]
    ).to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.MAX_LR,  # Initial LR is handled by OneCycleLR, this is max
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Loss Function (BCEWithLogitsLoss is more stable)
    # We sum the loss over the 5 streams (reduction='mean' averages over batch, we want that)
    criterion = nn.BCEWithLogitsLoss()

    # Scheduler
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        total_steps=total_steps,
        pct_start=Config.PCT_START,
    )

    best_auc = 0.0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.10f} | "
            f"Val Loss: {val_loss:.10f} | "
            f"Val AUC: {val_auc:.10f}"
        )

        # Checkpoint Best Model
        # We save the best model but continue training to allow full convergence (per strategy)
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_PATH)
            # print(f"New best model saved with AUC: {best_auc:.10f}")

    print(f"Training complete. Best Val AUC: {best_auc:.10f}")


def predict_model(test_loader, metadata):
    device = torch.device(Config.DEVICE)

    # Initialize Model Structure
    model = IPPFEModel(
        vocab_sizes=metadata["vocab_sizes"], num_cont=metadata["num_cont_features"]
    ).to(device)

    # Load Best Weights
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    predictions = []

    # Load sample submission to get IDs
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    ids = sample_sub[Config.ID_COL].values

    with torch.no_grad():
        for batch in test_loader:
            cat_x = batch["cat_features"].to(device)
            cont_x = batch["cont_features"].to(device)

            logits = model(cat_x, cont_x)
            probs = torch.sigmoid(logits)

            # Ensemble Strategy: Arithmetic Mean of 5 streams
            avg_probs = torch.mean(probs, dim=1)
            predictions.extend(avg_probs.cpu().numpy())

    # Ensure lengths match
    if len(predictions) != len(ids):
        # This might happen if test_loader drops last or has different size
        # But Config.SAMPLE_SUBMISSION_PATH should match test set size
        # We assume standard behavior here.
        print(f"Warning: Prediction count {len(predictions)} != ID count {len(ids)}")
        # Truncate or pad if absolutely necessary, but usually indicates an error
        min_len = min(len(predictions), len(ids))
        predictions = predictions[:min_len]
        ids = ids[:min_len]

    # Create Submission DataFrame
    submission_df = pd.DataFrame({Config.ID_COL: ids, Config.TARGET_COL: predictions})

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
