import os
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm

from library.config import Config
from library.dataset import CervicalSpineDataset
from library.utils import seed_everything

# =========================================================================
# Model Architecture
# =========================================================================


class CervicalSpineTransformer(nn.Module):
    def __init__(self):
        super(CervicalSpineTransformer, self).__init__()

        # 1. Backbone: EfficientNet-B4
        # num_classes=0 ensures we get the pooled feature vector (Global Average Pooling applied by timm)
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.BACKBONE_PRETRAINED,
            num_classes=0,
            in_chans=Config.IN_CHANNELS,
        )
        self.backbone_dim = self.backbone.num_features

        # Projection to hidden dimension
        self.projection = nn.Linear(self.backbone_dim, Config.HIDDEN_DIM)
        self.act = nn.ReLU()

        # 2. Sequence Encoder: Bidirectional LSTM
        # We halve the hidden size for bidirectional so the concatenated output matches HIDDEN_DIM
        lstm_hidden = (
            Config.HIDDEN_DIM // 2 if Config.BIDIRECTIONAL else Config.HIDDEN_DIM
        )
        self.lstm = nn.LSTM(
            input_size=Config.HIDDEN_DIM,
            hidden_size=lstm_hidden,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=Config.BIDIRECTIONAL,
            dropout=Config.LSTM_DROPOUT if Config.LSTM_LAYERS > 1 else 0,
        )

        # Learnable Positional Embeddings for the image sequence
        self.seq_pos_embed = nn.Parameter(
            torch.zeros(1, Config.SEQ_LEN, Config.HIDDEN_DIM)
        )

        # 3. Transformer Decoder
        # Learnable Queries for the 8 targets (C1-C7, Patient Overall)
        self.query_embed = nn.Parameter(
            torch.zeros(1, Config.NUM_QUERIES, Config.HIDDEN_DIM)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=Config.HIDDEN_DIM,
            nhead=Config.NHEAD,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=Config.DROPOUT,
            activation=Config.ACTIVATION,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=Config.NUM_DECODER_LAYERS
        )

        # 4. Classification Head
        self.classifier = nn.Linear(Config.HIDDEN_DIM, 1)

        self._init_weights()

    def _init_weights(self):
        # Initialize embeddings with normal distribution
        nn.init.normal_(self.seq_pos_embed, std=0.02)
        nn.init.normal_(self.query_embed, std=0.02)

        # Initialize Linear layers
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.constant_(self.projection.bias, 0)

        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Channels, Height, Width)
        Returns:
            logits: Output tensor of shape (Batch, Num_Queries)
        """
        b, s, c, h, w = x.shape

        # --- Backbone Feature Extraction ---
        # Flatten batch and sequence dimensions to process slices in parallel
        x = x.view(b * s, c, h, w)
        features = self.backbone(x)  # (B*S, Backbone_Dim)

        # Project and reshape back to sequence format
        features = self.projection(features)
        features = self.act(features)
        features = features.view(b, s, Config.HIDDEN_DIM)  # (B, S, Hidden_Dim)

        # --- Sequence Encoding ---
        # LSTM processes the sequence along the Z-axis
        lstm_out, _ = self.lstm(features)  # (B, S, Hidden_Dim)

        # Add Positional Embeddings (broadcasted across batch)
        memory = lstm_out + self.seq_pos_embed

        # --- Transformer Decoding ---
        # Expand queries to match batch size
        queries = self.query_embed.expand(b, -1, -1)  # (B, Num_Queries, Hidden_Dim)

        # Decoder: Queries attend to the Sequence Memory
        # Output: (B, Num_Queries, Hidden_Dim)
        out = self.decoder(tgt=queries, memory=memory)

        # --- Classification ---
        # Project to logits
        logits = self.classifier(out).squeeze(-1)  # (B, Num_Queries)

        return logits


# =========================================================================
# Training & Evaluation Functions
# =========================================================================


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Gradient Accumulation setup
    accumulation_steps = Config.ACCUMULATION_STEPS
    optimizer.zero_grad()

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)

        # Forward pass
        logits = model(images)
        loss = criterion(logits, targets)

        # Scale loss for accumulation
        loss = loss / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

        running_loss += loss.item() * accumulation_steps * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def predict_test_set(model, loader, device):
    model.eval()
    predictions = []
    study_ids = []

    # We need to map the 8 outputs to their names
    # Order matches the dataset target construction: C1-C7, then Patient Overall
    col_names = [f"C{i}" for i in range(1, 8)] + ["patient_overall"]

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.to(device)

            # Get Study UIDs from the dataset
            # The loader returns batches, so we need to fetch the corresponding UIDs
            # We assume the loader is sequential and not shuffled for test
            start_idx = i * loader.batch_size
            end_idx = start_idx + images.size(0)
            batch_uids = loader.dataset.df.iloc[start_idx:end_idx][
                "StudyInstanceUID"
            ].values

            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            for uid, prob_row in zip(batch_uids, probs):
                for col_name, prob in zip(col_names, prob_row):
                    row_id = f"{uid}_{col_name}"
                    predictions.append({"row_id": row_id, "fractured": prob})

    return pd.DataFrame(predictions)


def run(epochs=Config.EPOCHS, debug=Config.DEBUG):
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(f"Running on device: {device}")

    # --- Data Loading ---
    train_dataset = CervicalSpineDataset(split="train", debug=debug)
    val_dataset = CervicalSpineDataset(split="val", debug=debug)
    test_dataset = CervicalSpineDataset(split="test", debug=debug)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Model Setup ---
    model = CervicalSpineTransformer().to(device)

    # Loss: Weighted BCE
    # Using pos_weight to handle class imbalance (fractures are rare)
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs * len(train_loader) // Config.ACCUMULATION_STEPS,
        eta_min=Config.ETA_MIN,
    )

    # --- Training Loop ---
    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 3
    counter = 0

    print("Starting training...")

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        val_loss = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{epochs} | Time: {elapsed:.0f}s")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
            print(f"New best model saved to {Config.CHECKPOINT_PATH}")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    # --- Inference ---
    print("Loading best model for inference...")
    model.load_state_dict(best_model_wts)

    print("Generating predictions on test set...")
    submission_df = predict_test_set(model, test_loader, device)

    # Save submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
