import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.data_utils import get_dataloaders


class HybridTransformer(nn.Module):
    """
    Positional-Aware Hybrid Transformer Architecture.

    Processes sequence data via a Transformer Encoder and flattens the output to
    preserve positional rigidity. Processes numerical data via a dense projection.
    Fuses both streams into a high-capacity MLP.
    """

    def __init__(self):
        super(HybridTransformer, self).__init__()

        # --- Sequence Branch ---
        # +1 for padding index 0, though Config.VOCAB_SIZE should already account for it
        self.embedding = nn.Embedding(
            num_embeddings=Config.VOCAB_SIZE,
            embedding_dim=Config.EMBED_DIM,
            padding_idx=0,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.EMBED_DIM,
            nhead=Config.TRANSFORMER_HEADS,
            dim_feedforward=Config.TRANSFORMER_FF_DIM,
            dropout=Config.TRANSFORMER_DROPOUT,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.TRANSFORMER_LAYERS
        )

        # --- Numerical Branch ---
        # Projects numerical features to the same latent dimension as one token embedding
        self.num_proj = nn.Sequential(
            nn.Linear(len(Config.NUM_FEATURES), Config.EMBED_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # --- Fusion & MLP ---
        # GAP Sequence: Embed_Dim
        # Projected Numerical: Embed_Dim
        fusion_input_dim = Config.EMBED_DIM + Config.EMBED_DIM

        mlp_layers = []
        in_dim = fusion_input_dim

        for hidden_dim in Config.MLP_HIDDEN_DIMS:
            mlp_layers.append(nn.Linear(in_dim, hidden_dim))
            mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(Config.MLP_DROPOUT))
            in_dim = hidden_dim

        mlp_layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x_seq, x_num):
        # x_seq: (Batch, Seq_Len)
        # x_num: (Batch, Num_Features)

        # 1. Sequence Processing
        # Create padding mask: True where value is 0 (padding)
        # Transformer expects (Batch, Seq_Len) for src_key_padding_mask
        padding_mask = x_seq == 0

        emb = self.embedding(x_seq)  # (Batch, Seq_Len, Embed_Dim)

        # Transformer Encoder
        trans_out = self.transformer(
            emb, src_key_padding_mask=padding_mask
        )  # (Batch, Seq_Len, Embed_Dim)

        # Global Average Pooling (GAP)
        # Cite solution_lesson_node_00014: GAP Superior to Flattening in Hybrid Transformer-MLP Fusion
        seq_pooled = torch.mean(trans_out, dim=1)  # (Batch, Embed_Dim)

        # 2. Numerical Processing
        num_emb = self.num_proj(x_num)  # (Batch, Embed_Dim)

        # 3. Fusion
        combined = torch.cat([seq_pooled, num_emb], dim=1)

        # 4. Classification
        logits = self.mlp(combined)
        return logits


def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for x_seq, x_num, targets in loader:
        x_seq = x_seq.to(device)
        x_num = x_num.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(x_seq, x_num)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * x_seq.size(0)

        # Store for AUC
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.append(probs)
        all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for x_seq, x_num, targets in loader:
            x_seq = x_seq.to(device)
            x_num = x_num.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(x_seq, x_num)
            loss = criterion(logits, targets)

            running_loss += loss.item() * x_seq.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    try:
        val_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        val_auc = 0.5

    return val_loss, val_auc


def run_training(train_loader, val_loader):
    print("Initializing Model...")
    device = torch.device(Config.DEVICE)
    model = HybridTransformer().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss, train_auc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.5f} | Train AUC: {train_auc:.5f} | "
            f"Val Loss: {val_loss:.5f} | Val AUC: {val_auc:.10f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  -> New Best Model Saved (AUC: {best_val_auc:.10f})")

    print(f"Training Complete. Best Validation AUC: {best_val_auc:.10f}")
    return best_val_auc


def generate_submission(test_loader):
    print("Generating Submission...")
    device = torch.device(Config.DEVICE)

    # Load Best Model
    model = HybridTransformer().to(device)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        print("Loaded best model checkpoint.")
    else:
        print("Warning: No checkpoint found. Using untrained model.")

    model.eval()
    all_preds = []

    with torch.no_grad():
        for x_seq, x_num in test_loader:
            x_seq = x_seq.to(device)
            x_num = x_num.to(device)

            logits = model(x_seq, x_num)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)

    predictions = np.concatenate(all_preds).flatten()

    # Load Test IDs
    # We read the test CSV directly to get IDs.
    # The DataLoader is sequential (shuffle=False), so order matches.
    df_test = pd.read_csv(Config.TEST_PATH)
    ids = df_test["id"].values

    if len(ids) != len(predictions):
        raise ValueError(f"Mismatch: {len(ids)} IDs vs {len(predictions)} predictions")

    submission = pd.DataFrame({"id": ids, "target": predictions})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    # 1. Get Data
    train_loader, val_loader, test_loader = get_dataloaders()

    # 2. Train
    run_training(train_loader, val_loader)

    # 3. Predict
    generate_submission(test_loader)
