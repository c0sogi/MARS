import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.dataset import get_dataloaders, set_seed


class ResidualGatedBlock(nn.Module):
    """
    A residual block with Gated Linear Unit (GLU) activation.
    Structure: Input -> Linear(d->2d) -> GLU(2d->d) -> BatchNorm -> Dropout -> + Input
    """

    def __init__(self, dim, dropout_rate):
        super(ResidualGatedBlock, self).__init__()
        self.linear = nn.Linear(dim, dim * 2)
        self.glu = nn.GLU(dim=1)
        self.bn = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        out = self.linear(x)
        out = self.glu(out)
        out = self.bn(out)
        out = self.dropout(out)
        return out + residual


class ResGLUNet(nn.Module):
    """
    Deep Residual Gated Network for tabular data.
    Combines learned embeddings for categorical tokens and normalized continuous features.
    """

    def __init__(self):
        super(ResGLUNet, self).__init__()

        # Hyperparameters
        vocab_size = Config.VOCAB_SIZE
        embed_dim = Config.EMBED_DIM
        num_cont = Config.NUM_CONT_FEATURES
        seq_len = Config.TOKEN_SEQ_LEN
        hidden_dim = Config.HIDDEN_DIM
        num_blocks = Config.NUM_BLOCKS
        dropout_rate = Config.DROPOUT

        # 1. Input Processing
        # Shared embedding for the decomposed characters
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Calculate input dimension for projection
        # Continuous features + Flattened embeddings (10 tokens * 32 dim)
        input_proj_dim = num_cont + (seq_len * embed_dim)

        self.input_proj = nn.Linear(input_proj_dim, hidden_dim)

        # 2. Residual Gated Backbone
        self.blocks = nn.ModuleList(
            [ResidualGatedBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )

        # 3. Output Head
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_cat, x_cont):
        """
        Args:
            x_cat: (Batch, 10) LongTensor - Integer encoded characters
            x_cont: (Batch, 30) FloatTensor - Normalized continuous features
        """
        # Embed categorical tokens: (B, 10) -> (B, 10, 32)
        emb = self.embedding(x_cat)
        # Flatten embeddings: (B, 320)
        emb_flat = emb.view(emb.size(0), -1)

        # Concatenate with continuous features: (B, 350)
        x = torch.cat([emb_flat, x_cont], dim=1)

        # Project to hidden dimension
        x = self.input_proj(x)

        # Pass through residual blocks
        for block in self.blocks:
            x = block(x)

        # Final processing
        x = self.final_norm(x)
        logits = self.head(x)
        probs = self.sigmoid(logits)

        return probs


def train_model():
    """
    Trains the ResGLUNet model using the configuration specified in library.config.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Data
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # Initialize Model
    model = ResGLUNet().to(device)

    # Optimizer & Loss
    # High weight decay as per Idea description
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCELoss()

    # Training State
    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x_cat = batch["cat"].to(device)
            x_cont = batch["cont"].to(device)
            y = batch["target"].to(device)

            optimizer.zero_grad()
            preds = model(x_cat, x_cont)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x_cat = batch["cat"].to(device)
                x_cont = batch["cont"].to(device)
                y = batch["target"].to(device)

                preds = model(x_cat, x_cont)
                loss = criterion(preds, y)

                val_loss += loss.item()
                val_preds.append(preds.cpu().numpy())
                val_targets.append(y.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)

        try:
            val_auc = roc_auc_score(val_targets, val_preds)
        except ValueError:
            val_auc = 0.5

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"Val AUC: {val_auc:.6f}"
        )

        # --- Early Stopping & Checkpointing ---
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with AUC: {best_auc:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    return best_auc


def predict():
    """
    Loads the best model and generates predictions for the test set.
    Saves the result to submission.csv.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Test Data
    _, _, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # Load Model
    model = ResGLUNet().to(device)
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")

    print(f"Loading model from {Config.MODEL_SAVE_PATH}")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    all_preds = []

    print("Starting inference...")
    with torch.no_grad():
        for batch in test_loader:
            x_cat = batch["cat"].to(device)
            x_cont = batch["cont"].to(device)

            preds = model(x_cat, x_cont)
            all_preds.append(preds.cpu().numpy())

    # Flatten predictions
    all_preds = np.concatenate(all_preds).flatten()

    # Create Submission
    # We load test_metadata to ensure IDs match the test_loader order
    test_meta = pd.read_csv(Config.TEST_METADATA)

    submission = pd.DataFrame({"id": test_meta["id"], "target": all_preds})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
