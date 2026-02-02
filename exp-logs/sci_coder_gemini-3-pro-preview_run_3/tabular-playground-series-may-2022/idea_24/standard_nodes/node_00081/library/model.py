import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.data_processing import ManufacturingDataset, preprocess_features


# ==========================================
# Reproducibility
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================================
# Model Definition
# ==========================================
class FunnelStream(nn.Module):
    """
    A single stream of the Parallel Funnel Ensemble.
    Contains independent embeddings and a specific MLP backbone.
    """

    def __init__(self, vocab_sizes, cont_dim, embed_dim, hidden_layers, dropout_rate):
        super(FunnelStream, self).__init__()

        # Independent Embeddings for this stream
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=size, embedding_dim=embed_dim)
                for size in vocab_sizes
            ]
        )

        # Calculate input dimension for MLP
        # Continuous features + (Number of categorical features * Embedding dim)
        input_dim = cont_dim + (len(vocab_sizes) * embed_dim)

        # Construct MLP Backbone
        layers = []
        in_features = input_dim

        for width in hidden_layers:
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = width

        self.backbone = nn.Sequential(*layers)

        # Final Output Layer (1 logit)
        self.head = nn.Linear(in_features, 1)

    def forward(self, x_cont, x_cat):
        # x_cont: (batch, cont_dim)
        # x_cat: (batch, num_cat_features)

        # Process Embeddings
        embedded = []
        for i, emb_layer in enumerate(self.embeddings):
            # x_cat[:, i] is the column of indices for the i-th categorical feature
            embedded.append(emb_layer(x_cat[:, i]))

        # Concatenate embeddings: (batch, num_cat * embed_dim)
        x_emb = torch.cat(embedded, dim=1)

        # Early Fusion: Concatenate continuous and embeddings
        x = torch.cat([x_cont, x_emb], dim=1)

        # Pass through backbone
        x = self.backbone(x)

        # Output logit
        return self.head(x)


class ParallelFunnelEnsemble(nn.Module):
    """
    Safe-Spectrum Parallel Funnel Ensemble (SSPFE).
    Consists of 5 independent FunnelStreams.
    """

    def __init__(self, vocab_sizes, cont_dim, embed_dim, stream_configs):
        super(ParallelFunnelEnsemble, self).__init__()

        self.streams = nn.ModuleList()

        for config in stream_configs:
            stream = FunnelStream(
                vocab_sizes=vocab_sizes,
                cont_dim=cont_dim,
                embed_dim=embed_dim,
                hidden_layers=config["layers"],
                dropout_rate=config["dropout"],
            )
            self.streams.append(stream)

    def forward(self, x_cont, x_cat):
        # Forward pass through all streams
        # Returns tensor of shape (batch_size, num_streams)
        logits = [stream(x_cont, x_cat) for stream in self.streams]
        return torch.cat(logits, dim=1)


# ==========================================
# Training Function
# ==========================================
def train_model():
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load and Process Data
    train_df, val_df, test_df, vocab_sizes, cat_cols, cont_cols = preprocess_features(
        load_cached_data=True, config=Config
    )

    # 2. Create Datasets and Loaders
    train_dataset = ManufacturingDataset(train_df, cat_cols, cont_cols, mode="train")
    val_dataset = ManufacturingDataset(val_df, cat_cols, cont_cols, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Initialize Model
    model = ParallelFunnelEnsemble(
        vocab_sizes=vocab_sizes,
        cont_dim=len(cont_cols),
        embed_dim=Config.EMBED_DIM,
        stream_configs=Config.MODEL_STREAMS,
    ).to(device)

    # 4. Setup Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.MAX_LR, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0.0

        for x_cont, x_cat, y in train_loader:
            x_cont, x_cat, y = x_cont.to(device), x_cat.to(device), y.to(device)

            optimizer.zero_grad()

            # Forward pass: (batch, 5)
            logits = model(x_cont, x_cat)

            # Compute sum of BCE losses for each stream
            loss = 0
            for i in range(logits.shape[1]):
                loss += criterion(logits[:, i], y)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for x_cont, x_cat, y in val_loader:
                x_cont, x_cat, y = x_cont.to(device), x_cat.to(device), y.to(device)

                logits = model(x_cont, x_cat)

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(logits)

                # Ensemble prediction: Mean of probabilities
                mean_probs = torch.mean(probs, dim=1)

                val_preds.append(mean_probs.cpu().numpy())
                val_targets.append(y.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)

        val_auc = roc_auc_score(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val AUC: {val_auc:.10f}"
        )

        # Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print("New best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Val AUC: {best_auc:.10f}")


# ==========================================
# Prediction Function
# ==========================================
def predict_and_submit():
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load processed data (Test set)
    _, _, test_df, vocab_sizes, cat_cols, cont_cols = preprocess_features(
        load_cached_data=True, config=Config
    )

    test_dataset = ManufacturingDataset(test_df, cat_cols, cont_cols, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize Model
    model = ParallelFunnelEnsemble(
        vocab_sizes=vocab_sizes,
        cont_dim=len(cont_cols),
        embed_dim=Config.EMBED_DIM,
        stream_configs=Config.MODEL_STREAMS,
    ).to(device)

    # Load Best Weights
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    print("Generating predictions...")
    all_preds = []

    with torch.no_grad():
        for x_cont, x_cat, _ in test_loader:
            x_cont, x_cat = x_cont.to(device), x_cat.to(device)

            logits = model(x_cont, x_cat)
            probs = torch.sigmoid(logits)

            # Ensemble prediction: Mean of probabilities
            mean_probs = torch.mean(probs, dim=1)
            all_preds.append(mean_probs.cpu().numpy())

    final_preds = np.concatenate(all_preds)

    # Create Submission File
    submission = pd.DataFrame({"id": test_df["id"], "target": final_preds})

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


# ==========================================
# Main Execution
# ==========================================
# This block is provided to allow external execution if imported,
# but the prompt asks for the module content.
# The functions train_model() and predict_and_submit() are the entry points.
