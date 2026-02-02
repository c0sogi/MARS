import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


# ==========================================
# Configuration
# ==========================================
class Config:
    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_14"
    OUTPUT_SUBMISSION = "./submission/submission.csv"

    # Data Files
    TRAIN_FILE = "train.csv"
    TEST_FILE = "test.csv"
    TRAIN_META = "train_metadata.csv"
    VAL_META = "val_metadata.csv"
    TEST_META = "test_metadata.csv"

    # Model Hyperparameters
    EMBED_DIM = 32
    SEQ_LEN = 10
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_HEADS = 4
    RESFUNNEL_STAGES = [512, 256, 128]
    DROPOUT = 0.3

    # Training Hyperparameters
    SEED = 42
    BATCH_SIZE = 1024
    EPOCHS = 35
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    SCHEDULER_STEP_SIZE = 10
    SCHEDULER_GAMMA = 0.1
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4


# ==========================================
# Utilities
# ==========================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# Model Architecture
# ==========================================
class ConformerBlock(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        # Sub-module 1: MHSA
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        # Sub-module 2: Convolution
        self.norm2 = nn.LayerNorm(embed_dim)
        # Conv1d: (B, C, L) -> (B, C, L)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.act = nn.SiLU()  # Swish activation
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, D)

        # MHSA Branch
        res = x
        x_norm = self.norm1(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        x = res + self.dropout1(x_attn)

        # Conv Branch
        res = x
        x_norm = self.norm2(x)
        # Transpose for Conv1d: (B, L, D) -> (B, D, L)
        x_conv = x_norm.transpose(1, 2)
        x_conv = self.conv(x_conv)
        x_conv = self.act(x_conv)
        # Transpose back: (B, D, L) -> (B, L, D)
        x_conv = x_conv.transpose(1, 2)
        x = res + self.dropout2(x_conv)

        return x


class GLUBlock(nn.Module):
    def __init__(self, dim, dropout=0.35):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)
        self.linear = nn.Linear(dim, dim * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, dim)
        res = x
        out = self.norm(x)
        out = self.linear(out)
        # GLU: Split into value (a) and gate (b)
        a, b = out.chunk(2, dim=-1)
        out = a * torch.sigmoid(b)
        out = self.dropout(out)
        return res + out


class HybridResFunnel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Stream 1: Categorical (Transformer)
        # Vocab size 27 (1-26 for A-Z, 0 for padding/unknown)
        self.embedding = nn.Embedding(27, config.EMBED_DIM)
        self.pos_encoder = nn.Parameter(
            torch.zeros(1, config.SEQ_LEN, config.EMBED_DIM)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.EMBED_DIM,
            nhead=config.TRANSFORMER_HEADS,
            dim_feedforward=config.EMBED_DIM * 4,
            dropout=config.DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.TRANSFORMER_LAYERS
        )

        # Stream 1 Output Dimension
        s1_dim = config.SEQ_LEN * config.EMBED_DIM

        # Stream 2: Continuous (Raw)
        # f_00 to f_30 excluding f_27 -> 30 features
        s2_dim = 30

        # Fusion
        fusion_dim = s1_dim + s2_dim

        # Backbone: ResFunnel
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        # Projection to first stage width
        self.input_proj = nn.Sequential(
            nn.Linear(fusion_dim, config.RESFUNNEL_STAGES[0]),
            nn.BatchNorm1d(config.RESFUNNEL_STAGES[0]),
            nn.SiLU(),
        )

        for i, width in enumerate(config.RESFUNNEL_STAGES):
            # Stack Residual Gated Blocks (2 per stage)
            stage_blocks = nn.Sequential(
                GLUBlock(width, config.DROPOUT), GLUBlock(width, config.DROPOUT)
            )
            self.stages.append(stage_blocks)

            # Downsample logic
            if i < len(config.RESFUNNEL_STAGES) - 1:
                next_width = config.RESFUNNEL_STAGES[i + 1]
                ds = nn.Sequential(
                    nn.Linear(width, next_width), nn.BatchNorm1d(next_width), nn.SiLU()
                )
                self.downsamples.append(ds)
            else:
                self.downsamples.append(nn.Identity())

        # Output Head
        self.head = nn.Linear(config.RESFUNNEL_STAGES[-1], 1)

    def forward(self, x_cat, x_cont):
        # Stream 1 Processing
        x_c = self.embedding(x_cat)  # (B, L, D)
        x_c = x_c + self.pos_encoder
        x_c = self.transformer_encoder(x_c)
        x_c = x_c.flatten(1)  # (B, L*D)

        # Fusion
        x = torch.cat([x_c, x_cont], dim=1)

        # Backbone Processing
        x = self.input_proj(x)

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:
                x = self.downsamples[i](x)

        # Head
        logits = self.head(x)
        return logits


# ==========================================
# Data Processing
# ==========================================
def process_f27(series):
    """Maps strings of length 10 (A-Z) to integers (1-26)."""
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    char_map = {c: i + 1 for i, c in enumerate(chars)}

    # Convert series of strings to list of lists of characters
    # Then map to integers
    # Optimized for speed using list comprehension over numpy overhead for strings
    arr = np.array([list(s) for s in series])

    # Map characters to integers
    out = np.zeros(arr.shape, dtype=np.int64)
    for c, idx in char_map.items():
        out[arr == c] = idx
    return out


def load_and_process_data(load_cached_data=True):
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, "processed_data.npz")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        data = np.load(cache_path)
        return (
            data["X_cat_train"],
            data["X_cont_train"],
            data["y_train"],
            data["X_cat_val"],
            data["X_cont_val"],
            data["y_val"],
            data["X_cat_test"],
            data["X_cont_test"],
            data["test_ids"],
        )

    print("Processing data from scratch...")

    # Load Metadata
    train_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, Config.TRAIN_META))
    val_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, Config.VAL_META))
    test_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, Config.TEST_META))

    # Load Raw Data
    df_train_full = pd.read_csv(os.path.join(Config.INPUT_DIR, Config.TRAIN_FILE))
    df_test_full = pd.read_csv(os.path.join(Config.INPUT_DIR, Config.TEST_FILE))

    # Index by ID for fast lookup
    df_train_full.set_index("id", inplace=True)
    df_test_full.set_index("id", inplace=True)

    # Select subsets based on metadata
    train_df = df_train_full.loc[train_meta["id"]]
    val_df = df_train_full.loc[val_meta["id"]]
    test_df = df_test_full.loc[test_meta["id"]]

    # Targets
    y_train = train_df["target"].values.astype(np.float32)
    y_val = val_df["target"].values.astype(np.float32)

    # Feature Selection
    cat_col = "f_27"
    drop_cols = ["target"]
    cont_cols = [c for c in train_df.columns if c not in drop_cols and c != cat_col]

    # Process Categorical
    print("Encoding categorical features...")
    X_cat_train = process_f27(train_df[cat_col].values)
    X_cat_val = process_f27(val_df[cat_col].values)
    X_cat_test = process_f27(test_df[cat_col].values)

    # Process Continuous
    print("Normalizing continuous features...")
    scaler = StandardScaler()
    X_cont_train = scaler.fit_transform(train_df[cont_cols].values).astype(np.float32)
    X_cont_val = scaler.transform(val_df[cont_cols].values).astype(np.float32)
    X_cont_test = scaler.transform(test_df[cont_cols].values).astype(np.float32)

    test_ids = test_meta["id"].values

    # Cache
    print(f"Saving cache to {cache_path}")
    np.savez(
        cache_path,
        X_cat_train=X_cat_train,
        X_cont_train=X_cont_train,
        y_train=y_train,
        X_cat_val=X_cat_val,
        X_cont_val=X_cont_val,
        y_val=y_val,
        X_cat_test=X_cat_test,
        X_cont_test=X_cont_test,
        test_ids=test_ids,
    )

    return (
        X_cat_train,
        X_cont_train,
        y_train,
        X_cat_val,
        X_cont_val,
        y_val,
        X_cat_test,
        X_cont_test,
        test_ids,
    )


class TPSDataset(Dataset):
    def __init__(self, x_cat, x_cont, y=None):
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.x_cont = torch.tensor(x_cont, dtype=torch.float32)
        self.y = (
            torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y is not None else None
        )

    def __len__(self):
        return len(self.x_cat)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x_cat[idx], self.x_cont[idx], self.y[idx]
        return self.x_cat[idx], self.x_cont[idx]


# ==========================================
# Main Execution
# ==========================================
def main():
    set_seed(Config.SEED)

    # Load Data
    data = load_and_process_data(load_cached_data=True)
    (
        X_cat_train,
        X_cont_train,
        y_train,
        X_cat_val,
        X_cont_val,
        y_val,
        X_cat_test,
        X_cont_test,
        test_ids,
    ) = data

    # Create Datasets & Loaders
    train_dataset = TPSDataset(X_cat_train, X_cont_train, y_train)
    val_dataset = TPSDataset(X_cat_val, X_cont_val, y_val)
    test_dataset = TPSDataset(X_cat_test, X_cont_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
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

    # Initialize Model
    model = HybridResFunnel(Config).to(Config.DEVICE)

    # Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {Config.DEVICE}...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0

        for x_cat, x_cont, y in train_loader:
            x_cat, x_cont, y = (
                x_cat.to(Config.DEVICE),
                x_cont.to(Config.DEVICE),
                y.to(Config.DEVICE),
            )

            optimizer.zero_grad()
            logits = model(x_cat, x_cont)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for x_cat, x_cont, y in val_loader:
                x_cat, x_cont = x_cat.to(Config.DEVICE), x_cont.to(Config.DEVICE)
                logits = model(x_cat, x_cont)
                probs = torch.sigmoid(logits)
                val_preds.append(probs.cpu().numpy())
                val_targets.append(y.numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_auc = roc_auc_score(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val AUC: {val_auc:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

        scheduler.step()

    print(f"Training complete. Best Val AUC: {best_auc:.6f}")

    # Inference
    print("Generating submission...")
    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
    model.eval()

    test_preds = []
    with torch.no_grad():
        for x_cat, x_cont in test_loader:
            x_cat, x_cont = x_cat.to(Config.DEVICE), x_cont.to(Config.DEVICE)
            logits = model(x_cat, x_cont)
            probs = torch.sigmoid(logits)
            test_preds.append(probs.cpu().numpy())

    test_preds = np.concatenate(test_preds).flatten()

    # Save Submission
    os.makedirs(os.path.dirname(Config.OUTPUT_SUBMISSION), exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, "target": test_preds})

    submission.to_csv(Config.OUTPUT_SUBMISSION, index=False)
    print(f"Submission saved to {Config.OUTPUT_SUBMISSION}")
