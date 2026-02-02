import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------


class Config:
    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_optim")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Hyperparameters
    SEED = 42
    BATCH_SIZE = 1024
    EPOCHS = 40
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Architecture
    EMBED_DIM = 32
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_HEADS = 4
    BACKBONE_STAGES = [512, 256, 128]
    DROPOUT_TRANSFORMER = 0.1
    DROPOUT_BACKBONE = 0.35

    # Hardware
    NUM_WORKERS = 4


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------------------
# Data Processing
# ------------------------------------------------------------------------------


def process_data(load_cached_data=True):
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(Config.CACHE_DIR, "processed_data.npz")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            # Validate cache schema to handle stale or incompatible data (Cite debug_lesson_5)
            required_keys = [
                "X_num_train",
                "X_cat_train",
                "y_train",
                "X_num_val",
                "X_cat_val",
                "y_val",
                "X_num_test",
                "X_cat_test",
                "test_ids",
                "vocab_size",
            ]
            if all(k in data for k in required_keys):
                return (
                    data["X_num_train"],
                    data["X_cat_train"],
                    data["y_train"],
                    data["X_num_val"],
                    data["X_cat_val"],
                    data["y_val"],
                    data["X_num_test"],
                    data["X_cat_test"],
                    data["test_ids"],
                    int(data["vocab_size"]),
                )
            print("Cache schema mismatch. Reprocessing...")
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print("Processing data from scratch...")

    # Load Metadata
    train_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "train_metadata.csv"))
    val_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "val_metadata.csv"))
    test_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "test_metadata.csv"))

    # Load Raw Data
    df_train_full = pd.read_csv(os.path.join(Config.INPUT_DIR, "train.csv"))
    df_test_full = pd.read_csv(os.path.join(Config.INPUT_DIR, "test.csv"))

    # Indexing for fast lookup
    df_train_full.set_index("id", inplace=True)
    df_test_full.set_index("id", inplace=True)

    # Split based on metadata
    df_train = df_train_full.loc[train_meta["id"]].copy()
    y_train = train_meta["target"].values.astype(np.float32)

    df_val = df_train_full.loc[val_meta["id"]].copy()
    y_val = val_meta["target"].values.astype(np.float32)

    df_test = df_test_full.loc[test_meta["id"]].copy()
    test_ids = test_meta["id"].values

    # --- Feature Engineering ---

    # 1. Categorical: f_27
    # Build vocabulary from training data
    all_chars = sorted(list(set("".join(df_train["f_27"].unique()))))
    char_map = {
        c: i + 1 for i, c in enumerate(all_chars)
    }  # 1-based index, 0 is padding/unknown
    vocab_size = len(char_map) + 1

    def encode_f27(series):
        # Convert series of strings to numpy array of shape (N, 10)
        # Using list comprehension for simplicity; vectorized approaches exist but this is safe
        return np.array(
            [[char_map.get(c, 0) for c in s] for s in series], dtype=np.int64
        )

    X_cat_train = encode_f27(df_train["f_27"])
    X_cat_val = encode_f27(df_val["f_27"])
    X_cat_test = encode_f27(df_test["f_27"])

    # 2. Numerical: f_00 to f_30 (excluding f_27)
    num_cols = [c for c in df_train.columns if c != "f_27" and c != "target"]

    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(df_train[num_cols]).astype(np.float32)
    X_num_val = scaler.transform(df_val[num_cols]).astype(np.float32)
    X_num_test = scaler.transform(df_test[num_cols]).astype(np.float32)

    # Cache results
    np.savez(
        cache_path,
        X_num_train=X_num_train,
        X_cat_train=X_cat_train,
        y_train=y_train,
        X_num_val=X_num_val,
        X_cat_val=X_cat_val,
        y_val=y_val,
        X_num_test=X_num_test,
        X_cat_test=X_cat_test,
        test_ids=test_ids,
        vocab_size=vocab_size,
    )

    return (
        X_num_train,
        X_cat_train,
        y_train,
        X_num_val,
        X_cat_val,
        y_val,
        X_num_test,
        X_cat_test,
        test_ids,
        vocab_size,
    )


class ManufacturingDataset(Dataset):
    def __init__(self, X_num, X_cat, y=None):
        self.X_num = torch.from_numpy(X_num)
        self.X_cat = torch.from_numpy(X_cat)
        self.y = torch.from_numpy(y) if y is not None else None

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_num[idx], self.X_cat[idx], self.y[idx]
        return self.X_num[idx], self.X_cat[idx]


# ------------------------------------------------------------------------------
# Model Architecture
# ------------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.linear = nn.Linear(dim, dim * 2)
        self.glu = nn.GLU(dim=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Pre-Activation: BN -> Linear -> GLU -> Dropout -> Add
        out = self.bn(x)
        out = self.linear(out)
        out = self.glu(out)
        out = self.dropout(out)
        return x + out


class BalancedHybridNet(nn.Module):
    def __init__(
        self,
        num_features,
        cat_seq_len,
        vocab_size,
        embed_dim,
        transformer_layers,
        transformer_heads,
        backbone_stages,
        dropout_transformer,
        dropout_backbone,
    ):
        super().__init__()

        # Stream 1: Categorical Sequence
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Learnable Positional Embeddings initialized with random noise
        self.pos_encoder = nn.Parameter(torch.randn(1, cat_seq_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout_transformer,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        self.flat_dim = cat_seq_len * embed_dim

        # Fusion Stem
        fusion_input_dim = self.flat_dim + num_features
        self.stem = nn.Linear(fusion_input_dim, backbone_stages[0])

        # Backbone
        layers = []
        in_dim = backbone_stages[0]

        for out_dim in backbone_stages:
            # Transition layer if dimension changes
            if in_dim != out_dim:
                layers.append(nn.Linear(in_dim, out_dim))
                in_dim = out_dim

            # 2 Residual Blocks per stage
            layers.append(ResidualBlock(in_dim, dropout_backbone))
            layers.append(ResidualBlock(in_dim, dropout_backbone))

        self.backbone = nn.Sequential(*layers)

        # Output Head
        self.head = nn.Linear(backbone_stages[-1], 1)

    def forward(self, x_num, x_cat):
        # Stream 1
        emb = self.embedding(x_cat)  # (B, 10, 32)
        emb = emb + self.pos_encoder
        trans_out = self.transformer(emb)
        flat_cat = trans_out.reshape(trans_out.size(0), -1)

        # Fusion
        concat = torch.cat([flat_cat, x_num], dim=1)
        x = self.stem(concat)

        # Backbone
        x = self.backbone(x)

        # Head
        logits = self.head(x)
        return logits


# ------------------------------------------------------------------------------
# Training & Inference
# ------------------------------------------------------------------------------


def train_model():
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loading
    data = process_data(load_cached_data=True)
    X_num_train, X_cat_train, y_train = data[0], data[1], data[2]
    X_num_val, X_cat_val, y_val = data[3], data[4], data[5]
    X_num_test, X_cat_test, test_ids = data[6], data[7], data[8]
    vocab_size = data[9]

    train_dataset = ManufacturingDataset(X_num_train, X_cat_train, y_train)
    val_dataset = ManufacturingDataset(X_num_val, X_cat_val, y_val)
    test_dataset = ManufacturingDataset(X_num_test, X_cat_test)

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

    # Model Initialization
    model = BalancedHybridNet(
        num_features=X_num_train.shape[1],
        cat_seq_len=X_cat_train.shape[1],
        vocab_size=vocab_size,
        embed_dim=Config.EMBED_DIM,
        transformer_layers=Config.TRANSFORMER_LAYERS,
        transformer_heads=Config.TRANSFORMER_HEADS,
        backbone_stages=Config.BACKBONE_STAGES,
        dropout_transformer=Config.DROPOUT_TRANSFORMER,
        dropout_backbone=Config.DROPOUT_BACKBONE,
    ).to(device)

    # Weight Initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # Note: Transformer weights are already initialized by PyTorch, we leave them.

    model.apply(init_weights)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0

        for x_num, x_cat, y in train_loader:
            x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x_num, x_cat).squeeze()
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for x_num, x_cat, y in val_loader:
                x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
                logits = model(x_num, x_cat).squeeze()
                preds = torch.sigmoid(logits)
                val_preds.append(preds.cpu().numpy())
                val_targets.append(y.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_auc = roc_auc_score(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss/len(train_loader):.6f} | Val AUC: {val_auc:.10f} | LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    print(f"Training complete. Best Validation AUC: {best_auc:.10f}")

    # Inference
    print("Generating submission...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH))
    model.eval()

    test_preds = []
    with torch.no_grad():
        for x_num, x_cat in test_loader:
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            logits = model(x_num, x_cat).squeeze()
            preds = torch.sigmoid(logits)
            test_preds.append(preds.cpu().numpy())

    test_preds = np.concatenate(test_preds)

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub = pd.DataFrame({"id": test_ids, "target": test_preds})
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


# Execute
if __name__ == "__main__":
    train_model()
