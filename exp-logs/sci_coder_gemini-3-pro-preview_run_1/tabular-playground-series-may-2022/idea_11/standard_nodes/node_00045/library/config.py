import os
import gc
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# ==================================================================================
# CONFIGURATION
# ==================================================================================


class Config:
    # Paths
    TRAIN_META = "./metadata/train.csv"
    VAL_META = "./metadata/val.csv"
    TEST_META = "./metadata/test.csv"
    WORK_DIR = "./working/idea_11/"
    SUBMISSION_DIR = "./submission/"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Data
    SEED = 42
    NUM_WORKERS = 4

    # Model Hyperparameters
    BATCH_SIZE = 1024
    EPOCHS = 40
    LR = 1e-3
    HIDDEN_DIM = 256
    NUM_LAYERS = 6
    NHEAD = 8
    DROPOUT = 0.1
    MASK_PROB = 0.15
    LABEL_SMOOTHING = 0.01
    RECON_LAMBDA = 1.0

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Ensure directories exist
os.makedirs(Config.WORK_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Set Seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(Config.SEED)

# ==================================================================================
# DATA PROCESSING
# ==================================================================================


def process_data(load_cached_data=True):
    """
    Loads data from metadata CSVs, performs feature engineering, scaling, and tokenization.
    Implements caching using .npy files in Config.WORK_DIR.
    """
    cache_files = {
        "X_num_train": os.path.join(Config.WORK_DIR, "X_num_train.npy"),
        "X_seq_train": os.path.join(Config.WORK_DIR, "X_seq_train.npy"),
        "y_train": os.path.join(Config.WORK_DIR, "y_train.npy"),
        "X_num_val": os.path.join(Config.WORK_DIR, "X_num_val.npy"),
        "X_seq_val": os.path.join(Config.WORK_DIR, "X_seq_val.npy"),
        "y_val": os.path.join(Config.WORK_DIR, "y_val.npy"),
        "X_num_test": os.path.join(Config.WORK_DIR, "X_num_test.npy"),
        "X_seq_test": os.path.join(Config.WORK_DIR, "X_seq_test.npy"),
        "ids_test": os.path.join(Config.WORK_DIR, "ids_test.npy"),
        "vocab_size": os.path.join(Config.WORK_DIR, "vocab_size.npy"),
    }

    # Check cache
    if load_cached_data and all(os.path.exists(p) for p in cache_files.values()):
        print("Loading cached data...")
        data = {k: np.load(v) for k, v in cache_files.items() if k != "vocab_size"}
        vocab_size = int(np.load(cache_files["vocab_size"]))
        return data, vocab_size

    print("Processing data from scratch...")

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_META)
    val_df = pd.read_csv(Config.VAL_META)
    test_df = pd.read_csv(Config.TEST_META)

    # Feature Engineering: unique_characters
    def add_features(df):
        df["unique_characters"] = df["f_27"].apply(lambda x: len(set(x)))
        return df

    train_df = add_features(train_df)
    val_df = add_features(val_df)
    test_df = add_features(test_df)

    # Identify Columns
    # f_27 is sequence, target is target, id is id. Everything else is numeric.
    exclude = ["id", "target", "f_27", "source_path"]
    num_cols = [c for c in train_df.columns if c not in exclude]

    # Numerical Processing
    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(train_df[num_cols].values.astype(np.float32))
    X_num_val = scaler.transform(val_df[num_cols].values.astype(np.float32))
    X_num_test = scaler.transform(test_df[num_cols].values.astype(np.float32))

    # Sequence Processing (f_27)
    # Build Vocab
    all_chars = set()
    for s in train_df["f_27"]:
        all_chars.update(s)

    vocab = sorted(list(all_chars))
    char_to_idx = {c: i + 1 for i, c in enumerate(vocab)}  # 0 is padding/mask
    vocab_size = len(vocab) + 1

    def tokenize(series, max_len=10):
        # f_27 is fixed length 10 usually
        seqs = []
        for s in series:
            seq = [char_to_idx.get(c, 0) for c in s]
            if len(seq) < max_len:
                seq += [0] * (max_len - len(seq))
            else:
                seq = seq[:max_len]
            seqs.append(seq)
        return np.array(seqs, dtype=np.int64)

    X_seq_train = tokenize(train_df["f_27"])
    X_seq_val = tokenize(val_df["f_27"])
    X_seq_test = tokenize(test_df["f_27"])

    # Targets & IDs
    y_train = train_df["target"].values.astype(np.float32)
    y_val = val_df["target"].values.astype(np.float32)
    ids_test = test_df["id"].values

    # Save to Cache
    np.save(cache_files["X_num_train"], X_num_train)
    np.save(cache_files["X_seq_train"], X_seq_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_num_val"], X_num_val)
    np.save(cache_files["X_seq_val"], X_seq_val)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["X_num_test"], X_num_test)
    np.save(cache_files["X_seq_test"], X_seq_test)
    np.save(cache_files["ids_test"], ids_test)
    np.save(cache_files["vocab_size"], np.array([vocab_size]))

    data = {
        "X_num_train": X_num_train,
        "X_seq_train": X_seq_train,
        "y_train": y_train,
        "X_num_val": X_num_val,
        "X_seq_val": X_seq_val,
        "y_val": y_val,
        "X_num_test": X_num_test,
        "X_seq_test": X_seq_test,
        "ids_test": ids_test,
    }

    return data, vocab_size


# ==================================================================================
# DATASET
# ==================================================================================


class ResDeGUTDataset(Dataset):
    def __init__(self, X_num, X_seq, y=None, mask_prob=0.0, vocab_size=0):
        self.X_num = X_num
        self.X_seq = X_seq
        self.y = y
        self.mask_prob = mask_prob
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        x_n = self.X_num[idx]
        x_s = self.X_seq[idx]

        # Masking for Transformer Branch
        mask_s = np.zeros_like(x_s, dtype=np.bool_)
        target_s = x_s.copy()  # Target for reconstruction

        if self.mask_prob > 0:
            # Mask sequence
            mask_indices = np.random.rand(len(x_s)) < self.mask_prob
            x_s_masked = x_s.copy()
            x_s_masked[mask_indices] = 0  # 0 is mask token
            mask_s[mask_indices] = True
            x_s_out = x_s_masked
        else:
            x_s_out = x_s

        item = {
            "x_num": torch.tensor(x_n, dtype=torch.float32),
            "x_seq": torch.tensor(x_s_out, dtype=torch.long),
            "target_seq": torch.tensor(target_s, dtype=torch.long),
            "mask_seq": torch.tensor(mask_s, dtype=torch.bool),
        }

        if self.y is not None:
            item["target"] = torch.tensor(self.y[idx], dtype=torch.float32)

        return item


# ==================================================================================
# MODEL
# ==================================================================================


class ResDeGUT(nn.Module):
    def __init__(self, num_features, seq_len, vocab_size, config):
        super().__init__()
        self.config = config
        self.d_model = config.HIDDEN_DIM

        # Shared Embeddings
        self.num_tokenizer = nn.Linear(
            1, self.d_model
        )  # Tokenize each numerical feature
        self.seq_embedding = nn.Embedding(vocab_size, self.d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))

        # Positional Encoding for Transformer
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 1 + num_features + seq_len, self.d_model)
        )

        # Branch 1: Transformer (Deep)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.NHEAD,
            dim_feedforward=self.d_model * 4,
            dropout=config.DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.NUM_LAYERS
        )

        # Reconstruction Head (for sequence)
        self.seq_recon = nn.Linear(self.d_model, vocab_size)

        # Branch 2: Wide Residual (MLP)
        # Input: num_features (raw) + flattened sequence embeddings
        wide_dim = num_features + (seq_len * self.d_model)
        self.wide_mlp = nn.Sequential(
            nn.Linear(wide_dim, self.d_model),
            nn.BatchNorm1d(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )

        # Fusion Head
        self.head = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(self.d_model, 1),
        )

    def forward(self, x_num, x_seq):
        batch_size = x_num.size(0)

        # --- Shared Embeddings ---
        # Numerical: (B, N_num) -> (B, N_num, 1) -> (B, N_num, D)
        x_num_emb = self.num_tokenizer(x_num.unsqueeze(-1))

        # Sequence: (B, Seq_len) -> (B, Seq_len, D)
        x_seq_emb = self.seq_embedding(x_seq)

        # --- Branch 1: Transformer ---
        # Concat: [CLS] + Num + Seq
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_deep = torch.cat((cls_tokens, x_num_emb, x_seq_emb), dim=1)

        # Add Positional Encoding
        x_deep = x_deep + self.pos_embedding

        # Pass through Transformer
        x_deep_out = self.transformer(x_deep)

        # Extract CLS output
        cls_out = x_deep_out[:, 0, :]

        # Extract Sequence output for reconstruction (last seq_len tokens)
        seq_len = x_seq.size(1)
        seq_out = x_deep_out[:, -seq_len:, :]
        recon_logits = self.seq_recon(seq_out)

        # --- Branch 2: Wide Residual ---
        # Flatten sequence embeddings: (B, Seq_len * D)
        x_seq_flat = x_seq_emb.view(batch_size, -1)
        # Concat raw numericals + flattened seq embeddings
        x_wide_in = torch.cat((x_num, x_seq_flat), dim=1)

        wide_out = self.wide_mlp(x_wide_in)

        # --- Fusion ---
        fused = torch.cat((cls_out, wide_out), dim=1)
        logits = self.head(fused)

        return logits, recon_logits


# ==================================================================================
# TRAINING UTILS
# ==================================================================================


def train_one_epoch(
    model, loader, optimizer, scheduler, criterion_bce, criterion_ce, device, epoch
):
    model.train()
    total_loss = 0

    for batch in loader:
        x_num = batch["x_num"].to(device)
        x_seq = batch["x_seq"].to(device)
        target = batch["target"].to(device)
        target_seq = batch["target_seq"].to(device)
        mask_seq = batch["mask_seq"].to(device)

        optimizer.zero_grad()

        logits, recon_logits = model(x_num, x_seq)

        # Main Task Loss
        loss_bce = criterion_bce(logits.squeeze(), target)

        # Reconstruction Loss (Only on masked tokens)
        if mask_seq.sum() > 0:
            loss_recon = criterion_ce(recon_logits[mask_seq], target_seq[mask_seq])
        else:
            loss_recon = torch.tensor(0.0, device=device)

        loss = loss_bce + Config.RECON_LAMBDA * loss_recon

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion_bce, device):
    model.eval()
    total_loss = 0
    preds = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            x_num = batch["x_num"].to(device)
            x_seq = batch["x_seq"].to(device)
            target = batch["target"].to(device)

            logits, _ = model(x_num, x_seq)
            loss = criterion_bce(logits.squeeze(), target)

            total_loss += loss.item()
            preds.extend(torch.sigmoid(logits).squeeze().cpu().numpy())
            targets.extend(target.cpu().numpy())

    auc = roc_auc_score(targets, preds)
    return total_loss / len(loader), auc


def predict(model, loader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in loader:
            x_num = batch["x_num"].to(device)
            x_seq = batch["x_seq"].to(device)

            logits, _ = model(x_num, x_seq)
            preds.extend(torch.sigmoid(logits).squeeze().cpu().numpy())

    return preds


# ==================================================================================
# MAIN EXECUTION
# ==================================================================================


def main():
    print("Starting ResDeGUT Pipeline...")

    # 1. Load Data
    data, vocab_size = process_data(load_cached_data=True)

    # 2. Datasets & Loaders
    train_dataset = ResDeGUTDataset(
        data["X_num_train"],
        data["X_seq_train"],
        data["y_train"],
        mask_prob=Config.MASK_PROB,
        vocab_size=vocab_size,
    )
    val_dataset = ResDeGUTDataset(
        data["X_num_val"],
        data["X_seq_val"],
        data["y_val"],
        mask_prob=0.0,
        vocab_size=vocab_size,
    )
    test_dataset = ResDeGUTDataset(
        data["X_num_test"],
        data["X_seq_test"],
        None,
        mask_prob=0.0,
        vocab_size=vocab_size,
    )

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

    # 3. Model
    num_features = data["X_num_train"].shape[1]
    seq_len = data["X_seq_train"].shape[1]

    model = ResDeGUT(num_features, seq_len, vocab_size, Config).to(Config.DEVICE)

    # 4. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-2)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.LR,
        steps_per_epoch=len(train_loader),
        epochs=Config.EPOCHS,
        pct_start=0.1,
    )

    # Label Smoothing BCE
    class LabelSmoothingBCE(nn.Module):
        def __init__(self, smoothing=0.0):
            super().__init__()
            self.smoothing = smoothing
            self.bce = nn.BCEWithLogitsLoss()

        def forward(self, logits, targets):
            targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
            return self.bce(logits, targets_smooth)

    criterion_main = LabelSmoothingBCE(smoothing=Config.LABEL_SMOOTHING)
    criterion_val = nn.BCEWithLogitsLoss()
    criterion_recon = nn.CrossEntropyLoss()

    # 5. Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(Config.WORK_DIR, "best_model.pth")

    print(f"Training on {Config.DEVICE} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion_main,
            criterion_recon,
            Config.DEVICE,
            epoch,
        )
        val_loss, val_auc = validate(model, val_loader, criterion_val, Config.DEVICE)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"  New Best AUC! Model saved.")

    print(f"Training Complete. Best Val AUC: {best_auc:.6f}")

    # 6. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))

    predictions = predict(model, test_loader, Config.DEVICE)

    # 7. Submission
    print("Generating submission file...")
    sub_df = pd.DataFrame({"id": data["ids_test"], "target": predictions})

    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
