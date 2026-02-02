import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import roc_auc_score

# Suppress warnings
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================


class Config:
    # Paths
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    CACHE_DIR = "./working/idea_37"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Architecture
    N_STREAMS = 5
    EMBED_DIM = 16

    # Stream Configurations (Hidden Layers, Dropout)
    # Streams 1 & 2 (Anchors), 3 & 4 (Capacity Variants), 5 (Conservative)
    STREAM_CONFIGS = [
        {"layers": [512, 256, 128], "dropout": 0.20},
        {"layers": [512, 256, 128], "dropout": 0.20},
        {"layers": [1024, 512, 256], "dropout": 0.25},
        {"layers": [1024, 512, 256], "dropout": 0.25},
        {"layers": [512, 256, 128], "dropout": 0.30},
    ]

    # Training Hyperparameters
    SEED = 42
    EPOCHS = 50
    BATCH_SIZE = 1024
    LEARNING_RATE = 1e-2  # max_lr for OneCycle
    WEIGHT_DECAY = 2e-5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        cls.set_seed()

    @classmethod
    def set_seed(cls):
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True


# ==========================================
# DATA PROCESSING
# ==========================================


def get_unique_char_count(s):
    return len(set(s))


class TabularDataset(Dataset):
    def __init__(self, x_cont, x_cat, y=None):
        self.x_cont = torch.tensor(x_cont, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.y = (
            torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y is not None else None
        )

    def __len__(self):
        return len(self.x_cont)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x_cont[idx], self.x_cat[idx], self.y[idx]
        return self.x_cont[idx], self.x_cat[idx]


def get_data(load_cached_data=True, debug=False):
    cache_file = os.path.join(Config.CACHE_DIR, "processed_data.npy")

    # Try loading from cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            print(f"Loading cached data from {cache_file}...")
            data = np.load(cache_file, allow_pickle=True).item()

            # Reconstruct Datasets
            train_ds = TabularDataset(
                data["X_train_cont"], data["X_train_cat"], data["y_train"]
            )
            val_ds = TabularDataset(
                data["X_val_cont"], data["X_val_cat"], data["y_val"]
            )
            test_ds = TabularDataset(data["X_test_cont"], data["X_test_cat"], None)

            loaders = {
                "train": DataLoader(
                    train_ds,
                    batch_size=Config.BATCH_SIZE,
                    shuffle=True,
                    num_workers=Config.NUM_WORKERS,
                    pin_memory=True,
                ),
                "val": DataLoader(
                    val_ds,
                    batch_size=Config.BATCH_SIZE,
                    shuffle=False,
                    num_workers=Config.NUM_WORKERS,
                    pin_memory=True,
                ),
                "test": DataLoader(
                    test_ds,
                    batch_size=Config.BATCH_SIZE,
                    shuffle=False,
                    num_workers=Config.NUM_WORKERS,
                    pin_memory=True,
                ),
            }
            return loaders, data["dims"], data["ids"]
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print("Processing data from scratch...")

    # Load Data
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    if debug:
        train_df = train_df.head(5000)
        val_df = val_df.head(1000)
        test_df = test_df.head(1000)

    test_ids = test_df["id"].values

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    full_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)

    # Feature Engineering
    # 1. Set Cardinality
    full_df["f_27_unique_count"] = full_df["f_27"].apply(get_unique_char_count)

    # 2. String Decomposition
    for i in range(10):
        full_df[f"f_27_char_{i}"] = full_df["f_27"].str[i]

    # Identify Columns
    # Continuous: f_00 to f_28 (excluding f_27) + f_27_unique_count
    cont_cols = [f"f_{i:02d}" for i in range(29) if i != 27]
    cont_cols.append("f_27_unique_count")

    # Categorical: f_29, f_30 + f_27_char_0...9
    cat_cols = ["f_29", "f_30"] + [f"f_27_char_{i}" for i in range(10)]

    # Scaling (Fit on Train, Transform All)
    scaler = StandardScaler()
    train_mask = full_df["split"] == "train"
    scaler.fit(full_df.loc[train_mask, cont_cols])
    full_df.loc[:, cont_cols] = scaler.transform(full_df.loc[:, cont_cols])

    # Encoding (Transductive: Fit on All)
    oe = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64
    )
    full_df.loc[:, cat_cols] = oe.fit_transform(full_df.loc[:, cat_cols])

    for col in cat_cols:
        full_df[col] = full_df[col].astype(int)

    vocab_sizes = [int(full_df[col].max() + 1) for col in cat_cols]

    # Split
    train_proc = full_df[full_df["split"] == "train"]
    val_proc = full_df[full_df["split"] == "val"]
    test_proc = full_df[full_df["split"] == "test"]

    y_train = train_proc["target"].values.astype(np.float32)
    y_val = val_proc["target"].values.astype(np.float32)

    X_train_cont = train_proc[cont_cols].values.astype(np.float32)
    X_train_cat = train_proc[cat_cols].values.astype(np.int64)

    X_val_cont = val_proc[cont_cols].values.astype(np.float32)
    X_val_cat = val_proc[cat_cols].values.astype(np.int64)

    X_test_cont = test_proc[cont_cols].values.astype(np.float32)
    X_test_cat = test_proc[cat_cols].values.astype(np.int64)

    dims = {
        "n_cont": len(cont_cols),
        "n_cat": len(cat_cols),
        "vocab_sizes": vocab_sizes,
    }

    # Save to cache
    cache_data = {
        "X_train_cont": X_train_cont,
        "X_train_cat": X_train_cat,
        "y_train": y_train,
        "X_val_cont": X_val_cont,
        "X_val_cat": X_val_cat,
        "y_val": y_val,
        "X_test_cont": X_test_cont,
        "X_test_cat": X_test_cat,
        "dims": dims,
        "ids": test_ids,
    }
    np.save(cache_file, cache_data)

    # Create Loaders
    train_ds = TabularDataset(X_train_cont, X_train_cat, y_train)
    val_ds = TabularDataset(X_val_cont, X_val_cat, y_val)
    test_ds = TabularDataset(X_test_cont, X_test_cat, None)

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        ),
    }

    return loaders, dims, test_ids


# ==========================================
# MODEL
# ==========================================


class Stream(nn.Module):
    def __init__(self, n_cont, vocab_sizes, embed_dim, hidden_layers, dropout):
        super().__init__()

        # Independent Embeddings
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=v, embedding_dim=embed_dim)
                for v in vocab_sizes
            ]
        )

        n_cat_flat = len(vocab_sizes) * embed_dim
        input_dim = n_cont + n_cat_flat

        # Deep Path (Funnel)
        layers = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.deep_path = nn.Sequential(*layers)

        # Wide Path (Selective Residual)
        # Input: Only Continuous features (n_cont)
        self.wide_path = nn.Linear(n_cont, 1)

    def forward(self, x_cont, x_cat):
        # Embeddings
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(embs, dim=1)

        # Concat for Deep Path
        x_deep_in = torch.cat([x_cont, x_emb], dim=1)

        # Paths
        out_deep = self.deep_path(x_deep_in)
        out_wide = self.wide_path(x_cont)

        return out_deep + out_wide


class DARPEModel(nn.Module):
    def __init__(self, n_cont, vocab_sizes):
        super().__init__()
        self.streams = nn.ModuleList()

        for cfg in Config.STREAM_CONFIGS:
            self.streams.append(
                Stream(
                    n_cont=n_cont,
                    vocab_sizes=vocab_sizes,
                    embed_dim=Config.EMBED_DIM,
                    hidden_layers=cfg["layers"],
                    dropout=cfg["dropout"],
                )
            )

    def forward(self, x_cont, x_cat):
        outputs = []
        for stream in self.streams:
            outputs.append(stream(x_cont, x_cat))
        return outputs


# ==========================================
# TRAINING & INFERENCE
# ==========================================


def train_model(loaders, dims):
    model = DARPEModel(n_cont=dims["n_cont"], vocab_sizes=dims["vocab_sizes"])
    model.to(Config.DEVICE)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    steps_per_epoch = len(loaders["train"])
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    print(f"Starting training on {Config.DEVICE} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0

        for x_cont, x_cat, y in loaders["train"]:
            x_cont, x_cat, y = (
                x_cont.to(Config.DEVICE),
                x_cat.to(Config.DEVICE),
                y.to(Config.DEVICE),
            )

            optimizer.zero_grad()
            outputs = model(x_cont, x_cat)

            # Sum of losses for each stream
            loss = 0
            for out in outputs:
                loss += criterion(out, y)

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(loaders["train"])

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for x_cont, x_cat, y in loaders["val"]:
                x_cont, x_cat, y = (
                    x_cont.to(Config.DEVICE),
                    x_cat.to(Config.DEVICE),
                    y.to(Config.DEVICE),
                )
                outputs = model(x_cont, x_cat)

                # Average predictions (sigmoid applied)
                probs = [torch.sigmoid(out) for out in outputs]
                avg_prob = torch.mean(torch.stack(probs), dim=0)

                val_preds.append(avg_prob.cpu().numpy())
                val_targets.append(y.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_auc = roc_auc_score(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Loss: {train_loss:.5f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best Val AUC: {best_auc:.6f}")
    return best_model_path


def generate_submission(model_path, loaders, dims, test_ids):
    print("Generating submission...")
    model = DARPEModel(n_cont=dims["n_cont"], vocab_sizes=dims["vocab_sizes"])
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()

    all_preds = []

    with torch.no_grad():
        for x_cont, x_cat in loaders["test"]:
            x_cont, x_cat = x_cont.to(Config.DEVICE), x_cat.to(Config.DEVICE)
            outputs = model(x_cont, x_cat)

            probs = [torch.sigmoid(out) for out in outputs]
            avg_prob = torch.mean(torch.stack(probs), dim=0)

            all_preds.append(avg_prob.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()

    sub_df = pd.DataFrame({"id": test_ids, "target": all_preds})

    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    Config.setup()
    loaders, dims, test_ids = get_data(load_cached_data=True)
    best_model_path = train_model(loaders, dims)
    generate_submission(best_model_path, loaders, dims, test_ids)


main()
