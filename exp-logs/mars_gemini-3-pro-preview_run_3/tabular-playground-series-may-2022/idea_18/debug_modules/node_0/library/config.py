import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import roc_auc_score

# ==========================================
# 1. Global Configuration & Constants
# ==========================================
SEED = 42
BATCH_SIZE = 1024
EMBEDDING_DIM = 16
HIDDEN_LAYERS = [512, 256, 128]
DROPOUT = 0.20
WEIGHT_DECAY = 1e-5
LEARNING_RATE = 1e-2  # Max LR for OneCycle
NUM_STREAMS = 5
EPOCHS = 30  # Default max epochs
PATIENCE = 5

TRAIN_PATH = "./metadata/train.csv"
VAL_PATH = "./metadata/val.csv"
TEST_PATH = "./metadata/test.csv"
CACHE_DIR = "./working/idea_18/"
SUBMISSION_PATH = "./submission/submission.csv"


# Ensure reproducibility
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed()

# ==========================================
# 2. Data Processing & Caching
# ==========================================


def process_data(load_cached_data=True):
    """
    Loads, preprocesses, and caches data.
    Implements transductive vocabulary alignment and feature engineering.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Cache file paths
    cache_files = {
        "X_train_cat": os.path.join(CACHE_DIR, "X_train_cat.npy"),
        "X_train_cont": os.path.join(CACHE_DIR, "X_train_cont.npy"),
        "y_train": os.path.join(CACHE_DIR, "y_train.npy"),
        "X_val_cat": os.path.join(CACHE_DIR, "X_val_cat.npy"),
        "X_val_cont": os.path.join(CACHE_DIR, "X_val_cont.npy"),
        "y_val": os.path.join(CACHE_DIR, "y_val.npy"),
        "X_test_cat": os.path.join(CACHE_DIR, "X_test_cat.npy"),
        "X_test_cont": os.path.join(CACHE_DIR, "X_test_cont.npy"),
        "test_ids": os.path.join(CACHE_DIR, "test_ids.npy"),
        "vocab_sizes": os.path.join(CACHE_DIR, "vocab_sizes.npy"),
    }

    # 1. Try to load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in cache_files.values())
        if all_exist:
            print("Loading data from cache...")
            data = {k: np.load(v, allow_pickle=True) for k, v in cache_files.items()}
            # Convert vocab_sizes back to list of ints
            data["vocab_sizes"] = data["vocab_sizes"].tolist()
            return data

    print("Processing data from scratch...")

    # Load raw data
    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)
    df_test = pd.read_csv(TEST_PATH)

    # Combine for transductive processing
    # Mark source to split later
    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    # Target handling
    y_train = df_train["target"].values.astype(np.float32)
    y_val = df_val["target"].values.astype(np.float32)

    # Drop target from combined df
    df_full = pd.concat(
        [
            df_train.drop(columns=["target", "source_path"]),
            df_val.drop(columns=["target", "source_path"]),
            df_test.drop(columns=["source_path"]),
        ],
        axis=0,
        ignore_index=True,
    )

    # Feature Engineering: f_27 decomposition
    # Split string into 10 characters
    for i in range(10):
        df_full[f"ch_{i}"] = df_full["f_27"].str[i]

    # Unique character count
    df_full["unique_char_count"] = df_full["f_27"].apply(lambda x: len(set(x)))

    # Drop original f_27 and id (keep test ids separately)
    test_ids = df_test["id"].values
    split_col = df_full["split"]
    df_full = df_full.drop(columns=["f_27", "id", "split"])

    # Identify columns
    # Categorical: f_29, f_30, and the 10 char columns
    cat_cols = ["f_29", "f_30"] + [f"ch_{i}" for i in range(10)]
    # Continuous: Everything else
    cont_cols = [c for c in df_full.columns if c not in cat_cols]

    # Encoding Categorical (Transductive)
    enc = OrdinalEncoder(dtype=np.int64)
    df_full[cat_cols] = enc.fit_transform(df_full[cat_cols])

    # Calculate vocab sizes (max index + 1 for embedding)
    vocab_sizes = [int(df_full[c].max() + 1) for c in cat_cols]

    # Scaling Continuous
    scaler = StandardScaler()
    df_full[cont_cols] = scaler.fit_transform(df_full[cont_cols])

    # Split back
    mask_train = split_col == "train"
    mask_val = split_col == "val"
    mask_test = split_col == "test"

    X_train_cat = df_full.loc[mask_train, cat_cols].values.astype(np.int64)
    X_train_cont = df_full.loc[mask_train, cont_cols].values.astype(np.float32)

    X_val_cat = df_full.loc[mask_val, cat_cols].values.astype(np.int64)
    X_val_cont = df_full.loc[mask_val, cont_cols].values.astype(np.float32)

    X_test_cat = df_full.loc[mask_test, cat_cols].values.astype(np.int64)
    X_test_cont = df_full.loc[mask_test, cont_cols].values.astype(np.float32)

    # Save to cache
    np.save(cache_files["X_train_cat"], X_train_cat)
    np.save(cache_files["X_train_cont"], X_train_cont)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_val_cat"], X_val_cat)
    np.save(cache_files["X_val_cont"], X_val_cont)
    np.save(cache_files["y_val"], y_val)
    np.save(cache_files["X_test_cat"], X_test_cat)
    np.save(cache_files["X_test_cont"], X_test_cont)
    np.save(cache_files["test_ids"], test_ids)
    np.save(cache_files["vocab_sizes"], np.array(vocab_sizes))

    return {
        "X_train_cat": X_train_cat,
        "X_train_cont": X_train_cont,
        "y_train": y_train,
        "X_val_cat": X_val_cat,
        "X_val_cont": X_val_cont,
        "y_val": y_val,
        "X_test_cat": X_test_cat,
        "X_test_cont": X_test_cont,
        "test_ids": test_ids,
        "vocab_sizes": vocab_sizes,
    }


class TabularDataset(Dataset):
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
# 3. Model Architecture (PIFE)
# ==========================================


class PIFEStream(nn.Module):
    def __init__(
        self, vocab_sizes, num_cont_features, embedding_dim, hidden_layers, dropout
    ):
        super().__init__()
        # Independent embeddings for this stream
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=vs, embedding_dim=embedding_dim)
                for vs in vocab_sizes
            ]
        )

        input_dim = num_cont_features + (len(vocab_sizes) * embedding_dim)

        layers = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        # Final projection to 1 output
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat, x_cont):
        # x_cat: [batch, num_cat_features]
        # x_cont: [batch, num_cont_features]

        embedded = []
        for i, emb_layer in enumerate(self.embeddings):
            embedded.append(emb_layer(x_cat[:, i]))

        x_emb = torch.cat(embedded, dim=1)
        x_in = torch.cat([x_emb, x_cont], dim=1)

        return self.mlp(x_in)


class PIFEModel(nn.Module):
    def __init__(
        self,
        vocab_sizes,
        num_cont_features,
        embedding_dim=EMBEDDING_DIM,
        hidden_layers=HIDDEN_LAYERS,
        dropout=DROPOUT,
        num_streams=NUM_STREAMS,
    ):
        super().__init__()
        self.streams = nn.ModuleList(
            [
                PIFEStream(
                    vocab_sizes,
                    num_cont_features,
                    embedding_dim,
                    hidden_layers,
                    dropout,
                )
                for _ in range(num_streams)
            ]
        )

    def forward(self, x_cat, x_cont):
        # Returns [batch, num_streams]
        outputs = []
        for stream in self.streams:
            outputs.append(stream(x_cat, x_cont))
        return torch.cat(outputs, dim=1)


# ==========================================
# 4. Training & Inference
# ==========================================


def train_model(load_cached_data=True, epochs=EPOCHS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    data = process_data(load_cached_data)

    train_dataset = TabularDataset(
        data["X_train_cat"], data["X_train_cont"], data["y_train"]
    )
    val_dataset = TabularDataset(data["X_val_cat"], data["X_val_cont"], data["y_val"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize Model
    num_cont = data["X_train_cont"].shape[1]
    vocab_sizes = data["vocab_sizes"]

    model = PIFEModel(vocab_sizes, num_cont).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # OneCycleLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=100.0,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x_cat, x_cont, y in train_loader:
            x_cat, x_cont, y = x_cat.to(device), x_cont.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x_cat, x_cont)  # [batch, 5]

            # Loss is sum of BCE for each stream
            loss = 0
            for i in range(NUM_STREAMS):
                loss += criterion(outputs[:, i : i + 1], y)

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for x_cat, x_cont, y in val_loader:
                x_cat, x_cont = x_cat.to(device), x_cont.to(device)
                outputs = model(x_cat, x_cont)

                # Average probabilities across streams
                probs = torch.sigmoid(outputs).mean(dim=1)
                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(y.numpy())

        val_auc = roc_auc_score(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.6f} | Val AUC: {val_auc:.10f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # Save best model state
            torch.save(model.state_dict(), os.path.join(CACHE_DIR, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Best Validation AUC: {best_val_auc:.10f}")
    return model, device, data


def generate_submission(model=None, device=None, data=None):
    if model is None or device is None or data is None:
        # If called independently, reload everything
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = process_data(load_cached_data=True)
        num_cont = data["X_test_cont"].shape[1]
        vocab_sizes = data["vocab_sizes"]
        model = PIFEModel(vocab_sizes, num_cont).to(device)
        model.load_state_dict(torch.load(os.path.join(CACHE_DIR, "best_model.pth")))
    else:
        # Load best weights
        model.load_state_dict(torch.load(os.path.join(CACHE_DIR, "best_model.pth")))

    test_dataset = TabularDataset(data["X_test_cat"], data["X_test_cont"])
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    model.eval()
    all_preds = []

    print("Generating predictions...")
    with torch.no_grad():
        for x_cat, x_cont in test_loader:
            x_cat, x_cont = x_cat.to(device), x_cont.to(device)
            outputs = model(x_cat, x_cont)

            # Average probabilities
            probs = torch.sigmoid(outputs).mean(dim=1)
            all_preds.extend(probs.cpu().numpy())

    # Create submission file
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission = pd.DataFrame({"id": data["test_ids"], "target": all_preds})

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
