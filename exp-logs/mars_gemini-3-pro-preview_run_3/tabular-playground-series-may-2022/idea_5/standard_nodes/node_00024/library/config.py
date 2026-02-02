import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import roc_auc_score


class Config:
    # Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"

    # Data Files
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Hyperparameters
    EMBEDDING_DIM = 32
    HIDDEN_LAYERS = [512, 256, 128]
    DROPOUT_RATE = 0.25

    # Training Hyperparameters
    SEED = 42
    BATCH_SIZE = 2048
    EPOCHS = 30
    LEARNING_RATE = 1e-2
    PATIENCE = 5
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TabularDataset(Dataset):
    def __init__(self, x_cont, x_cat, y=None):
        self.x_cont = torch.FloatTensor(x_cont)
        self.x_cat = torch.LongTensor(x_cat)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.x_cont)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x_cont[idx], self.x_cat[idx], self.y[idx]
        return self.x_cont[idx], self.x_cat[idx]


class GatedBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        # GLU requires input size 2 * out_dim
        self.linear = nn.Linear(in_dim, out_dim * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.ln(x)
        x = self.linear(x)
        x = F.glu(x, dim=-1)
        x = self.dropout(x)
        return x


class LayerNormGatedFunnelNetwork(nn.Module):
    def __init__(
        self, num_cont, cat_cardinalities, embedding_dim, hidden_layers, dropout
    ):
        super().__init__()

        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, embedding_dim) for card in cat_cardinalities]
        )

        # Calculate input dimension: Continuous features + Flattened embeddings
        self.input_dim = num_cont + len(cat_cardinalities) * embedding_dim

        layers = []
        in_dim = self.input_dim

        for h_dim in hidden_layers:
            layers.append(GatedBlock(in_dim, h_dim, dropout))
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_layers[-1], 1)

    def forward(self, x_cont, x_cat):
        # x_cat: [batch, num_cat_features]
        emb_list = []
        for i, emb_layer in enumerate(self.embeddings):
            emb_list.append(emb_layer(x_cat[:, i]))

        # Concatenate embeddings: [batch, num_cat * emb_dim]
        x_emb = torch.cat(emb_list, dim=1)

        # Concatenate with continuous: [batch, num_cont + num_cat * emb_dim]
        x = torch.cat([x_cont, x_emb], dim=1)

        x = self.backbone(x)
        return self.head(x)


def process_data(load_cached_data=True):
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_files = {
        "train_cont": os.path.join(Config.WORKING_DIR, "train_cont.npy"),
        "train_cat": os.path.join(Config.WORKING_DIR, "train_cat.npy"),
        "train_y": os.path.join(Config.WORKING_DIR, "train_y.npy"),
        "val_cont": os.path.join(Config.WORKING_DIR, "val_cont.npy"),
        "val_cat": os.path.join(Config.WORKING_DIR, "val_cat.npy"),
        "val_y": os.path.join(Config.WORKING_DIR, "val_y.npy"),
        "test_cont": os.path.join(Config.WORKING_DIR, "test_cont.npy"),
        "test_cat": os.path.join(Config.WORKING_DIR, "test_cat.npy"),
        "test_ids": os.path.join(Config.WORKING_DIR, "test_ids.npy"),
        "meta": os.path.join(Config.WORKING_DIR, "meta.npy"),
    }

    # Check cache
    if load_cached_data and all(os.path.exists(p) for p in cache_files.values()):
        print("Loading cached data...")
        data = {k: np.load(v) for k, v in cache_files.items() if k != "meta"}
        meta = np.load(cache_files["meta"], allow_pickle=True).item()
        return data, meta

    print("Processing data from scratch...")

    # Load Metadata CSVs
    df_train = pd.read_csv(Config.TRAIN_PATH)
    df_val = pd.read_csv(Config.VAL_PATH)
    df_test = pd.read_csv(Config.TEST_PATH)

    # Store IDs and Targets
    train_y = df_train["target"].values
    val_y = df_val["target"].values
    test_ids = df_test["id"].values

    # Combine for global vocabulary and scaling
    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    # Drop target and id for feature processing
    full_df = pd.concat(
        [
            df_train.drop(columns=["target", "id", "source_path"]),
            df_val.drop(columns=["target", "id", "source_path"]),
            df_test.drop(columns=["id", "source_path"]),
        ],
        axis=0,
        ignore_index=True,
    )

    # --- Feature Engineering ---

    # 1. f_27 decomposition: Split string into 10 columns
    f27_chars = full_df["f_27"].apply(list).tolist()
    f27_cols = [f"f_27_{i}" for i in range(10)]
    df_f27 = pd.DataFrame(f27_chars, columns=f27_cols)

    # 2. Unique character count
    full_df["unique_char_count"] = full_df["f_27"].apply(lambda x: len(set(x)))

    # 3. Categorical Features: f_29, f_30, and the 10 f_27 chars
    cat_features = f27_cols + ["f_29", "f_30"]

    # Add f_27 char columns to full_df
    for col in f27_cols:
        full_df[col] = df_f27[col]

    # Ordinal Encoding for Categoricals (Global Vocabulary)
    enc = OrdinalEncoder(dtype=np.int64)
    full_df[cat_features] = enc.fit_transform(full_df[cat_features].astype(str))

    # Get cardinalities for embeddings
    cat_cardinalities = [int(full_df[col].max() + 1) for col in cat_features]

    # 4. Continuous Features: All except cat features, split, f_27
    cont_features = [
        c for c in full_df.columns if c not in cat_features + ["split", "f_27"]
    ]

    scaler = StandardScaler()
    full_df[cont_features] = scaler.fit_transform(full_df[cont_features])

    # --- Split back ---
    x_train = full_df[full_df["split"] == "train"]
    x_val = full_df[full_df["split"] == "val"]
    x_test = full_df[full_df["split"] == "test"]

    # Extract arrays
    train_cont = x_train[cont_features].values.astype(np.float32)
    train_cat = x_train[cat_features].values.astype(np.int64)

    val_cont = x_val[cont_features].values.astype(np.float32)
    val_cat = x_val[cat_features].values.astype(np.int64)

    test_cont = x_test[cont_features].values.astype(np.float32)
    test_cat = x_test[cat_features].values.astype(np.int64)

    # Save to cache
    np.save(cache_files["train_cont"], train_cont)
    np.save(cache_files["train_cat"], train_cat)
    np.save(cache_files["train_y"], train_y)
    np.save(cache_files["val_cont"], val_cont)
    np.save(cache_files["val_cat"], val_cat)
    np.save(cache_files["val_y"], val_y)
    np.save(cache_files["test_cont"], test_cont)
    np.save(cache_files["test_cat"], test_cat)
    np.save(cache_files["test_ids"], test_ids)

    meta = {"cat_cardinalities": cat_cardinalities, "num_cont": len(cont_features)}
    np.save(cache_files["meta"], meta)

    data = {
        "train_cont": train_cont,
        "train_cat": train_cat,
        "train_y": train_y,
        "val_cont": val_cont,
        "val_cat": val_cat,
        "val_y": val_y,
        "test_cont": test_cont,
        "test_cat": test_cat,
        "test_ids": test_ids,
    }

    return data, meta


def train_model(load_cached_data=True, epochs=Config.EPOCHS):
    set_seed(Config.SEED)

    # Data
    data, meta = process_data(load_cached_data)

    train_dataset = TabularDataset(
        data["train_cont"], data["train_cat"], data["train_y"]
    )
    val_dataset = TabularDataset(data["val_cont"], data["val_cat"], data["val_y"])
    test_dataset = TabularDataset(data["test_cont"], data["test_cat"])

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

    # Model
    model = LayerNormGatedFunnelNetwork(
        num_cont=meta["num_cont"],
        cat_cardinalities=meta["cat_cardinalities"],
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_layers=Config.HIDDEN_LAYERS,
        dropout=Config.DROPOUT_RATE,
    ).to(Config.DEVICE)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-2
    )
    criterion = nn.BCEWithLogitsLoss()

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,
    )

    # Training Loop
    best_auc = 0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {Config.DEVICE}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for x_cont, x_cat, y in train_loader:
            x_cont, x_cat, y = (
                x_cont.to(Config.DEVICE),
                x_cat.to(Config.DEVICE),
                y.to(Config.DEVICE).unsqueeze(1),
            )

            optimizer.zero_grad()
            outputs = model(x_cont, x_cat)
            loss = criterion(outputs, y)
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
            for x_cont, x_cat, y in val_loader:
                x_cont, x_cat = x_cont.to(Config.DEVICE), x_cat.to(Config.DEVICE)
                outputs = model(x_cont, x_cat)
                val_preds.append(torch.sigmoid(outputs).cpu().numpy())
                val_targets.append(y.numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_auc = roc_auc_score(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val AUC: {val_auc:.10f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    # Load best model for inference
    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))

    # Inference
    print("Generating predictions...")
    model.eval()
    test_preds = []

    with torch.no_grad():
        for x_cont, x_cat in test_loader:
            x_cont, x_cat = x_cont.to(Config.DEVICE), x_cat.to(Config.DEVICE)
            outputs = model(x_cont, x_cat)
            test_preds.append(torch.sigmoid(outputs).cpu().numpy())

    test_preds = np.concatenate(test_preds).flatten()

    # Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission = pd.DataFrame({"id": data["test_ids"], "target": test_preds})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
