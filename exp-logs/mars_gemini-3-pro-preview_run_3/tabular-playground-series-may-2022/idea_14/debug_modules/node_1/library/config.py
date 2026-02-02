import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import roc_auc_score


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
class Config:
    # Directories
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_14"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"

    # Input Files
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data Parameters
    SEED = 42
    NUM_WORKERS = 4

    # Model Architecture
    EMBEDDING_DIM = 16
    HIDDEN_LAYERS = [512, 256, 128]
    DROPOUT = 0.30

    # Training Hyperparameters
    BATCH_SIZE = 1024
    EPOCHS = 30
    LEARNING_RATE = 1e-2  # Max LR for OneCycle
    WEIGHT_DECAY = 1e-5
    PATIENCE = 5

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------------------
def set_seed(seed=Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------
class ManufacturingDataset(Dataset):
    def __init__(self, X_cat, X_cont, y=None):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_cont = torch.tensor(X_cont, dtype=torch.float32)
        self.y = (
            torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y is not None else None
        )

    def __len__(self):
        return len(self.X_cat)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_cat[idx], self.X_cont[idx], self.y[idx]
        return self.X_cat[idx], self.X_cont[idx]


# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------
class GatedBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate):
        super().__init__()
        # Project to 2x output size for GLU (splits into Value and Gate)
        self.linear = nn.Linear(in_features, out_features * 2)
        self.glu = nn.GLU(dim=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        x = self.glu(x)
        x = self.dropout(x)
        return x


class GatedFunnelNetwork(nn.Module):
    def __init__(self, vocab_sizes, num_cont_features, cfg=Config):
        super().__init__()

        # Embeddings
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=size, embedding_dim=cfg.EMBEDDING_DIM)
                for size in vocab_sizes
            ]
        )

        # Calculate input dimension after flattening embeddings + continuous features
        cat_input_dim = len(vocab_sizes) * cfg.EMBEDDING_DIM
        total_input_dim = cat_input_dim + num_cont_features

        # Funnel Layers
        layers = []
        in_dim = total_input_dim

        for hidden_dim in cfg.HIDDEN_LAYERS:
            layers.append(GatedBlock(in_dim, hidden_dim, cfg.DROPOUT))
            in_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Output Head
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x_cat, x_cont):
        # Process embeddings
        emb_list = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(emb_list, dim=1)

        # Early Fusion
        x = torch.cat([x_emb, x_cont], dim=1)

        # Forward pass
        x = self.feature_extractor(x)
        return self.head(x)


# ------------------------------------------------------------------------------
# Data Processing
# ------------------------------------------------------------------------------
def process_data(load_cached_data=True):
    """
    Loads, preprocesses, and caches data.
    Returns: (train_loader, val_loader, test_loader, vocab_sizes, num_cont)
    """
    set_seed()

    # Cache paths
    train_cache = os.path.join(Config.CACHE_DIR, "train_processed.parquet")
    val_cache = os.path.join(Config.CACHE_DIR, "val_processed.parquet")
    test_cache = os.path.join(Config.CACHE_DIR, "test_processed.parquet")
    meta_cache = os.path.join(Config.CACHE_DIR, "metadata.npy")

    # 1. Try Loading Cache
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
        and os.path.exists(meta_cache)
    ):
        print("Loading cached data...")
        df_train = pd.read_parquet(train_cache)
        df_val = pd.read_parquet(val_cache)
        df_test = pd.read_parquet(test_cache)
        metadata = np.load(meta_cache, allow_pickle=True).item()
        vocab_sizes = metadata["vocab_sizes"]

        # Identify columns
        cat_cols = [c for c in df_train.columns if c.startswith("cat_")]
        cont_cols = [c for c in df_train.columns if c.startswith("cont_")]

        # Create Datasets
        train_ds = ManufacturingDataset(
            df_train[cat_cols].values,
            df_train[cont_cols].values,
            df_train["target"].values,
        )
        val_ds = ManufacturingDataset(
            df_val[cat_cols].values, df_val[cont_cols].values, df_val["target"].values
        )
        test_ds = ManufacturingDataset(
            df_test[cat_cols].values, df_test[cont_cols].values, None
        )

        # Create Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader, vocab_sizes, len(cont_cols)

    # 2. Process from Scratch
    print("Processing data from scratch...")

    # Load raw metadata CSVs
    df_train = pd.read_csv(Config.TRAIN_PATH)
    df_val = pd.read_csv(Config.VAL_PATH)
    df_test = pd.read_csv(Config.TEST_PATH)

    # Mark splits for later separation
    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    # Concatenate for transductive processing
    full_df = pd.concat([df_train, df_val, df_test], axis=0, ignore_index=True)

    # Feature Engineering: f_27 decomposition
    # Split string into characters
    chars = full_df["f_27"].apply(lambda x: list(x))
    char_df = pd.DataFrame(chars.tolist(), columns=[f"f_27_{i}" for i in range(10)])

    # Unique character count
    full_df["unique_char_count"] = full_df["f_27"].apply(lambda x: len(set(x)))

    # Add char columns to full_df
    full_df = pd.concat([full_df, char_df], axis=1)

    # Define Column Groups
    # Categorical: f_27 characters, f_29, f_30
    cat_features = [f"f_27_{i}" for i in range(10)] + ["f_29", "f_30"]

    # Continuous: f_00 to f_26, f_28, unique_char_count
    # Note: f_27 is excluded (handled above), f_29/f_30 are categorical
    cont_features = [f"f_{i:02d}" for i in range(29) if i != 27] + ["unique_char_count"]

    # Encoding Categoricals
    print("Encoding categorical features...")
    ord_enc = OrdinalEncoder(
        dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1
    )
    full_df[cat_features] = ord_enc.fit_transform(full_df[cat_features].astype(str))

    # Calculate vocab sizes (max index + 1)
    vocab_sizes = full_df[cat_features].max().astype(int).values + 1

    # Scaling Continuous
    print("Scaling continuous features...")
    scaler = StandardScaler()
    full_df[cont_features] = scaler.fit_transform(full_df[cont_features])

    # Rename columns for clarity in cache
    rename_map = {c: f"cat_{c}" for c in cat_features}
    rename_map.update({c: f"cont_{c}" for c in cont_features})
    full_df = full_df.rename(columns=rename_map)

    final_cat_cols = [f"cat_{c}" for c in cat_features]
    final_cont_cols = [f"cont_{c}" for c in cont_features]

    # Split back
    train_proc = full_df[full_df["split"] == "train"].copy()
    val_proc = full_df[full_df["split"] == "val"].copy()
    test_proc = full_df[full_df["split"] == "test"].copy()

    # Save to cache
    print("Saving to cache...")
    train_proc.to_parquet(train_cache)
    val_proc.to_parquet(val_cache)
    test_proc.to_parquet(test_cache)

    metadata = {"vocab_sizes": vocab_sizes}
    np.save(meta_cache, metadata)

    # Create Datasets & Loaders
    train_ds = ManufacturingDataset(
        train_proc[final_cat_cols].values,
        train_proc[final_cont_cols].values,
        train_proc["target"].values,
    )
    val_ds = ManufacturingDataset(
        val_proc[final_cat_cols].values,
        val_proc[final_cont_cols].values,
        val_proc["target"].values,
    )
    test_ds = ManufacturingDataset(
        test_proc[final_cat_cols].values, test_proc[final_cont_cols].values, None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, vocab_sizes, len(final_cont_cols)


# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------
def train_model(load_cached_data=True):
    set_seed()

    # Data
    train_loader, val_loader, test_loader, vocab_sizes, num_cont = process_data(
        load_cached_data
    )

    # Model
    model = GatedFunnelNetwork(vocab_sizes, num_cont, Config).to(Config.DEVICE)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training on {Config.DEVICE}...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0

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
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0.0

        with torch.no_grad():
            for x_cat, x_cont, y in val_loader:
                x_cat, x_cont, y = (
                    x_cat.to(Config.DEVICE),
                    x_cont.to(Config.DEVICE),
                    y.to(Config.DEVICE),
                )
                logits = model(x_cat, x_cont)
                loss = criterion(logits, y)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(y.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_auc = roc_auc_score(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | Val AUC: {val_auc:.10f}"
        )

        # Early Stopping & Save Best
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"--> New Best Model Saved (AUC: {best_auc:.10f})")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    # Generate Submission
    predict_and_submit(model, test_loader, vocab_sizes, num_cont)


# ------------------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------------------
def predict_and_submit(model, test_loader, vocab_sizes, num_cont):
    print("Generating submission...")

    # Load best weights
    model = GatedFunnelNetwork(vocab_sizes, num_cont, Config).to(Config.DEVICE)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.eval()

    preds = []

    with torch.no_grad():
        for x_cat, x_cont in test_loader:
            x_cat, x_cont = x_cat.to(Config.DEVICE), x_cont.to(Config.DEVICE)
            logits = model(x_cat, x_cont)
            probs = torch.sigmoid(logits)
            preds.extend(probs.cpu().numpy().flatten())

    # Load test IDs from metadata
    df_test = pd.read_csv(Config.TEST_PATH)
    submission = pd.DataFrame({"id": df_test["id"], "target": preds})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


# Entry point for the module
def run_pipeline():
    train_model(load_cached_data=True)
