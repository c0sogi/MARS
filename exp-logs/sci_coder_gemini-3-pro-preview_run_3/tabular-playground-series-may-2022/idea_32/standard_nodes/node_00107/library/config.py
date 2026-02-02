import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import roc_auc_score


# ==========================================
# Configuration
# ==========================================
class Config:
    # General
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    TRAIN_PATH = "./metadata/train.csv"
    VAL_PATH = "./metadata/val.csv"
    TEST_PATH = "./metadata/test.csv"
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"
    SUBMISSION_PATH = "./working/submission.csv"
    MODEL_SAVE_PATH = "./working/best_model.pth"
    CACHE_DIR = "./working/idea_32/"

    # Training Hyperparameters
    BATCH_SIZE = 1024
    EPOCHS = 50
    LEARNING_RATE = 1e-2  # Max LR for OneCycle
    WEIGHT_DECAY = 2e-5

    # Model Architecture
    EMBEDDING_DIM = 16

    # Stream Configurations
    # Format: {'hidden_dims': list, 'dropout': float}
    STREAMS_CONFIG = [
        {"hidden_dims": [512, 256, 128], "dropout": 0.20},  # Stream 1 (Anchor)
        {"hidden_dims": [512, 256, 128], "dropout": 0.20},  # Stream 2 (Anchor)
        {"hidden_dims": [1024, 512, 256], "dropout": 0.25},  # Stream 3 (Wide/Capacity)
        {"hidden_dims": [512, 256, 128], "dropout": 0.15},  # Stream 4 (Aggressive)
        {"hidden_dims": [512, 256, 128], "dropout": 0.30},  # Stream 5 (Conservative)
    ]


# ==========================================
# Utils
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# Data Processing
# ==========================================
def process_data(load_cached_data=True):
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    train_cache = os.path.join(Config.CACHE_DIR, "train_processed.parquet")
    val_cache = os.path.join(Config.CACHE_DIR, "val_processed.parquet")
    test_cache = os.path.join(Config.CACHE_DIR, "test_processed.parquet")
    meta_cache = os.path.join(Config.CACHE_DIR, "metadata.npy")

    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
        and os.path.exists(meta_cache)
    ):
        print("Loading cached data...")
        metadata = np.load(meta_cache, allow_pickle=True).item()

        # Validate metadata to ensure it matches current schema requirements (Cite debug_lesson_3)
        required_keys = ["cat_cols", "cont_cols", "vocab_sizes"]
        if all(k in metadata for k in required_keys):
            train_df = pd.read_parquet(train_cache)
            val_df = pd.read_parquet(val_cache)
            test_df = pd.read_parquet(test_cache)
            return train_df, val_df, test_df, metadata
        else:
            print("Cached metadata missing required keys. Reprocessing...")

    print("Processing data from scratch...")
    # Load raw data
    df_train = pd.read_csv(Config.TRAIN_PATH)
    df_val = pd.read_csv(Config.VAL_PATH)
    df_test = pd.read_csv(Config.TEST_PATH)

    # Feature Engineering Helper
    def engineer_features(df):
        # f_27 decomposition
        # Split string into 10 characters
        chars = df["f_27"].apply(lambda x: list(x))
        char_df = pd.DataFrame(
            chars.tolist(), columns=[f"f_27_{i}" for i in range(10)], index=df.index
        )

        # Unique character count
        df["unique_character_count"] = df["f_27"].apply(lambda x: len(set(x)))

        # Drop original f_27
        df = df.drop(columns=["f_27"])

        # Concat chars
        df = pd.concat([df, char_df], axis=1)
        return df

    df_train = engineer_features(df_train)
    df_val = engineer_features(df_val)
    df_test = engineer_features(df_test)

    # Identify columns
    # Categorical: f_29, f_30, and f_27_0...f_27_9
    cat_cols = ["f_29", "f_30"] + [f"f_27_{i}" for i in range(10)]
    # Continuous: All others except id, target, source_path
    exclude = ["id", "target", "source_path"] + cat_cols
    cont_cols = [c for c in df_train.columns if c not in exclude]

    # Transductive Encoding for Categoricals
    # Fit on all data to handle vocabulary
    full_cat = pd.concat(
        [df_train[cat_cols], df_val[cat_cols], df_test[cat_cols]], axis=0
    )
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(full_cat)

    df_train[cat_cols] = encoder.transform(df_train[cat_cols]).astype(int)
    df_val[cat_cols] = encoder.transform(df_val[cat_cols]).astype(int)
    df_test[cat_cols] = encoder.transform(df_test[cat_cols]).astype(int)

    # Get vocab sizes (max index + 1)
    vocab_sizes = [
        int(full_cat[col].nunique()) + 1 for col in cat_cols
    ]  # +1 for potential unknown safety
    # Actually OrdinalEncoder produces 0..N-1.
    # We will use the max value found + 1 as the embedding dictionary size.
    # Since we fit on all, unknown shouldn't happen, but good practice.
    vocab_sizes = [int(encoder.categories_[i].size) for i in range(len(cat_cols))]

    # Scaling Continuous Features
    scaler = StandardScaler()
    scaler.fit(df_train[cont_cols])

    df_train[cont_cols] = scaler.transform(df_train[cont_cols]).astype(np.float32)
    df_val[cont_cols] = scaler.transform(df_val[cont_cols]).astype(np.float32)
    df_test[cont_cols] = scaler.transform(df_test[cont_cols]).astype(np.float32)

    # Save to cache
    df_train.to_parquet(train_cache)
    df_val.to_parquet(val_cache)
    df_test.to_parquet(test_cache)

    metadata = {
        "cat_cols": cat_cols,
        "cont_cols": cont_cols,
        "vocab_sizes": vocab_sizes,
    }
    np.save(meta_cache, metadata)

    return df_train, df_val, df_test, metadata


# ==========================================
# Dataset
# ==========================================
class ManufacturingDataset(Dataset):
    def __init__(self, df, cat_cols, cont_cols, target_col=None):
        self.cat_data = df[cat_cols].values.astype(np.int64)
        self.cont_data = df[cont_cols].values.astype(np.float32)
        self.targets = (
            df[target_col].values.astype(np.float32)
            if target_col in df.columns
            else None
        )

    def __len__(self):
        return len(self.cat_data)

    def __getitem__(self, idx):
        cat = torch.tensor(self.cat_data[idx], dtype=torch.long)
        cont = torch.tensor(self.cont_data[idx], dtype=torch.float32)

        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return cat, cont, target
        return cat, cont


# ==========================================
# Model Architecture
# ==========================================
class Stream(nn.Module):
    def __init__(self, vocab_sizes, num_cont, hidden_dims, dropout_rate, embed_dim):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=v, embedding_dim=embed_dim)
                for v in vocab_sizes
            ]
        )

        # Input dim = continuous + flattened embeddings
        input_dim = num_cont + (len(vocab_sizes) * embed_dim)

        # Layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)

        # Auxiliary Head (attached to Layer 1 output)
        self.aux_head = nn.Linear(hidden_dims[0], 1)

        # Layer 2
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate)

        # Layer 3
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(dropout_rate)

        # Main Head
        self.main_head = nn.Linear(hidden_dims[2], 1)

    def forward(self, x_cat, x_cont):
        # Embeddings
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(embs, dim=1)

        # Early Fusion
        x = torch.cat([x_cont, x_emb], dim=1)

        # Layer 1
        x = self.drop1(self.act1(self.bn1(self.fc1(x))))

        # Aux Output
        aux_out = self.aux_head(x)

        # Layer 2
        x = self.drop2(self.act2(self.bn2(self.fc2(x))))

        # Layer 3
        x = self.drop3(self.act3(self.bn3(self.fc3(x))))

        # Main Output
        main_out = self.main_head(x)

        return main_out, aux_out


class DSPFE(nn.Module):
    def __init__(self, vocab_sizes, num_cont, stream_configs, embed_dim):
        super().__init__()
        self.streams = nn.ModuleList()
        for cfg in stream_configs:
            self.streams.append(
                Stream(
                    vocab_sizes=vocab_sizes,
                    num_cont=num_cont,
                    hidden_dims=cfg["hidden_dims"],
                    dropout_rate=cfg["dropout"],
                    embed_dim=embed_dim,
                )
            )

    def forward(self, x_cat, x_cont):
        main_outputs = []
        aux_outputs = []

        for stream in self.streams:
            m, a = stream(x_cat, x_cont)
            main_outputs.append(m)
            aux_outputs.append(a)

        return main_outputs, aux_outputs


# ==========================================
# Training & Inference
# ==========================================
def train_model(model, train_loader, val_loader, config):
    model.to(config.DEVICE)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=config.EPOCHS,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0

        for x_cat, x_cont, y in train_loader:
            x_cat, x_cont, y = (
                x_cat.to(config.DEVICE),
                x_cont.to(config.DEVICE),
                y.to(config.DEVICE).unsqueeze(1),
            )

            optimizer.zero_grad()
            main_outs, aux_outs = model(x_cat, x_cont)

            loss = 0
            for m, a in zip(main_outs, aux_outs):
                loss += criterion(m, y) + config.AUX_LOSS_WEIGHT * criterion(a, y)

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for x_cat, x_cont, y in val_loader:
                x_cat, x_cont = x_cat.to(config.DEVICE), x_cont.to(config.DEVICE)
                main_outs, _ = model(x_cat, x_cont)

                # Average probabilities from all streams
                probs = [torch.sigmoid(m) for m in main_outs]
                avg_prob = torch.stack(probs).mean(dim=0)

                val_preds.append(avg_prob.cpu().numpy())
                val_targets.append(y.numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_auc = roc_auc_score(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"Saved new best model with AUC: {best_auc:.6f}")


def predict_submission(model, test_loader, test_ids, config):
    model.load_state_dict(
        torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    )
    model.to(config.DEVICE)
    model.eval()

    all_preds = []

    with torch.no_grad():
        for x_cat, x_cont in test_loader:
            x_cat, x_cont = x_cat.to(config.DEVICE), x_cont.to(config.DEVICE)
            main_outs, _ = model(x_cat, x_cont)

            probs = [torch.sigmoid(m) for m in main_outs]
            avg_prob = torch.stack(probs).mean(dim=0)
            all_preds.append(avg_prob.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()

    submission = pd.DataFrame({"id": test_ids, "target": all_preds})
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")


# ==========================================
# Main Execution
# ==========================================
def main():
    set_seed(Config.SEED)

    # Process Data
    df_train, df_val, df_test, metadata = process_data(load_cached_data=True)

    # Datasets
    train_dataset = ManufacturingDataset(
        df_train, metadata["cat_cols"], metadata["cont_cols"], "target"
    )
    val_dataset = ManufacturingDataset(
        df_val, metadata["cat_cols"], metadata["cont_cols"], "target"
    )
    test_dataset = ManufacturingDataset(
        df_test, metadata["cat_cols"], metadata["cont_cols"], None
    )

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    model = DSPFE(
        vocab_sizes=metadata["vocab_sizes"],
        num_cont=len(metadata["cont_cols"]),
        stream_configs=Config.STREAMS_CONFIG,
        embed_dim=Config.EMBEDDING_DIM,
    )

    # Train
    train_model(model, train_loader, val_loader, Config)

    # Predict
    predict_submission(model, test_loader, df_test["id"].values, Config)


if __name__ == "__main__":
    main()
