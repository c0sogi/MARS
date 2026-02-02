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
# CONFIGURATION
# ==========================================

TRAIN_PARAMS = {
    "batch_size": 1024,
    "epochs": 50,
    "learning_rate": 1e-2,
    "weight_decay": 1e-4,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4,
}

STREAM_CONFIGS = [
    {"hidden_dims": [512, 256, 128], "dropout": 0.20},  # Stream 1 (Anchor)
    {"hidden_dims": [512, 256, 128], "dropout": 0.20},  # Stream 2 (Anchor)
    {"hidden_dims": [1024, 512, 256], "dropout": 0.25},  # Stream 3 (Capacity Variant)
    {"hidden_dims": [512, 256, 128], "dropout": 0.15},  # Stream 4 (Safe-Aggressive)
    {"hidden_dims": [512, 256, 128], "dropout": 0.30},  # Stream 5 (Conservative)
]

CACHE_DIR = "./working/idea_optimized/"
SUBMISSION_DIR = "./submission/"
METADATA_DIR = "./metadata/"

# ==========================================
# UTILS
# ==========================================


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================================
# DATA PROCESSING
# ==========================================


def process_data(load_cached_data=True):
    os.makedirs(CACHE_DIR, exist_ok=True)

    train_cache = os.path.join(CACHE_DIR, "train_processed.parquet")
    val_cache = os.path.join(CACHE_DIR, "val_processed.parquet")
    test_cache = os.path.join(CACHE_DIR, "test_processed.parquet")
    meta_cache = os.path.join(CACHE_DIR, "metadata.npy")

    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
        and os.path.exists(meta_cache)
    ):
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
        metadata = np.load(meta_cache, allow_pickle=True).item()
        return train_df, val_df, test_df, metadata

    # Load metadata splits
    df_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    df_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    df_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Feature Engineering
    def engineer_features(df):
        # f_27 decomposition
        for i in range(10):
            df[f"ch_{i}"] = df["f_27"].str[i]

        # unique_character_count
        df["unique_char_count"] = df["f_27"].apply(lambda x: len(set(x)))
        return df

    df_train = engineer_features(df_train)
    df_val = engineer_features(df_val)
    df_test = engineer_features(df_test)

    # Define columns
    cont_cols = [f"f_{i:02d}" for i in range(27)] + ["f_28", "unique_char_count"]
    cat_cols = [f"ch_{i}" for i in range(10)] + ["f_29", "f_30"]

    # Transductive Vocabulary Alignment
    full_cat = pd.concat(
        [df_train[cat_cols], df_val[cat_cols], df_test[cat_cols]], axis=0
    )

    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int32
    )
    enc.fit(full_cat)

    df_train[cat_cols] = enc.transform(df_train[cat_cols])
    df_val[cat_cols] = enc.transform(df_val[cat_cols])
    df_test[cat_cols] = enc.transform(df_test[cat_cols])

    cat_cardinalities = [int(full_cat[col].nunique()) for col in cat_cols]

    # Normalization (Fit on Train only)
    scaler = StandardScaler()
    df_train[cont_cols] = scaler.fit_transform(df_train[cont_cols])
    df_val[cont_cols] = scaler.transform(df_val[cont_cols])
    df_test[cont_cols] = scaler.transform(df_test[cont_cols])

    # Cast to appropriate types
    for col in cont_cols:
        df_train[col] = df_train[col].astype(np.float32)
        df_val[col] = df_val[col].astype(np.float32)
        df_test[col] = df_test[col].astype(np.float32)

    for col in cat_cols:
        df_train[col] = df_train[col].astype(np.int32)
        df_val[col] = df_val[col].astype(np.int32)
        df_test[col] = df_test[col].astype(np.int32)

    # Save to cache
    df_train.to_parquet(train_cache)
    df_val.to_parquet(val_cache)
    df_test.to_parquet(test_cache)

    metadata = {
        "cont_cols": cont_cols,
        "cat_cols": cat_cols,
        "cat_cardinalities": cat_cardinalities,
    }
    np.save(meta_cache, metadata)

    return df_train, df_val, df_test, metadata


class TabularDataset(Dataset):
    def __init__(self, df, cont_cols, cat_cols, target_col=None):
        self.cont = df[cont_cols].values
        self.cat = df[cat_cols].values
        self.target = df[target_col].values if target_col in df.columns else None

    def __len__(self):
        return len(self.cont)

    def __getitem__(self, idx):
        x_cont = torch.tensor(self.cont[idx], dtype=torch.float32)
        x_cat = torch.tensor(self.cat[idx], dtype=torch.long)

        if self.target is not None:
            y = torch.tensor(self.target[idx], dtype=torch.float32)
            return x_cont, x_cat, y
        else:
            return x_cont, x_cat


# ==========================================
# MODEL
# ==========================================


class Stream(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.out(self.mlp(x))


class IAPEModel(nn.Module):
    def __init__(self, num_cont, cat_cardinalities, stream_configs):
        super().__init__()
        self.num_streams = len(stream_configs)
        self.emb_dim = 16

        # Independent embeddings for each stream
        self.stream_embeddings = nn.ModuleList()
        self.streams_deep = nn.ModuleList()
        self.streams_wide_linear = nn.ModuleList()

        # Calculate Interaction Dimension: 10 chars from f_27 -> 10*9/2 = 45 pairs
        self.num_interactions = 45
        self.wide_input_dim = num_cont + self.num_interactions

        # Deep input dim: Continuous + All Embeddings
        self.deep_input_dim = num_cont + len(cat_cardinalities) * self.emb_dim

        for config in stream_configs:
            # Embeddings for this stream
            embs = nn.ModuleList(
                [nn.Embedding(c, self.emb_dim) for c in cat_cardinalities]
            )
            self.stream_embeddings.append(embs)

            # Deep Path
            self.streams_deep.append(
                Stream(self.deep_input_dim, config["hidden_dims"], config["dropout"])
            )

            # Wide Path Linear Projection
            self.streams_wide_linear.append(nn.Linear(self.wide_input_dim, 1))

    def forward(self, x_cont, x_cat):
        outputs = []
        for i in range(self.num_streams):
            # 1. Get Embeddings for this stream
            embeddings = []
            for j, emb_layer in enumerate(self.stream_embeddings[i]):
                embeddings.append(emb_layer(x_cat[:, j]))

            # Stack embeddings: [batch, 12, 16] -> Flatten for Deep Path: [batch, 12*16]
            all_embs_flat = torch.cat(embeddings, dim=1)

            # 2. Deep Path
            deep_in = torch.cat([x_cont, all_embs_flat], dim=1)
            deep_out = self.streams_deep[i](deep_in)

            # 3. Wide Path (Interaction)
            # Only use first 10 embeddings (f_27 chars)
            char_embs = embeddings[:10]

            # Compute pairwise dot products
            interactions = []
            for idx1 in range(10):
                for idx2 in range(idx1 + 1, 10):
                    dot = (char_embs[idx1] * char_embs[idx2]).sum(dim=1, keepdim=True)
                    interactions.append(dot)

            interaction_vec = torch.cat(interactions, dim=1)

            wide_in = torch.cat([x_cont, interaction_vec], dim=1)
            wide_out = self.streams_wide_linear[i](wide_in)

            # 4. Aggregation
            total_logit = deep_out + wide_out
            outputs.append(total_logit)

        return torch.cat(outputs, dim=1)


# ==========================================
# EXECUTION
# ==========================================


def run_pipeline():
    set_seed(TRAIN_PARAMS["seed"])

    # Data
    train_df, val_df, test_df, metadata = process_data(load_cached_data=True)

    train_ds = TabularDataset(
        train_df, metadata["cont_cols"], metadata["cat_cols"], "target"
    )
    val_ds = TabularDataset(
        val_df, metadata["cont_cols"], metadata["cat_cols"], "target"
    )
    test_ds = TabularDataset(test_df, metadata["cont_cols"], metadata["cat_cols"])

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_PARAMS["batch_size"],
        shuffle=True,
        num_workers=TRAIN_PARAMS["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=TRAIN_PARAMS["batch_size"],
        shuffle=False,
        num_workers=TRAIN_PARAMS["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=TRAIN_PARAMS["batch_size"],
        shuffle=False,
        num_workers=TRAIN_PARAMS["num_workers"],
        pin_memory=True,
    )

    # Model
    model = IAPEModel(
        num_cont=len(metadata["cont_cols"]),
        cat_cardinalities=metadata["cat_cardinalities"],
        stream_configs=STREAM_CONFIGS,
    ).to(TRAIN_PARAMS["device"])

    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAIN_PARAMS["learning_rate"],
        weight_decay=TRAIN_PARAMS["weight_decay"],
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=TRAIN_PARAMS["learning_rate"],
        epochs=TRAIN_PARAMS["epochs"],
        steps_per_epoch=len(train_loader),
    )

    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_model_path = os.path.join(CACHE_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(TRAIN_PARAMS["epochs"]):
        model.train()
        train_loss = 0

        for x_cont, x_cat, y in train_loader:
            x_cont, x_cat, y = (
                x_cont.to(TRAIN_PARAMS["device"]),
                x_cat.to(TRAIN_PARAMS["device"]),
                y.to(TRAIN_PARAMS["device"]),
            )

            optimizer.zero_grad()
            logits = model(x_cont, x_cat)

            # Sum of BCE losses
            loss = 0
            for i in range(5):
                loss += criterion(logits[:, i], y)

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for x_cont, x_cat, y in val_loader:
                x_cont, x_cat = x_cont.to(TRAIN_PARAMS["device"]), x_cat.to(
                    TRAIN_PARAMS["device"]
                )
                logits = model(x_cont, x_cat)
                probs = torch.sigmoid(logits).mean(dim=1).cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend(y.numpy())

        val_auc = roc_auc_score(val_targets, val_preds)
        print(
            f"Epoch {epoch+1}/{TRAIN_PARAMS['epochs']} | Train Loss: {train_loss/len(train_loader):.6f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

    print(f"Best Validation AUC: {best_auc:.6f}")

    # Inference
    print("Generating submission...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_preds = []
    with torch.no_grad():
        for x_cont, x_cat in test_loader:
            x_cont, x_cat = x_cont.to(TRAIN_PARAMS["device"]), x_cat.to(
                TRAIN_PARAMS["device"]
            )
            logits = model(x_cont, x_cat)
            probs = torch.sigmoid(logits).mean(dim=1).cpu().numpy()
            test_preds.extend(probs)

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission = pd.DataFrame({"id": test_df["id"], "target": test_preds})
    submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)
    print("Submission saved.")


run_pipeline()
