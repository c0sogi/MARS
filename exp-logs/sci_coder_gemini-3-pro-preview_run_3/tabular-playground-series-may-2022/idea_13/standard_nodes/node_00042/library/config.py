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
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


# ==========================================
# Configuration
# ==========================================
class Config:
    # Reproducibility
    SEED = 42

    # File Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_13"
    SUBMISSION_DIR = "./submission"

    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Data Processing
    CACHE_DIR = WORKING_DIR
    LOAD_CACHED_DATA = True

    # Model Architecture (MLP)
    EMBEDDING_DIM = 16
    HIDDEN_LAYERS = [512, 256, 128]
    DROPOUT_RATE = 0.2  # Standard Dropout

    # Training
    BATCH_SIZE = 1024
    EPOCHS = 30
    LEARNING_RATE = 1e-2  # Max LR for OneCycle
    WEIGHT_DECAY = 1e-5
    PATIENCE = 5
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Ensure directories exist
os.makedirs(Config.WORKING_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


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
def feature_engineering(df):
    """
    Applies feature engineering:
    1. Decomposes f_27 into 10 characters.
    2. Computes unique_character_count for f_27.
    """
    # Decompose f_27
    # Assuming f_27 is always length 10 based on typical synthetic datasets of this type
    # We will safely iterate.

    # Convert to string just in case
    s = df["f_27"].astype(str)

    # 1. Unique character count
    df["unique_character_count"] = s.apply(lambda x: len(set(x)))

    # 2. Split into characters
    # We create 10 new columns: f_27_0, f_27_1, ...
    # Vectorized string splitting
    # Pad with a placeholder if shorter than 10 (though likely fixed)
    max_len = 10
    chars = s.str.pad(width=max_len, side="right", fillchar="_").apply(list)
    chars_df = pd.DataFrame(chars.tolist(), index=df.index)
    chars_df.columns = [f"f_27_{i}" for i in range(max_len)]

    # Concatenate
    df = pd.concat([df, chars_df], axis=1)

    return df


def process_data(load_cached_data=True):
    """
    Loads, processes, and caches data.
    """
    cache_files = {
        "train": os.path.join(Config.CACHE_DIR, "train_processed.parquet"),
        "val": os.path.join(Config.CACHE_DIR, "val_processed.parquet"),
        "test": os.path.join(Config.CACHE_DIR, "test_processed.parquet"),
        "meta": os.path.join(
            Config.CACHE_DIR, "meta.npy"
        ),  # Stores vocab sizes, scalers
    }

    # Check if cache exists
    if load_cached_data and all(os.path.exists(p) for p in cache_files.values()):
        print("Loading cached data...")
        train_df = pd.read_parquet(cache_files["train"])
        val_df = pd.read_parquet(cache_files["val"])
        test_df = pd.read_parquet(cache_files["test"])
        meta = np.load(cache_files["meta"], allow_pickle=True).item()
        return train_df, val_df, test_df, meta

    print("Processing data from scratch...")

    # Load Metadata CSVs
    train_df = pd.read_csv(Config.TRAIN_META)
    val_df = pd.read_csv(Config.VAL_META)
    test_df = pd.read_csv(Config.TEST_META)

    # Feature Engineering
    print("Applying feature engineering...")
    train_df = feature_engineering(train_df)
    val_df = feature_engineering(val_df)
    test_df = feature_engineering(test_df)

    # Define Column Groups
    # Continuous: f_00 to f_28 (excluding f_27) + unique_character_count
    cont_cols = [f"f_{i:02d}" for i in range(29) if i != 27] + [
        "unique_character_count"
    ]

    # Categorical: f_29, f_30 + f_27_0...f_27_9
    cat_cols = ["f_29", "f_30"] + [f"f_27_{i}" for i in range(10)]

    # Transductive Label Encoding
    # Fit on Train + Val + Test to handle vocabulary alignment
    print("Encoding categorical features...")
    full_cat = pd.concat(
        [train_df[cat_cols], val_df[cat_cols], test_df[cat_cols]], axis=0
    )
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64
    )
    encoder.fit(full_cat)

    train_df[cat_cols] = encoder.transform(train_df[cat_cols])
    val_df[cat_cols] = encoder.transform(val_df[cat_cols])
    test_df[cat_cols] = encoder.transform(test_df[cat_cols])

    # Get vocab sizes (max index + 1)
    # We add 1 for potential unknowns if any, though transductive covers all seen
    vocab_sizes = [int(full_cat[col].nunique()) for col in cat_cols]

    # Normalization (StandardScaler)
    # Fit on Train only
    print("Normalizing continuous features...")
    scaler = StandardScaler()
    train_df[cont_cols] = scaler.fit_transform(train_df[cont_cols])
    val_df[cont_cols] = scaler.transform(val_df[cont_cols])
    test_df[cont_cols] = scaler.transform(test_df[cont_cols])

    # Save to Cache
    print("Saving to cache...")
    train_df.to_parquet(cache_files["train"])
    val_df.to_parquet(cache_files["val"])
    test_df.to_parquet(cache_files["test"])

    meta = {"cont_cols": cont_cols, "cat_cols": cat_cols, "vocab_sizes": vocab_sizes}
    np.save(cache_files["meta"], meta)

    return train_df, val_df, test_df, meta


# ==========================================
# Dataset
# ==========================================
class ManufacturingDataset(Dataset):
    def __init__(self, df, cont_cols, cat_cols, target_col="target", is_test=False):
        self.cont_data = df[cont_cols].values.astype(np.float32)
        self.cat_data = df[cat_cols].values.astype(np.int64)
        self.is_test = is_test

        if not is_test:
            self.targets = df[target_col].values.astype(np.float32)
        else:
            self.targets = None
            self.ids = df["id"].values

    def __len__(self):
        return len(self.cont_data)

    def __getitem__(self, idx):
        x_cont = self.cont_data[idx]
        x_cat = self.cat_data[idx]

        if self.is_test:
            return x_cont, x_cat, self.ids[idx]
        else:
            return x_cont, x_cat, self.targets[idx]


# ==========================================
# Model: Self-Normalizing Funnel Network
# ==========================================
class SNN(nn.Module):
    def __init__(self, num_cont, vocab_sizes, embed_dim, hidden_layers, dropout_rate):
        super(SNN, self).__init__()

        # Embeddings
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=v, embedding_dim=embed_dim)
                for v in vocab_sizes
            ]
        )

        # Calculate input dimension for the MLP
        # Continuous features + (Number of categorical features * Embedding dim)
        input_dim = num_cont + (len(vocab_sizes) * embed_dim)

        # Backbone
        layers = []
        in_dim = input_dim

        for h_dim in hidden_layers:
            linear = nn.Linear(in_dim, h_dim)
            # LeCun Normal Initialization for SNN
            nn.init.normal_(linear.weight, std=np.sqrt(1.0 / in_dim))
            nn.init.zeros_(linear.bias)

            layers.append(linear)
            layers.append(nn.SELU())
            layers.append(nn.AlphaDropout(p=dropout_rate))
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)

        # Head
        self.head = nn.Linear(in_dim, 1)
        # Init head
        nn.init.normal_(self.head.weight, std=np.sqrt(1.0 / in_dim))
        nn.init.zeros_(self.head.bias)

    def forward(self, x_cont, x_cat):
        # Process embeddings
        embedded = []
        for i, emb_layer in enumerate(self.embeddings):
            embedded.append(emb_layer(x_cat[:, i]))

        # Flatten and concatenate
        x_emb = torch.cat(embedded, dim=1)
        x = torch.cat([x_cont, x_emb], dim=1)

        # Pass through backbone
        x = self.backbone(x)

        # Output
        logits = self.head(x)
        return logits


# ==========================================
# Training & Inference
# ==========================================
def train_model(train_loader, val_loader, meta):
    set_seed(Config.SEED)

    model = SNN(
        num_cont=len(meta["cont_cols"]),
        vocab_sizes=meta["vocab_sizes"],
        embed_dim=Config.EMBEDDING_DIM,
        hidden_layers=Config.HIDDEN_LAYERS,
        dropout_rate=Config.DROPOUT_RATE,
    ).to(Config.DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=Config.EPOCHS,
        pct_start=0.1,
    )

    best_auc = 0.0
    patience_counter = 0

    print(f"Starting training on {Config.DEVICE}...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0

        for x_cont, x_cat, y in train_loader:
            x_cont, x_cat, y = (
                x_cont.to(Config.DEVICE),
                x_cat.to(Config.DEVICE),
                y.to(Config.DEVICE),
            )

            optimizer.zero_grad()
            logits = model(x_cont, x_cat).squeeze()
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for x_cont, x_cat, y in val_loader:
                x_cont, x_cat = x_cont.to(Config.DEVICE), x_cat.to(Config.DEVICE)
                logits = model(x_cont, x_cat).squeeze()
                probs = torch.sigmoid(logits)

                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(y.numpy())

        val_auc = roc_auc_score(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val AUC: {val_auc:.10f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Best Validation AUC: {best_auc:.10f}")
    return best_auc


def generate_submission(test_loader, meta):
    print("Generating submission...")

    # Load best model
    model = SNN(
        num_cont=len(meta["cont_cols"]),
        vocab_sizes=meta["vocab_sizes"],
        embed_dim=Config.EMBEDDING_DIM,
        hidden_layers=Config.HIDDEN_LAYERS,
        dropout_rate=Config.DROPOUT_RATE,
    ).to(Config.DEVICE)

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.eval()

    ids_list = []
    preds_list = []

    with torch.no_grad():
        for x_cont, x_cat, ids in test_loader:
            x_cont, x_cat = x_cont.to(Config.DEVICE), x_cat.to(Config.DEVICE)
            logits = model(x_cont, x_cat).squeeze()
            probs = torch.sigmoid(logits)

            ids_list.extend(ids)
            preds_list.extend(probs.cpu().numpy())

    submission = pd.DataFrame({"id": ids_list, "target": preds_list})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_pipeline():
    set_seed(Config.SEED)

    # 1. Process Data
    train_df, val_df, test_df, meta = process_data(
        load_cached_data=Config.LOAD_CACHED_DATA
    )

    # 2. Create Datasets & Loaders
    train_dataset = ManufacturingDataset(
        train_df, meta["cont_cols"], meta["cat_cols"], is_test=False
    )
    val_dataset = ManufacturingDataset(
        val_df, meta["cont_cols"], meta["cat_cols"], is_test=False
    )
    test_dataset = ManufacturingDataset(
        test_df, meta["cont_cols"], meta["cat_cols"], is_test=True
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

    # 3. Train
    train_model(train_loader, val_loader, meta)

    # 4. Predict
    generate_submission(test_loader, meta)


# Execute pipeline
if __name__ == "__main__":
    pass  # Standard block kept empty as per instructions, but code below runs in global scope if needed.
