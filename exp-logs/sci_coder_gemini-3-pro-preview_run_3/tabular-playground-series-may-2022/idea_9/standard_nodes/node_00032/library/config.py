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
class Config:
    # Hyperparameters
    SEED = 42
    EMBEDDING_DIM = 16
    HIDDEN_DIMS = [512, 256, 128]
    BATCH_SIZE = 1024
    WEIGHT_DECAY = 1e-5
    DROPOUT = 0.2
    MAX_LR = 1e-2
    EPOCHS = 30
    PATIENCE = 7

    # Paths
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    WORKING_DIR = "./working/idea_9"
    CACHE_DIR = WORKING_DIR
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)


def set_seed(seed=Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# DATASET
# ==========================================
class TabularDataset(Dataset):
    def __init__(self, x_cont, x_cat, y=None):
        self.x_cont = torch.FloatTensor(x_cont)
        self.x_cat = torch.LongTensor(x_cat)
        self.y = torch.FloatTensor(y).unsqueeze(1) if y is not None else None

    def __len__(self):
        return len(self.x_cont)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x_cont[idx], self.x_cat[idx], self.y[idx]
        else:
            return self.x_cont[idx], self.x_cat[idx]


# ==========================================
# DATA PROCESSING
# ==========================================
def process_data(load_cached_data=True):
    cache_path_data = os.path.join(Config.CACHE_DIR, "data_processed.pt")

    if load_cached_data and os.path.exists(cache_path_data):
        print("Loading cached data...")
        return torch.load(cache_path_data, weights_only=False)

    print("Processing data from scratch...")
    # Load Data
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # Store IDs for submission
    test_ids = test_df["id"].values

    # Feature Engineering on f_27
    def split_f27(df):
        # Split string into characters (assuming fixed length of 10)
        s = df["f_27"].astype(str)
        chars = list(zip(*[list(v) for v in s]))
        for i, c in enumerate(chars):
            df[f"f_27_{i}"] = c

        # Unique character count
        df["unique_character_count"] = s.apply(lambda x: len(set(x)))
        return df

    train_df = split_f27(train_df)
    val_df = split_f27(val_df)
    test_df = split_f27(test_df)

    # Define Column Groups
    # Continuous: f_00 to f_28 (excluding f_27) + unique_character_count
    cont_cols = [f"f_{i:02d}" for i in range(29) if i != 27] + [
        "unique_character_count"
    ]
    # Categorical: f_27_0...f_27_9, f_29, f_30
    cat_cols = [f"f_27_{i}" for i in range(10)] + ["f_29", "f_30"]

    # Transductive Categorical Encoding (Fit on Train + Val + Test)
    all_cat = pd.concat(
        [train_df[cat_cols], val_df[cat_cols], test_df[cat_cols]], axis=0
    )
    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64
    )
    enc.fit(all_cat)

    train_cat = enc.transform(train_df[cat_cols])
    val_cat = enc.transform(val_df[cat_cols])
    test_cat = enc.transform(test_df[cat_cols])

    # Calculate vocabulary sizes for embeddings
    vocab_sizes = [int(all_cat[c].nunique()) for c in cat_cols]

    # Continuous Scaling
    scaler = StandardScaler()
    train_cont = scaler.fit_transform(train_df[cont_cols])
    val_cont = scaler.transform(val_df[cont_cols])
    test_cont = scaler.transform(test_df[cont_cols])

    # Targets
    train_y = train_df["target"].values
    val_y = val_df["target"].values

    # Pack into dict
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
        "vocab_sizes": vocab_sizes,
        "cont_dim": train_cont.shape[1],
    }

    # Save cache
    torch.save(data, cache_path_data)
    return data


# ==========================================
# MODEL
# ==========================================
class InputInjectedFunnelMLP(nn.Module):
    def __init__(self, cont_dim, vocab_sizes, embed_dim, hidden_dims, dropout):
        super().__init__()

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=v, embedding_dim=embed_dim)
                for v in vocab_sizes
            ]
        )

        # Input dimension = continuous + (num_categorical * embed_dim)
        self.input_dim = cont_dim + (len(vocab_sizes) * embed_dim)

        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dims[0]), nn.ReLU(), nn.Dropout(dropout)
        )

        # Layer 2 (Input Injection: Concat Previous Layer + Original Input)
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dims[0] + self.input_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Layer 3 (Input Injection)
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dims[1] + self.input_dim, hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Output Head
        self.head = nn.Linear(hidden_dims[2], 1)

    def forward(self, x_cont, x_cat):
        # Embeddings
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(embs, dim=1)

        # Concatenate continuous and embeddings -> x_in
        x_in = torch.cat([x_cont, x_emb], dim=1)

        # Forward Pass with Input Injection
        h1 = self.layer1(x_in)

        h1_inj = torch.cat([h1, x_in], dim=1)
        h2 = self.layer2(h1_inj)

        h2_inj = torch.cat([h2, x_in], dim=1)
        h3 = self.layer3(h2_inj)

        out = self.head(h3)
        return out


# ==========================================
# TRAINING & INFERENCE
# ==========================================
def train_and_predict():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process Data
    data = process_data(load_cached_data=True)

    # Create Datasets
    train_ds = TabularDataset(data["train_cont"], data["train_cat"], data["train_y"])
    val_ds = TabularDataset(data["val_cont"], data["val_cat"], data["val_y"])
    test_ds = TabularDataset(data["test_cont"], data["test_cat"])

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Init Model
    model = InputInjectedFunnelMLP(
        cont_dim=data["cont_dim"],
        vocab_sizes=data["vocab_sizes"],
        embed_dim=Config.EMBEDDING_DIM,
        hidden_dims=Config.HIDDEN_DIMS,
        dropout=Config.DROPOUT,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.MAX_LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        steps_per_epoch=len(train_loader),
        epochs=Config.EPOCHS,
        pct_start=0.1,
    )

    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    best_auc = 0
    patience_counter = 0
    best_model_state = None

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0
        for x_cont, x_cat, y in train_loader:
            x_cont, x_cat, y = x_cont.to(device), x_cat.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x_cont, x_cat)
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
                x_cont, x_cat, y = x_cont.to(device), x_cat.to(device), y.to(device)
                logits = model(x_cont, x_cat)
                probs = torch.sigmoid(logits)
                val_preds.append(probs.cpu().numpy())
                val_targets.append(y.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_auc = roc_auc_score(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
            # Save best model
            torch.save(
                best_model_state, os.path.join(Config.WORKING_DIR, "best_model.pth")
            )
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Best Validation AUC: {best_auc:.6f}")

    # Inference
    print("Generating predictions...")
    model.load_state_dict(best_model_state)
    model.eval()

    test_preds = []
    with torch.no_grad():
        for x_cont, x_cat in test_loader:
            x_cont, x_cat = x_cont.to(device), x_cat.to(device)
            logits = model(x_cont, x_cat)
            probs = torch.sigmoid(logits)
            test_preds.append(probs.cpu().numpy())

    test_preds = np.concatenate(test_preds).flatten()

    # Submission
    sub_df = pd.DataFrame({"id": data["test_ids"], "target": test_preds})

    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
