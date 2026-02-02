import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import roc_auc_score

# ==========================================
# 1. Low-Level Utilities
# ==========================================


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ==========================================
# 2. Data Processing & Caching
# ==========================================


def process_data(
    load_cached_data=True, base_dir="./metadata", cache_dir="./working/idea_33"
):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "processed_data.npz")
    vocab_file = os.path.join(cache_dir, "vocab_sizes.npy")

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_file) and os.path.exists(vocab_file):
        print("Loading cached data from", cache_dir)
        try:
            data = np.load(cache_file)
            vocab_sizes = np.load(vocab_file)
            return (
                torch.tensor(data["X_cat_train"], dtype=torch.long),
                torch.tensor(data["X_cont_train"], dtype=torch.float32),
                torch.tensor(data["y_train"], dtype=torch.float32),
                torch.tensor(data["X_cat_val"], dtype=torch.long),
                torch.tensor(data["X_cont_val"], dtype=torch.float32),
                torch.tensor(data["y_val"], dtype=torch.float32),
                torch.tensor(data["X_cat_test"], dtype=torch.long),
                torch.tensor(data["X_cont_test"], dtype=torch.float32),
                torch.tensor(data["test_ids"], dtype=torch.long),
                vocab_sizes.tolist(),
            )
        except Exception as e:
            print(f"Cache load failed ({e}). Reprocessing...")

    print("Processing data from scratch...")

    # Load Metadata CSVs
    train_df = pd.read_csv(os.path.join(base_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(base_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(base_dir, "test.csv"))

    # Feature Engineering Function
    def engineer_features(df):
        # 1. Unique character count (Continuous)
        df["unique_char_count"] = df["f_27"].apply(lambda x: len(set(x)))

        # 2. String decomposition (10 chars -> Categorical)
        for i in range(10):
            df[f"p_{i}"] = df["f_27"].str[i]
        return df

    train_df = engineer_features(train_df)
    val_df = engineer_features(val_df)
    test_df = engineer_features(test_df)

    # Define Column Groups
    # Categorical: f_29, f_30, p_0...p_9
    cat_cols = ["f_29", "f_30"] + [f"p_{i}" for i in range(10)]

    # Continuous: f_00...f_28 (excluding f_27) + unique_char_count
    # Note: f_29, f_30 are categorical, so exclude them from continuous range if present
    # The dataset has f_00 to f_30.
    cont_cols = [f"f_{i:02d}" for i in range(29) if i != 27] + ["unique_char_count"]

    # Transductive Label Encoding (Train + Val + Test)
    # Convert to string to handle mixed types safely
    all_cat = pd.concat(
        [train_df[cat_cols], val_df[cat_cols], test_df[cat_cols]], axis=0
    ).astype(str)

    encoder = OrdinalEncoder(dtype=np.int64)
    encoder.fit(all_cat)

    X_cat_train = encoder.transform(train_df[cat_cols].astype(str))
    X_cat_val = encoder.transform(val_df[cat_cols].astype(str))
    X_cat_test = encoder.transform(test_df[cat_cols].astype(str))

    vocab_sizes = [int(all_cat[col].nunique()) for col in cat_cols]

    # Continuous Normalization (Fit on Train only)
    scaler = StandardScaler()
    scaler.fit(train_df[cont_cols])

    X_cont_train = scaler.transform(train_df[cont_cols])
    X_cont_val = scaler.transform(val_df[cont_cols])
    X_cont_test = scaler.transform(test_df[cont_cols])

    # Targets and IDs
    y_train = train_df["target"].values
    y_val = val_df["target"].values
    test_ids = test_df["id"].values

    # Cache Results
    np.savez(
        cache_file,
        X_cat_train=X_cat_train,
        X_cont_train=X_cont_train,
        y_train=y_train,
        X_cat_val=X_cat_val,
        X_cont_val=X_cont_val,
        y_val=y_val,
        X_cat_test=X_cat_test,
        X_cont_test=X_cont_test,
        test_ids=test_ids,
    )
    np.save(vocab_file, np.array(vocab_sizes))

    return (
        torch.tensor(X_cat_train, dtype=torch.long),
        torch.tensor(X_cont_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(X_cat_val, dtype=torch.long),
        torch.tensor(X_cont_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
        torch.tensor(X_cat_test, dtype=torch.long),
        torch.tensor(X_cont_test, dtype=torch.float32),
        torch.tensor(test_ids, dtype=torch.long),
        vocab_sizes,
    )


# ==========================================
# 3. Model Architecture (CR-HPE)
# ==========================================


class CRHPEModel(nn.Module):
    def __init__(self, vocab_sizes, num_cont):
        super(CRHPEModel, self).__init__()
        self.num_streams = 5
        self.emb_dim = 16

        # Independent Embeddings per stream: List[List[Embedding]]
        # Outer list: Stream index
        # Inner list: Feature index
        self.embeddings = nn.ModuleList(
            [
                nn.ModuleList([nn.Embedding(v, self.emb_dim) for v in vocab_sizes])
                for _ in range(self.num_streams)
            ]
        )

        # Input dimension for MLP = (Num Cat * Emb Dim) + Num Cont
        input_dim = len(vocab_sizes) * self.emb_dim + num_cont

        # Deep Paths (Funnels)
        self.mlps = nn.ModuleList()
        for i in range(self.num_streams):
            if i < 3:  # Streams 1, 2, 3 (Standard Capacity)
                layers = [
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.20),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.20),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.20),
                    nn.Linear(128, 1),
                ]
            else:  # Streams 4, 5 (High Capacity)
                layers = [
                    nn.Linear(input_dim, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(256, 1),
                ]
            self.mlps.append(nn.Sequential(*layers))

        # Linear Paths (Continuous Skip Connection)
        # Projects only continuous features to output
        self.linears = nn.ModuleList(
            [nn.Linear(num_cont, 1) for _ in range(self.num_streams)]
        )

    def forward(self, x_cat, x_cont):
        outputs = []
        for i in range(self.num_streams):
            # 1. Embeddings
            # Retrieve embeddings for this stream
            embs = [emb(x_cat[:, j]) for j, emb in enumerate(self.embeddings[i])]
            x_emb = torch.cat(embs, dim=1)

            # 2. Early Fusion
            x_fused = torch.cat([x_emb, x_cont], dim=1)

            # 3. Deep Path
            deep_out = self.mlps[i](x_fused)

            # 4. Linear Path (Residual on Continuous)
            linear_out = self.linears[i](x_cont)

            # 5. Aggregation
            outputs.append(deep_out + linear_out)

        return outputs


# ==========================================
# 4. Training Loop & Execution
# ==========================================


def train_and_predict(load_cached_data=True, epochs=50, batch_size=1024):
    seed_everything(42)
    device = get_device()

    # Load Data
    data = process_data(load_cached_data)
    X_cat_train, X_cont_train, y_train = data[0], data[1], data[2]
    X_cat_val, X_cont_val, y_val = data[3], data[4], data[5]
    X_cat_test, X_cont_test, test_ids = data[6], data[7], data[8]
    vocab_sizes = data[9]

    # Create DataLoaders
    train_dataset = TensorDataset(X_cat_train, X_cont_train, y_train)
    val_dataset = TensorDataset(X_cat_val, X_cont_val, y_val)
    test_dataset = TensorDataset(X_cat_test, X_cont_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize Model
    model = CRHPEModel(vocab_sizes, X_cont_train.shape[1]).to(device)

    # Optimization
    optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=2e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-2,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_model_path = "./working/best_model.pth"

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for cat_batch, cont_batch, y_batch in train_loader:
            cat_batch = cat_batch.to(device)
            cont_batch = cont_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()

            # Forward pass returns list of outputs from 5 streams
            outputs = model(cat_batch, cont_batch)

            # Sum loss across all streams
            loss = 0
            for out in outputs:
                loss += criterion(out, y_batch)

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
            for cat_batch, cont_batch, y_batch in val_loader:
                cat_batch = cat_batch.to(device)
                cont_batch = cont_batch.to(device)
                outputs = model(cat_batch, cont_batch)

                # Ensemble averaging (Sigmoid -> Mean)
                probs = torch.zeros_like(outputs[0])
                for out in outputs:
                    probs += torch.sigmoid(out)
                probs /= len(outputs)

                val_preds.append(probs.cpu().numpy())
                val_targets.append(y_batch.numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        auc = roc_auc_score(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.6f} | Val AUC: {auc:.10f}"
        )

        # Checkpointing
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best Validation AUC: {best_auc:.10f}")

    # Inference on Test Set
    print("Generating predictions on Test Set...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    test_preds = []

    with torch.no_grad():
        for cat_batch, cont_batch in test_loader:
            cat_batch = cat_batch.to(device)
            cont_batch = cont_batch.to(device)
            outputs = model(cat_batch, cont_batch)

            probs = torch.zeros_like(outputs[0])
            for out in outputs:
                probs += torch.sigmoid(out)
            probs /= len(outputs)

            test_preds.append(probs.cpu().numpy())

    test_preds = np.concatenate(test_preds).flatten()

    # Save Submission
    os.makedirs("./submission", exist_ok=True)
    sub_df = pd.DataFrame({"id": test_ids, "target": test_preds})
    sub_df.to_csv("./submission/submission.csv", index=False)
    print("Submission saved to ./submission/submission.csv")
