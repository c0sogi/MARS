import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import random
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


class Config:
    # Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    CACHE_DIR = "./working/idea_29"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data
    SEED = 42
    NUM_WORKERS = 4

    # Model Hyperparameters
    EMBED_DIM = 32
    SEQ_LEN = 11  # 10 chars + 1 state token
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_HEADS = 4
    TRANSFORMER_DROPOUT = 0.1

    BACKBONE_STAGES = [512, 256, 128]
    BLOCKS_PER_STAGE = 3
    BACKBONE_DROPOUT = 0.35
    STOCHASTIC_DEPTH_MAX = 0.2

    # Training Hyperparameters
    BATCH_SIZE = 1024
    EPOCHS = 40
    LR = 1e-3
    WEIGHT_DECAY_GROUP1 = 1e-2
    WEIGHT_DECAY_GROUP2 = 0.0  # Bias, LayerNorm, PosEmbed
    LR_STEP_SIZE = 10
    LR_GAMMA = 0.1

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ManufacturingDataset(Dataset):
    def __init__(self, X_cont, X_cat, y=None):
        self.X_cont = torch.FloatTensor(X_cont)
        self.X_cat = torch.LongTensor(X_cat)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X_cont)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_cont[idx], self.X_cat[idx], self.y[idx]
        else:
            return self.X_cont[idx], self.X_cat[idx]


def load_and_process_data(load_cached_data=True):
    """
    Loads data using metadata, processes features, and caches the result.
    """
    cache_file = os.path.join(Config.CACHE_DIR, "processed_data.npz")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return (
            data["X_train_cont"],
            data["X_train_cat"],
            data["y_train"],
            data["X_val_cont"],
            data["X_val_cat"],
            data["y_val"],
            data["X_test_cont"],
            data["X_test_cat"],
            data["test_ids"],
        )

    print("Processing data from scratch...")

    # Load Metadata
    train_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "train_metadata.csv"))
    val_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "val_metadata.csv"))
    test_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "test_metadata.csv"))

    # Load Raw Data
    df_train_raw = pd.read_csv(os.path.join(Config.INPUT_DIR, "train.csv"))
    df_test_raw = pd.read_csv(os.path.join(Config.INPUT_DIR, "test.csv"))

    # Map raw data to splits using ID
    # Create a lookup dictionary for speed
    train_dict = df_train_raw.set_index("id").T.to_dict()
    test_dict = df_test_raw.set_index("id").T.to_dict()

    def get_data_from_ids(meta_df, data_dict, is_test=False):
        ids = meta_df["id"].values
        # Extract rows efficiently
        rows = [data_dict[i] for i in ids]
        df = pd.DataFrame(rows)
        df["id"] = ids  # Ensure ID column is preserved/restored correctly

        if not is_test:
            # Ensure target is correct from metadata (though it should match raw)
            df["target"] = meta_df["target"].values

        return df

    # Reconstruct DataFrames based on metadata splits
    # Note: Using pandas merge is safer and often faster than dict lookup for large DFs
    df_train = df_train_raw[df_train_raw["id"].isin(train_meta["id"])].copy()
    # Ensure order matches metadata
    df_train = df_train.set_index("id").reindex(train_meta["id"]).reset_index()

    df_val = df_train_raw[df_train_raw["id"].isin(val_meta["id"])].copy()
    df_val = df_val.set_index("id").reindex(val_meta["id"]).reset_index()

    df_test = df_test_raw[df_test_raw["id"].isin(test_meta["id"])].copy()
    df_test = df_test.set_index("id").reindex(test_meta["id"]).reset_index()

    # Feature Engineering
    # 1. Continuous Features
    cont_cols = [f"f_{i:02d}" for i in range(31) if i != 27]
    scaler = StandardScaler()
    X_train_cont = scaler.fit_transform(df_train[cont_cols].values)
    X_val_cont = scaler.transform(df_val[cont_cols].values)
    X_test_cont = scaler.transform(df_test[cont_cols].values)

    # 2. Categorical Feature (f_27)
    # Decompose into 10 characters
    def process_f27(df):
        # Split string into list of chars
        chars = df["f_27"].apply(lambda x: list(x))
        # Convert to dataframe of columns
        char_df = pd.DataFrame(chars.tolist(), index=df.index)
        return char_df

    train_chars = process_f27(df_train)
    val_chars = process_f27(df_val)
    test_chars = process_f27(df_test)

    # Ordinal Encode
    # We assume standard uppercase alphabet.
    # To be safe, we fit an encoder on all unique chars found in train
    # Or simply map A->0, B->1 manually if we know the domain.
    # Let's use LabelEncoder per position or global? Usually global for letters.
    # The problem description implies standard chars. Let's use a fixed map for consistency.
    import string

    char_map = {c: i for i, c in enumerate(string.ascii_uppercase)}
    # Add a fallback for unknown just in case, though unlikely

    def encode_chars(char_df):
        # Apply map
        encoded = char_df.applymap(lambda x: char_map.get(x, 0)).values
        return encoded.astype(np.int64)

    X_train_cat = encode_chars(train_chars)
    X_val_cat = encode_chars(val_chars)
    X_test_cat = encode_chars(test_chars)

    # Targets
    y_train = df_train["target"].values.astype(np.float32)
    y_val = df_val["target"].values.astype(np.float32)
    test_ids = df_test["id"].values

    # Cache
    np.savez(
        cache_file,
        X_train_cont=X_train_cont,
        X_train_cat=X_train_cat,
        y_train=y_train,
        X_val_cont=X_val_cont,
        X_val_cat=X_val_cat,
        y_val=y_val,
        X_test_cont=X_test_cont,
        X_test_cat=X_test_cat,
        test_ids=test_ids,
    )

    print(f"Data processed and saved to {cache_file}")
    return (
        X_train_cont,
        X_train_cat,
        y_train,
        X_val_cont,
        X_val_cat,
        y_val,
        X_test_cont,
        X_test_cat,
        test_ids,
    )


# --- Model Components ---


class SwiGLUBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Linear projects to 2*dim for GLU splitting
        self.linear = nn.Linear(dim, dim * 2)
        self.dropout = nn.Dropout(Config.BACKBONE_DROPOUT)
        self.drop_path = drop_path

    def forward(self, x):
        input_x = x
        x = self.norm(x)
        x = self.linear(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * F.silu(gate)  # SwiGLU activation
        x = self.dropout(x)

        # Stochastic Depth (DropPath)
        if self.training and self.drop_path > 0:
            keep_prob = 1 - self.drop_path
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(
                shape, dtype=x.dtype, device=x.device
            )
            random_tensor.floor_()
            x = x.div(keep_prob) * random_tensor

        return input_x + x


class ContextAwareSwishGatedResFunnel(nn.Module):
    def __init__(self):
        super().__init__()

        # --- Stream 1: Context-Aware Categorical ---
        self.char_embed = nn.Embedding(26 + 1, Config.EMBED_DIM)  # +1 for safety
        self.state_proj = nn.Linear(30, Config.EMBED_DIM)

        # Positional Embedding: 10 chars + 1 state token = 11
        self.pos_embed = nn.Parameter(
            torch.randn(1, Config.SEQ_LEN, Config.EMBED_DIM) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.EMBED_DIM,
            nhead=Config.TRANSFORMER_HEADS,
            dim_feedforward=Config.EMBED_DIM * 4,
            dropout=Config.TRANSFORMER_DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Usually better for convergence
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.TRANSFORMER_LAYERS
        )

        # Flattened output dim: 10 chars * 32
        self.flat_dim = 10 * Config.EMBED_DIM

        # --- Fusion ---
        # Input to fusion: Flattened Transformer (320) + Raw Continuous (30)
        fusion_input_dim = self.flat_dim + 30
        self.stem = nn.Linear(fusion_input_dim, Config.BACKBONE_STAGES[0])

        # --- Backbone: Swish-Gated ResFunnel ---
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()

        # Stochastic depth schedule
        total_blocks = len(Config.BACKBONE_STAGES) * Config.BLOCKS_PER_STAGE
        dp_rates = [
            x.item()
            for x in torch.linspace(0, Config.STOCHASTIC_DEPTH_MAX, total_blocks)
        ]

        current_dim = Config.BACKBONE_STAGES[0]
        block_idx = 0

        for i, stage_dim in enumerate(Config.BACKBONE_STAGES):
            # Transition (Downsample/Project) if not first stage
            if i > 0:
                prev_dim = Config.BACKBONE_STAGES[i - 1]
                self.transitions.append(
                    nn.Sequential(
                        nn.LayerNorm(prev_dim), nn.Linear(prev_dim, stage_dim)
                    )
                )
            else:
                self.transitions.append(nn.Identity())

            # Blocks
            stage_blocks = nn.ModuleList()
            for _ in range(Config.BLOCKS_PER_STAGE):
                stage_blocks.append(
                    SwiGLUBlock(stage_dim, drop_path=dp_rates[block_idx])
                )
                block_idx += 1
            self.stages.append(stage_blocks)

        # --- Head ---
        self.final_norm = nn.LayerNorm(Config.BACKBONE_STAGES[-1])
        self.head = nn.Linear(Config.BACKBONE_STAGES[-1], 1)

        self._init_weights()

    def _init_weights(self):
        # Kaiming Uniform for SwiGLU/Linear
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Xavier for Transformer (override)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_cont, x_cat):
        # x_cont: (B, 30)
        # x_cat: (B, 10)

        B = x_cont.shape[0]

        # --- Stream 1 ---
        # Embed chars
        char_emb = self.char_embed(x_cat)  # (B, 10, 32)

        # Create State Token
        state_token = self.state_proj(x_cont).unsqueeze(1)  # (B, 1, 32)

        # Concat sequence: [State, Char1, ..., Char10]
        seq = torch.cat([state_token, char_emb], dim=1)  # (B, 11, 32)

        # Add Positional Embeddings
        seq = seq + self.pos_embed

        # Transformer
        seq_out = self.transformer(seq)  # (B, 11, 32)

        # Discard state token (index 0), keep chars (indices 1-10)
        char_out = seq_out[:, 1:, :]  # (B, 10, 32)

        # Flatten
        flat_chars = char_out.reshape(B, -1)  # (B, 320)

        # --- Fusion ---
        # Concat with raw continuous
        fused = torch.cat([flat_chars, x_cont], dim=1)  # (B, 350)
        x = self.stem(fused)

        # --- Backbone ---
        for i, stage in enumerate(self.stages):
            # Transition
            if not isinstance(self.transitions[i], nn.Identity):
                x = self.transitions[i](x)

            # Blocks
            for block in stage:
                x = block(x)

        x = self.final_norm(x)
        logits = self.head(x)
        return logits


# --- Training Loop ---


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for x_cont, x_cat, y in loader:
        x_cont, x_cat, y = x_cont.to(device), x_cat.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x_cont, x_cat).squeeze()
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_cont.size(0)

    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_cont, x_cat, y in loader:
            x_cont, x_cat, y = x_cont.to(device), x_cat.to(device), y.to(device)

            logits = model(x_cont, x_cat).squeeze()
            loss = criterion(logits, y)

            total_loss += loss.item() * x_cont.size(0)
            preds = torch.sigmoid(logits)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    auc = roc_auc_score(all_targets, all_preds)
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, auc


def run_training():
    set_seed(Config.SEED)

    # Data
    X_train_cont, X_train_cat, y_train, X_val_cont, X_val_cat, y_val, _, _, _ = (
        load_and_process_data()
    )

    train_dataset = ManufacturingDataset(X_train_cont, X_train_cat, y_train)
    val_dataset = ManufacturingDataset(X_val_cont, X_val_cat, y_val)

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

    # Model
    model = ContextAwareSwishGatedResFunnel().to(Config.DEVICE)

    # Optimizer Groups (Decoupled Weight Decay)
    param_groups = []
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "bias" in name or "norm" in name or "pos_embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": Config.WEIGHT_DECAY_GROUP1},
        {"params": no_decay_params, "weight_decay": Config.WEIGHT_DECAY_GROUP2},
    ]

    optimizer = optim.AdamW(param_groups, lr=Config.LR)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.LR_STEP_SIZE, gamma=Config.LR_GAMMA
    )
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0

    print(f"Starting training on {Config.DEVICE}...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE
        )
        val_loss, val_auc = validate(model, val_loader, criterion, Config.DEVICE)
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                model.state_dict(), os.path.join(Config.WORKING_DIR, "best_model.pth")
            )

    print(f"Training complete. Best Val AUC: {best_auc:.6f}")
    return best_auc


def generate_submission():
    set_seed(Config.SEED)

    # Load Data
    _, _, _, _, _, _, X_test_cont, X_test_cat, test_ids = load_and_process_data()

    test_dataset = ManufacturingDataset(X_test_cont, X_test_cat, None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Model
    model = ContextAwareSwishGatedResFunnel().to(Config.DEVICE)
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print("Model not found, training first...")
        run_training()

    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.eval()

    all_preds = []

    with torch.no_grad():
        for x_cont, x_cat in test_loader:
            x_cont, x_cat = x_cont.to(Config.DEVICE), x_cat.to(Config.DEVICE)
            logits = model(x_cont, x_cat).squeeze()
            preds = torch.sigmoid(logits)
            all_preds.append(preds.cpu().numpy())

    all_preds = np.concatenate(all_preds)

    # Save Submission
    submission = pd.DataFrame({"id": test_ids, "target": all_preds})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
