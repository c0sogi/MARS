import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import random
import gc


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
class Config:
    # Reproducibility
    SEED = 42

    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_31"
    CACHE_PATH = os.path.join(WORKING_DIR, "processed_data.npz")
    SUBMISSION_PATH = "./submission/submission.csv"
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Data
    NUM_FEATURES = 30
    SEQ_LEN = 10
    VOCAB_SIZE = 26 + 1  # A-Z + padding/unknown

    # Model Architecture
    EMBED_DIM = 32
    BACKBONE_STAGES = [512, 256, 128]
    BLOCKS_PER_STAGE = 3
    DROPOUT_TRANSFORMER = 0.1
    DROPOUT_BACKBONE = 0.35
    DROPOUT_HEAD = 0.5
    MSD_HEADS = 5
    STOCHASTIC_DEPTH_MAX = 0.2

    # Training
    BATCH_SIZE = 1024
    EPOCHS = 40
    LR = 1e-3
    WD_WEIGHTS = 1e-2
    WD_BIAS_NORM = 0.0
    PATIENCE = 5


# ------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------------------
# Data Processing
# ------------------------------------------------------------------------------
def process_data(config, load_cached_data=True):
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(config.CACHE_PATH):
        print(f"Loading cached data from {config.CACHE_PATH}")
        data = np.load(config.CACHE_PATH, allow_pickle=True)
        return (
            data["X_train_seq"],
            data["X_train_cont"],
            data["y_train"],
            data["X_val_seq"],
            data["X_val_cont"],
            data["y_val"],
            data["X_test_seq"],
            data["X_test_cont"],
            data["test_ids"],
        )

    print("Processing data from scratch...")

    # Load Metadata
    train_meta = pd.read_csv(os.path.join(config.METADATA_DIR, "train_metadata.csv"))
    val_meta = pd.read_csv(os.path.join(config.METADATA_DIR, "val_metadata.csv"))
    test_meta = pd.read_csv(os.path.join(config.METADATA_DIR, "test_metadata.csv"))

    # Load Raw Data
    df_train_raw = pd.read_csv(os.path.join(config.INPUT_DIR, "train.csv"))
    df_test_raw = pd.read_csv(os.path.join(config.INPUT_DIR, "test.csv"))

    # Helper to extract features
    def extract_features(meta_df, raw_df, is_test=False):
        # Merge to get features for the specific split
        # Cite debug_lesson_1: Prevent column collision for 'target'
        df = meta_df.merge(
            raw_df.drop(columns=["target"], errors="ignore"), on="id", how="left"
        )

        # Sequence Feature (f_27)
        # Map A=1, B=2, ...
        seq_data = np.zeros((len(df), config.SEQ_LEN), dtype=np.int32)
        # Handle potential NaNs or non-strings if any (though dataset is clean)
        f27_values = df["f_27"].fillna("A" * config.SEQ_LEN).values

        for i, s in enumerate(f27_values):
            s_trunc = str(s)[: config.SEQ_LEN]
            for j, char in enumerate(s_trunc):
                val = ord(char) - ord("A") + 1
                if val < 1 or val > 26:
                    val = 0  # Safety
                seq_data[i, j] = val

        # Continuous Features
        cont_cols = [
            f"f_{i:02d}" for i in range(31) if i != 27
        ]  # f_00 to f_30 excluding f_27
        cont_data = df[cont_cols].values.astype(np.float32)

        if is_test:
            return seq_data, cont_data, df["id"].values
        else:
            return seq_data, cont_data, df["target"].values

    # Extract
    X_train_seq, X_train_cont, y_train = extract_features(train_meta, df_train_raw)
    X_val_seq, X_val_cont, y_val = extract_features(val_meta, df_train_raw)
    X_test_seq, X_test_cont, test_ids = extract_features(
        test_meta, df_test_raw, is_test=True
    )

    # Normalize Continuous Features
    scaler = StandardScaler()
    X_train_cont = scaler.fit_transform(X_train_cont)
    X_val_cont = scaler.transform(X_val_cont)
    X_test_cont = scaler.transform(X_test_cont)

    # Cache
    np.savez(
        config.CACHE_PATH,
        X_train_seq=X_train_seq,
        X_train_cont=X_train_cont,
        y_train=y_train,
        X_val_seq=X_val_seq,
        X_val_cont=X_val_cont,
        y_val=y_val,
        X_test_seq=X_test_seq,
        X_test_cont=X_test_cont,
        test_ids=test_ids,
    )

    return (
        X_train_seq,
        X_train_cont,
        y_train,
        X_val_seq,
        X_val_cont,
        y_val,
        X_test_seq,
        X_test_cont,
        test_ids,
    )


class ManufacturingDataset(Dataset):
    def __init__(self, seq, cont, target=None):
        self.seq = torch.tensor(seq, dtype=torch.long)
        self.cont = torch.tensor(cont, dtype=torch.float32)
        self.target = (
            torch.tensor(target, dtype=torch.float32) if target is not None else None
        )

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        if self.target is not None:
            return self.seq[idx], self.cont[idx], self.target[idx]
        else:
            return self.seq[idx], self.cont[idx]


# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Linear map to 2*dim for splitting (gate and value)
        self.proj = nn.Linear(dim, 2 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        x_proj = self.proj(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        return self.out_proj(F.silu(x1) * x2)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ResidualBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.swiglu = SwiGLU(dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        # Pre-Norm -> SwiGLU -> DropPath -> Add
        return x + self.drop_path(self.swiglu(self.norm(x)))


class PostNormTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        # Init pos embed with low variance noise
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,  # Post-Norm
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        # x: [B, L]
        x = self.embedding(x)  # [B, L, D]
        x = x + self.pos_embed
        x = self.encoder(x)
        return x.flatten(1)  # [B, L*D]


class HybridSwiGLUModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Stream 1: Sequence
        self.transformer = PostNormTransformer(
            config.VOCAB_SIZE,
            config.EMBED_DIM,
            config.SEQ_LEN,
            config.DROPOUT_TRANSFORMER,
        )
        seq_out_dim = config.SEQ_LEN * config.EMBED_DIM

        # Stream 2: Continuous (Raw)
        cont_dim = config.NUM_FEATURES

        # Fusion
        fusion_dim = seq_out_dim + cont_dim
        self.stem = nn.Linear(fusion_dim, config.BACKBONE_STAGES[0])

        # Backbone
        layers = []
        in_dim = config.BACKBONE_STAGES[0]

        # Stochastic Depth Schedule
        total_blocks = len(config.BACKBONE_STAGES) * config.BLOCKS_PER_STAGE
        dp_rates = [
            x.item()
            for x in torch.linspace(0, config.STOCHASTIC_DEPTH_MAX, total_blocks)
        ]
        block_idx = 0

        for stage_dim in config.BACKBONE_STAGES:
            # Transition (if dimension changes)
            if in_dim != stage_dim:
                layers.append(nn.LayerNorm(in_dim))
                layers.append(nn.Linear(in_dim, stage_dim))
                in_dim = stage_dim

            # Blocks
            for _ in range(config.BLOCKS_PER_STAGE):
                layers.append(ResidualBlock(stage_dim, drop_path=dp_rates[block_idx]))
                block_idx += 1

        self.backbone = nn.Sequential(*layers)
        self.backbone_dropout = nn.Dropout(config.DROPOUT_BACKBONE)

        # Head (Multi-Sample Dropout)
        final_dim = config.BACKBONE_STAGES[-1]
        self.head_drop = nn.Dropout(config.DROPOUT_HEAD)
        self.head_fc = nn.Linear(final_dim, 1)
        self.msd_heads = config.MSD_HEADS

        # Init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, seq, cont):
        # Stream 1
        seq_feat = self.transformer(seq)

        # Fusion
        x = torch.cat([seq_feat, cont], dim=1)
        x = self.stem(x)

        # Backbone
        x = self.backbone(x)
        x = self.backbone_dropout(x)

        # MSD Head
        if self.training:
            logits = []
            for _ in range(self.msd_heads):
                logits.append(self.head_fc(self.head_drop(x)))
            return torch.stack(logits, dim=0).squeeze(-1)  # [Heads, Batch]
        else:
            # Inference: Dropout is off, so just one pass
            return self.head_fc(x).squeeze(-1)


# ------------------------------------------------------------------------------
# Training Logic
# ------------------------------------------------------------------------------
def train_model():
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    data = process_data(Config)
    (
        X_train_seq,
        X_train_cont,
        y_train,
        X_val_seq,
        X_val_cont,
        y_val,
        X_test_seq,
        X_test_cont,
        test_ids,
    ) = data

    train_ds = ManufacturingDataset(X_train_seq, X_train_cont, y_train)
    val_ds = ManufacturingDataset(X_val_seq, X_val_cont, y_val)
    test_ds = ManufacturingDataset(X_test_seq, X_test_cont)

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

    # Model
    model = HybridSwiGLUModel(Config).to(device)

    # Optimizer
    param_groups = [
        {"params": [], "weight_decay": Config.WD_WEIGHTS},
        {"params": [], "weight_decay": Config.WD_BIAS_NORM},
    ]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name or "norm" in name or "pos_embed" in name:
            param_groups[1]["params"].append(param)
        else:
            param_groups[0]["params"].append(param)

    optimizer = torch.optim.AdamW(param_groups, lr=Config.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.BCEWithLogitsLoss()

    # Loop
    best_auc = 0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0

        for seq, cont, target in train_loader:
            seq, cont, target = seq.to(device), cont.to(device), target.to(device)

            optimizer.zero_grad()
            logits = model(seq, cont)  # [Heads, Batch]

            # MSD Loss: Mean over heads
            loss = 0
            for i in range(Config.MSD_HEADS):
                loss += criterion(logits[i], target)
            loss /= Config.MSD_HEADS

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for seq, cont, target in val_loader:
                seq, cont, target = seq.to(device), cont.to(device), target.to(device)
                logit = model(seq, cont)
                val_preds.append(torch.sigmoid(logit).cpu().numpy())
                val_targets.append(target.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_auc = roc_auc_score(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.6f}"
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

    # Inference
    print("Loading best model for inference...")
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH))

    model.eval()
    test_preds = []
    with torch.no_grad():
        for seq, cont in test_loader:
            seq, cont = seq.to(device), cont.to(device)
            logit = model(seq, cont)
            test_preds.append(torch.sigmoid(logit).cpu().numpy())

    test_preds = np.concatenate(test_preds)

    # Submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, "target": test_preds})
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


# Execute
# train_model()
