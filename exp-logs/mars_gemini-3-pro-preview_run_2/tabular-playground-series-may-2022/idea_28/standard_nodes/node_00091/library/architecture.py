import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.dataset import get_dataloaders

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit.
    Splits input into two parts (x, g), applies SiLU to g, and returns x * SiLU(g).
    """

    def forward(self, x):
        x, g = x.chunk(2, dim=-1)
        return x * F.silu(g)


class ResFunnelBlock(nn.Module):
    """
    Pre-Norm Residual Block with SwiGLU activation, Dropout, and Stochastic Depth.
    Structure: x + DropPath(Dropout(SwiGLU(Linear(LayerNorm(x)))))
    The internal Linear layer expands dimension by 2x to accommodate GLU splitting.
    """

    def __init__(self, dim, drop_path=0.0, dropout=0.35):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # Expand to 2x dim because SwiGLU will halve it back to dim
        self.linear = nn.Linear(dim, dim * 2)
        self.swiglu = SwiGLU()
        self.dropout = nn.Dropout(dropout)
        self.drop_path_rate = drop_path

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.linear(x)
        x = self.swiglu(x)
        x = self.dropout(x)

        # Stochastic Depth (DropPath)
        if self.drop_path_rate > 0.0 and self.training:
            keep_prob = 1.0 - self.drop_path_rate
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask = x.new_empty(shape).bernoulli_(keep_prob)
            x = x.div(keep_prob) * mask

        return shortcut + x


class SwishGatedResFunnel(nn.Module):
    def __init__(self):
        super().__init__()

        # --- Stream 1: Categorical Sequence (Transformer) ---
        # f_27 has characters 'A'-'Z' (mapped to 0-25)
        self.char_embed = nn.Embedding(26, 32)

        # Learnable Positional Embedding: initialized with low variance noise
        self.pos_embed = nn.Parameter(torch.randn(1, 10, 32) * 0.02)

        # Standard Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # --- Fusion ---
        # Transformer output: 10 tokens * 32 dim = 320
        # Continuous input: 30 features
        # Total fusion input: 350
        self.stem = nn.Linear(350, 512)

        # --- Backbone: Swish-Gated ResFunnel ---
        # Configuration: 3 Stages (512 -> 256 -> 128), 3 blocks each.

        # Stochastic Depth Schedule: Linear 0.0 to 0.2 over 9 blocks
        dpr = [x.item() for x in torch.linspace(0, 0.2, 9)]

        # Stage 1: 512 dim
        self.stage1_blocks = nn.ModuleList(
            [ResFunnelBlock(512, drop_path=dpr[i]) for i in range(3)]
        )
        self.trans1 = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 256))

        # Stage 2: 256 dim
        self.stage2_blocks = nn.ModuleList(
            [ResFunnelBlock(256, drop_path=dpr[3 + i]) for i in range(3)]
        )
        self.trans2 = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 128))

        # Stage 3: 128 dim
        self.stage3_blocks = nn.ModuleList(
            [ResFunnelBlock(128, drop_path=dpr[6 + i]) for i in range(3)]
        )

        # --- Output Head ---
        self.head = nn.Linear(128, 1)

        self._init_weights()

    def _init_weights(self):
        # Apply Kaiming Uniform to SwiGLU blocks and Linear layers
        # Apply Xavier to Transformer

        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "transformer" in name:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    # Cite Lesson 82: Avoid ReLU-Targeted Initialization for GLU. Prefer Xavier.
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, num_x, cat_x):
        # num_x: (B, 30)
        # cat_x: (B, 10)

        # Stream 1
        emb = self.char_embed(cat_x)  # (B, 10, 32)
        emb = emb + self.pos_embed
        seq = self.transformer(emb)  # (B, 10, 32)
        seq_flat = seq.reshape(seq.size(0), -1)  # (B, 320)

        # Stream 2 & Fusion
        concat = torch.cat([seq_flat, num_x], dim=1)  # (B, 350)
        x = self.stem(concat)  # (B, 512)

        # Backbone
        for blk in self.stage1_blocks:
            x = blk(x)
        x = self.trans1(x)

        for blk in self.stage2_blocks:
            x = blk(x)
        x = self.trans2(x)

        for blk in self.stage3_blocks:
            x = blk(x)

        # Head
        logits = self.head(x)
        return torch.sigmoid(logits)


# ------------------------------------------------------------------------------
# Training & Execution
# ------------------------------------------------------------------------------


def train_and_predict():
    print("Initializing Swish-Gated ResFunnel Pipeline...")

    # 1. Data Loading
    batch_size = 1024
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=batch_size, num_workers=4, load_cached_data=True
    )

    # 2. Model Setup
    model = SwishGatedResFunnel().to(DEVICE)

    # 3. Parameter-Grouped Optimization (Decoupled Weight Decay)
    # Group 1: Decay 1e-2 (Weights of Linear, Embedding, Attention)
    # Group 2: Decay 0.0 (Biases, LayerNorm, Positional Embeddings)
    param_groups = [
        {"params": [], "weight_decay": 1e-2},
        {"params": [], "weight_decay": 0.0},
    ]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check for parameters that should not decay
        if (
            param.ndim <= 1
            or name.endswith(".bias")
            or "norm" in name
            or "pos_embed" in name
        ):
            param_groups[1]["params"].append(param)
        else:
            param_groups[0]["params"].append(param)

    optimizer = optim.AdamW(param_groups, lr=1e-3)

    # Scheduler: Aggressive Step Decay (factor 0.1 every 10 epochs)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    criterion = nn.BCELoss()

    # 4. Training Loop
    epochs = 40
    best_auc = 0.0
    best_model_path = "./working/best_model.pth"
    os.makedirs("./working", exist_ok=True)

    print(f"Starting training for {epochs} epochs on {DEVICE}...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            num_x = batch["numerical"].to(DEVICE)
            cat_x = batch["categorical"].to(DEVICE)
            target = batch["target"].to(DEVICE)

            optimizer.zero_grad()
            output = model(num_x, cat_x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * num_x.size(0)

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                num_x = batch["numerical"].to(DEVICE)
                cat_x = batch["categorical"].to(DEVICE)
                target = batch["target"]

                output = model(num_x, cat_x)
                val_preds.append(output.cpu().numpy())
                val_targets.append(target.numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)

        val_auc = roc_auc_score(val_targets, val_preds)

        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val AUC = {val_auc}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best Val AUC: {best_auc}")

    # 5. Inference
    print("Starting inference on test set...")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            num_x = batch["numerical"].to(DEVICE)
            cat_x = batch["categorical"].to(DEVICE)
            output = model(num_x, cat_x)
            test_preds.append(output.cpu().numpy())

    test_preds = np.concatenate(test_preds).flatten()

    # 6. Submission
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    df_sub = pd.DataFrame({"id": test_ids, "target": test_preds})

    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
