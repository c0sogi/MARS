import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm.layers
from library.utils import seed_everything, compute_auc, get_optimizer_params
from library.dataset import get_dataloaders

# ------------------------------------------------------------------------------
# Model Components
# ------------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit.
    Expects input dimension to be 2 * hidden_dim.
    Splits input into two halves: gate and value.
    Returns: Swish(gate) * value.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class ResFunnelBlock(nn.Module):
    """
    Pre-LayerNorm Direct SwiGLU Residual Block.
    x_out = x + DropPath(Dropout(SwiGLU(Linear(LayerNorm(x)))))
    """

    def __init__(self, dim, dropout=0.35, drop_path=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, 2 * dim)
        self.swiglu = SwiGLU()
        self.dropout = nn.Dropout(dropout)
        self.drop_path = (
            timm.layers.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, x):
        input_x = x
        x = self.norm(x)
        x = self.linear(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = self.drop_path(x)
        return input_x + x


class HybridSwiGLUResFunnel(nn.Module):
    def __init__(self):
        super().__init__()

        # --- Stream 1: Categorical (Transformer) ---
        # Cite solution_lesson_node_00125: Prefer pure Self-Attention over Conformer
        self.vocab_size = 32
        self.emb_dim = 32
        self.seq_len = 10

        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.emb_dim))

        # Standard Transformer Encoder
        # Cite solution_lesson_node_00095: Post-Norm for shallow networks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=4,
            dim_feedforward=self.emb_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # --- Stream 2: Continuous ---
        self.num_cont_features = 30

        # --- Fusion ---
        # Flattened Transformer (10 * 32 = 320) + Continuous (30) = 350
        self.fusion_input_dim = (self.seq_len * self.emb_dim) + self.num_cont_features
        self.backbone_dim = 512
        self.stem = nn.Linear(self.fusion_input_dim, self.backbone_dim)

        # --- Backbone ---
        # Stages: 512 -> 256 -> 128
        # Cite solution_lesson_node_00075: Moderate depth (2 blocks per stage)
        dims = [512, 256, 128]
        depths = [2, 2, 2]
        total_blocks = sum(depths)

        # Linear Stochastic Depth Schedule 0.0 -> 0.2
        dpr = [x.item() for x in torch.linspace(0, 0.2, total_blocks)]

        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()

        curr_dim = dims[0]
        block_idx = 0

        # Build Stages
        for stage_idx in range(3):
            # Blocks
            blocks = []
            for _ in range(depths[stage_idx]):
                blocks.append(
                    ResFunnelBlock(curr_dim, dropout=0.35, drop_path=dpr[block_idx])
                )
                block_idx += 1
            self.stages.append(nn.Sequential(*blocks))

            # Transition (if not last stage)
            if stage_idx < 2:
                next_dim = dims[stage_idx + 1]
                # Cite solution_lesson_node_00088: Pre-Norm Transition
                self.transitions.append(
                    nn.Sequential(nn.LayerNorm(curr_dim), nn.Linear(curr_dim, next_dim))
                )
                curr_dim = next_dim

        # --- Head ---
        self.head = nn.Linear(curr_dim, 1)

        # --- Initialization ---
        self.apply(self._init_weights)
        # Explicit Pos Embed Init (Low Variance) Cite solution_lesson_node_00081
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Kaiming Uniform for SwiGLU/Stem
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            # Unit Variance for Embeddings Cite solution_lesson_node_00114
            nn.init.normal_(m.weight, mean=0.0, std=1.0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.MultiheadAttention):
            # Xavier for Attention Cite solution_lesson_node_00072
            if m.in_proj_weight is not None:
                nn.init.xavier_uniform_(m.in_proj_weight)
            if m.out_proj.weight is not None:
                nn.init.xavier_uniform_(m.out_proj.weight)
            if m.in_proj_bias is not None:
                nn.init.constant_(m.in_proj_bias, 0)
            if m.out_proj.bias is not None:
                nn.init.constant_(m.out_proj.bias, 0)

    def forward(self, continuous, categorical):
        # Stream 1: Categorical
        # categorical: (N, 10)
        x_cat = self.embedding(categorical)  # (N, 10, 32)
        x_cat = x_cat + self.pos_embed

        x_cat = self.transformer(x_cat)

        x_cat_flat = x_cat.flatten(1)  # (N, 320)

        # Stream 2: Continuous (N, 30) - Passed raw
        # Cite solution_lesson_node_00124: Do not mix normalization scopes (No LayerNorm here)

        # Fusion
        x = torch.cat([x_cat_flat, continuous], dim=1)  # (N, 350)
        x = self.stem(x)  # (N, 512)

        # Backbone
        # Stage 1
        x = self.stages[0](x)
        x = self.transitions[0](x)
        # Stage 2
        x = self.stages[1](x)
        x = self.transitions[1](x)
        # Stage 3
        x = self.stages[2](x)

        # Head
        x = self.head(x)
        return torch.sigmoid(x).squeeze(-1)


# ------------------------------------------------------------------------------
# Training Pipeline
# ------------------------------------------------------------------------------


def run_training(epochs=40, batch_size=1024, load_cached_data=True):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data Loading
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data, num_workers=4
    )

    # 2. Model Setup
    model = PostNormConformerSwiGLU().to(device)

    # 3. Optimizer (Decoupled Weight Decay)
    optimizer_params = get_optimizer_params(model, weight_decay=1e-2)
    optimizer = optim.AdamW(optimizer_params, lr=1e-3)

    # 4. Scheduler (Step Decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    criterion = nn.BCELoss()

    best_auc = 0.0
    best_model_path = "./working/best_model.pth"
    os.makedirs("./working", exist_ok=True)

    # 5. Training Loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            cont = batch["continuous"].to(device)
            cat = batch["categorical"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()
            pred = model(cont, cat)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                cont = batch["continuous"].to(device)
                cat = batch["categorical"].to(device)
                target = batch["target"].to(device)

                pred = model(cont, cat)
                loss = criterion(pred, target)
                val_loss += loss.item()

                val_preds.append(pred.cpu().numpy())
                val_targets.append(target.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        avg_val_loss = val_loss / len(val_loader)
        auc = compute_auc(val_targets, val_preds)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val AUC: {auc:.10f}"
        )

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), best_model_path)

        scheduler.step()

    print(f"Training complete. Best Validation AUC: {best_auc:.10f}")

    # 6. Inference
    print("Starting inference on test set...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            cont = batch["continuous"].to(device)
            cat = batch["categorical"].to(device)

            pred = model(cont, cat)
            test_preds.append(pred.cpu().numpy())

    test_preds = np.concatenate(test_preds)

    # 7. Submission
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    df_sub = pd.DataFrame({"id": test_ids, "target": test_preds})

    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    run_training()
