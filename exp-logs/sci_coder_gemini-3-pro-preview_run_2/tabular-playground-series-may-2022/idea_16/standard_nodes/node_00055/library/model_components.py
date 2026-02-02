import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, compute_auc, save_checkpoint, load_checkpoint
from library.dataset import get_dataloaders

# --------------------------------------------------------------------------
# Neural Network Components
# --------------------------------------------------------------------------


class GatedFeedForward(nn.Module):
    """
    Gated Feed-Forward Network for the Transformer.
    Structure: UpProj -> GLU (Sigmoid) -> DownProj
    Strictly follows the expansion logic: UpProj to 8x, GLU reduces to 4x hidden.
    """

    def __init__(self, d_model, expansion_factor=8, dropout=0.0):
        super().__init__()
        # Input: d_model
        # UpProj Output: d_model * expansion_factor
        # GLU Output (Hidden): (d_model * expansion_factor) / 2
        # DownProj Input: GLU Output

        self.up_proj_dim = d_model * expansion_factor
        self.hidden_dim = self.up_proj_dim // 2

        self.up_proj = nn.Linear(d_model, self.up_proj_dim)
        self.down_proj = nn.Linear(self.hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [Batch, Seq, Dim]
        x = self.up_proj(x)
        # GLU: Split last dim in half, A * sigmoid(B)
        x = F.glu(x, dim=-1)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x


class GatedTransformerLayer(nn.Module):
    """
    Transformer Layer with Gated FFN and strictly Layer Normalization.
    Uses Pre-Norm configuration for stability.
    """

    def __init__(self, d_model, num_heads, dropout, ffn_expansion):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = GatedFeedForward(
            d_model, expansion_factor=ffn_expansion, dropout=dropout
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-Norm: x = x + Dropout(Sublayer(Norm(x)))

        # Self-Attention Block
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_out)

        # Gated FFN Block
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout2(ffn_out)

        return x


class DirectGLUBlock(nn.Module):
    """
    Direct GLU Residual Block for the Backbone.
    Formula: x_out = x + Dropout(GLU(Linear(BatchNorm(x))))
    Topology: Pre-Activation. Cite {solution_lesson_node_00054}
    """

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()

        # Pre-Activation BN
        self.bn = nn.BatchNorm1d(in_dim)

        # Main Path
        # Linear projects to 2 * out_dim for GLU
        self.linear = nn.Linear(in_dim, out_dim * 2)
        self.dropout = nn.Dropout(dropout)

        # Projected Residual Connection
        if in_dim != out_dim:
            self.project_res = nn.Linear(in_dim, out_dim)
        else:
            self.project_res = nn.Identity()

    def forward(self, x):
        # Pre-Activation
        x_norm = self.bn(x)

        # Main path
        out = self.linear(x_norm)
        out = F.glu(out, dim=-1)  # Reduces dim by half
        out = self.dropout(out)

        # Shortcut is added to the result
        return self.project_res(x) + out


class GatedTransformerResFunnelHybrid(nn.Module):
    """
    Hybrid Architecture:
    1. Gated Transformer Encoder for Categorical Sequences (f_27).
    2. Raw pass-through for Continuous Features.
    3. Late Fusion.
    4. ResFunnel-GLU Backbone.
    """

    def __init__(self):
        super().__init__()

        # --- Stream 1: Gated Transformer (Categorical) ---
        self.embed_dim = Config.EMBED_DIM
        self.seq_len = Config.SEQ_LEN

        self.embedding = nn.Embedding(Config.VOCAB_SIZE, self.embed_dim)
        # Learnable Positional Embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, self.embed_dim))

        self.transformer_blocks = nn.ModuleList(
            [
                GatedTransformerLayer(
                    d_model=self.embed_dim,
                    num_heads=Config.NUM_HEADS,
                    dropout=Config.TRANSFORMER_DROPOUT,
                    ffn_expansion=Config.FFN_EXPANSION_FACTOR,
                )
                for _ in range(Config.NUM_TRANSFORMER_LAYERS)
            ]
        )

        # Output dim after flattening: seq_len * embed_dim
        self.stream1_out_dim = self.seq_len * self.embed_dim

        # --- Stream 2: Continuous ---
        # 30 features
        self.num_cont = Config.NUM_CONT_FEATURES

        # --- Fusion ---
        self.fusion_dim = self.stream1_out_dim + self.num_cont

        # --- Backbone: ResFunnel-GLU ---
        stages = Config.BACKBONE_STAGES  # [512, 256, 128]
        dropout = Config.BACKBONE_DROPOUT

        self.backbone = nn.Sequential()

        current_dim = self.fusion_dim

        # Construct stages
        for i, stage_width in enumerate(stages):
            # Transition Block (handles dimension change if any)
            self.backbone.add_module(
                f"stage_{i}_block_1", DirectGLUBlock(current_dim, stage_width, dropout)
            )
            # Stacked Block (deepening the representation at same width)
            self.backbone.add_module(
                f"stage_{i}_block_2", DirectGLUBlock(stage_width, stage_width, dropout)
            )
            current_dim = stage_width

        self.final_dim = current_dim

        # Final BN for Pre-Activation Backbone
        self.final_bn = nn.BatchNorm1d(self.final_dim)

        # Output Head
        self.head = nn.Linear(self.final_dim, 1)

    def forward(self, cont_features, cat_features):
        # --- Stream 1: Categorical ---
        # cat_features: [B, 10]
        x_cat = self.embedding(cat_features)  # [B, 10, 32]
        x_cat = x_cat + self.pos_embedding  # Broadcast add

        for block in self.transformer_blocks:
            x_cat = block(x_cat)

        # Flatten
        x_cat = x_cat.flatten(start_dim=1)  # [B, 320]

        # --- Stream 2: Continuous ---
        x_cont = cont_features  # [B, 30]

        # --- Fusion ---
        x = torch.cat([x_cat, x_cont], dim=1)  # [B, 350]

        # --- Backbone ---
        x = self.backbone(x)
        x = self.final_bn(x)

        # --- Output ---
        # Return logits for numerical stability with BCEWithLogitsLoss
        logits = self.head(x)
        return logits


# --------------------------------------------------------------------------
# Training & Inference Logic
# --------------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for batch in loader:
        cont = batch["cont_features"].to(device)
        cat = batch["cat_features"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)

        optimizer.zero_grad()

        logits = model(cont, cat)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * cont.size(0)

        # Store for AUC
        all_targets.append(targets.detach().cpu().numpy())
        all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    epoch_auc = compute_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            cont = batch["cont_features"].to(device)
            cat = batch["cat_features"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)

            logits = model(cont, cat)
            loss = criterion(logits, targets)

            running_loss += loss.item() * cont.size(0)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(torch.sigmoid(logits).cpu().numpy())

    val_loss = running_loss / len(loader.dataset)

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    val_auc = compute_auc(all_targets, all_preds)

    return val_loss, val_auc


def train_model():
    """
    Executes the training pipeline.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Starting training on {device}...")

    # Data
    train_loader, val_loader, _, _ = get_dataloaders(load_cached_data=True)

    # Model
    model = GatedTransformerResFunnelHybrid().to(device)

    # Optimization
    # AdamW with high weight decay as per idea
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Aggressive Step Decay
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    criterion = nn.BCEWithLogitsLoss()

    # Loop
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch}/{Config.EPOCHS} | LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | Train AUC: {train_auc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        # Checkpoint based on AUC
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_auc, Config.MODEL_CHECKPOINT
            )
            print(f"  >>> New Best Model Saved! AUC: {val_auc:.6f}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc:.6f}")


def generate_submission():
    """
    Loads the best model and generates predictions for the test set.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print("Generating submission...")

    # Data
    _, _, test_loader, test_ids = get_dataloaders(load_cached_data=True)

    # Model
    model = GatedTransformerResFunnelHybrid().to(device)

    # Load Weights
    checkpoint = load_checkpoint(Config.MODEL_CHECKPOINT, model, device=device)
    print(
        f"Loaded model from epoch {checkpoint['epoch']} with Val AUC: {checkpoint['score']:.6f}"
    )

    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            cont = batch["cont_features"].to(device)
            cat = batch["cat_features"].to(device)

            logits = model(cont, cat)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()

    # Create DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "target": all_preds})

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")


def run():
    """
    Main entry point to run training and submission generation.
    """
    train_model()
    generate_submission()
