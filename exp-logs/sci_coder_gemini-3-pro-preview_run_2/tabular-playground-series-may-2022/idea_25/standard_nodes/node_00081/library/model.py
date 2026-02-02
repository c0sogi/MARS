import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

# Import from provided libraries
from library.config import Config
from library.data_loader import get_dataloaders
from library.utils import seed_everything, custom_weight_init

# ------------------------------------------------------------------------------
# Component Definitions
# ------------------------------------------------------------------------------


class SequenceEncoder(nn.Module):
    """
    Stream 1: Categorical Sequence Encoder using GELU-Transformer.
    Features:
    - Learnable Positional Embeddings initialized with Random Noise.
    - Standard Transformer Encoder with GELU activation and Low Dropout.
    """

    def __init__(self, vocab_size, embed_dim, seq_len, n_layers, n_heads, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Learnable Positional Embeddings initialized with Random Noise (Lesson 72)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, embed_dim))

        # Standard Transformer Encoder with GELU (Lesson 68)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,  # Standard configuration
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.flatten_dim = seq_len * embed_dim

    def forward(self, x):
        # x: (Batch, Seq_Len)
        x = self.embedding(x)  # (Batch, Seq_Len, Embed_Dim)
        x = x + self.pos_encoder  # Add random-init positional embeddings
        x = self.transformer(x)
        x = x.reshape(x.size(0), -1)  # Flatten
        return x


class GLUResidualBlock(nn.Module):
    """
    Pre-Activation Direct GLU Residual Block.
    Structure: x + Dropout(GLU(Linear(BatchNorm(x))))
    """

    def __init__(self, dim, dropout_rate):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        # Linear projects to dim*2 to accommodate GLU halving
        self.linear = nn.Linear(dim, dim * 2)
        self.glu = nn.GLU(dim=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.bn(x)
        out = self.linear(out)
        out = self.glu(out)
        out = self.dropout(out)
        return x + out


class ProjectedResidualTransition(nn.Module):
    """
    Transition layer between stages using Projected Residual Connections.
    Strictly preserves gradient flow when dimensions change.
    """

    def __init__(self, in_dim, out_dim, dropout_rate):
        super().__init__()
        # Projection for the residual shortcut
        self.project = nn.Linear(in_dim, out_dim)

        # Main processing path with dimension change
        self.bn = nn.BatchNorm1d(in_dim)
        self.linear = nn.Linear(in_dim, out_dim * 2)
        self.glu = nn.GLU(dim=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        shortcut = self.project(x)

        out = self.bn(x)
        out = self.linear(out)
        out = self.glu(out)
        out = self.dropout(out)

        return shortcut + out


class ResFunnelBackbone(nn.Module):
    """
    Balanced Process-Compress ResFunnel Backbone.
    Organized into stages with decreasing widths.
    Each stage has exactly 2 Residual Blocks.
    Transitions use Projected Residual Connections.
    """

    def __init__(self, stages, dropout_rate):
        super().__init__()
        layers = []
        current_dim = stages[0]

        # Stage 1 (Initial width)
        # Exactly 2 stacked Residual Blocks
        layers.append(GLUResidualBlock(current_dim, dropout_rate))
        layers.append(GLUResidualBlock(current_dim, dropout_rate))

        # Subsequent Stages
        for next_dim in stages[1:]:
            # Transition (Process-Compress)
            layers.append(
                ProjectedResidualTransition(current_dim, next_dim, dropout_rate)
            )
            current_dim = next_dim

            # Stage N (2 Blocks)
            layers.append(GLUResidualBlock(current_dim, dropout_rate))
            layers.append(GLUResidualBlock(current_dim, dropout_rate))

        self.net = nn.Sequential(*layers)
        self.out_dim = current_dim

    def forward(self, x):
        return self.net(x)


class BalancedProcessCompressHybrid(nn.Module):
    """
    The main architecture fusing categorical sequence and continuous streams.
    """

    def __init__(
        self,
        num_continuous,
        cat_seq_len,
        vocab_size,
        embed_dim=32,
        transformer_layers=2,
        transformer_heads=4,
        backbone_stages=[512, 256, 128],
        dropout_transformer=0.1,
        dropout_backbone=0.35,
    ):
        super().__init__()

        # Stream 1: Categorical Sequence
        self.seq_encoder = SequenceEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            seq_len=cat_seq_len,
            n_layers=transformer_layers,
            n_heads=transformer_heads,
            dropout=dropout_transformer,
        )

        # Stream 2: Continuous Preservation (No processing layers here)

        # Fusion Layer: Linear Stem (Structural Correction)
        # Concatenate flattened sequence + raw continuous -> Linear Projection
        fusion_dim = self.seq_encoder.flatten_dim + num_continuous
        self.stem = nn.Linear(fusion_dim, backbone_stages[0])

        # Backbone
        self.backbone = ResFunnelBackbone(backbone_stages, dropout_backbone)

        # Output Head: Minimalist
        self.head = nn.Linear(self.backbone.out_dim, 1)

    def forward(self, x_num, x_cat):
        # Stream 1
        x_seq = self.seq_encoder(x_cat)

        # Fusion: Concatenation
        x_fused = torch.cat([x_seq, x_num], dim=1)

        # Linear Stem (No BN immediately)
        x = self.stem(x_fused)

        # Backbone
        x = self.backbone(x)

        # Head
        logits = self.head(x)
        return logits


# ------------------------------------------------------------------------------
# Training & Execution Logic
# ------------------------------------------------------------------------------


def run_training(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=False):
    """
    Executes the training pipeline including data loading, model initialization,
    training loop, validation, and submission generation.
    """
    # Reproducibility
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader, vocab_size = get_dataloaders(
        batch_size=batch_size, num_workers=Config.NUM_WORKERS, load_cached_data=True
    )

    # Debugging: if debug is True, we assume the user might want to run a quick check.
    # Since we can't easily slice loaders, we rely on the epochs being low if debug is intended,
    # or we break loops early. Here we just print a message.
    if debug:
        print("Debug mode: Training for limited steps/epochs.")
        epochs = 2

    # Determine input dimensions from a batch
    sample_x_num, sample_x_cat, _ = next(iter(train_loader))
    num_continuous = sample_x_num.shape[1]
    cat_seq_len = sample_x_cat.shape[1]

    print(
        f"Vocab Size: {vocab_size}, Seq Len: {cat_seq_len}, Num Continuous: {num_continuous}"
    )

    # 2. Model Initialization
    model = BalancedProcessCompressHybrid(
        num_continuous=num_continuous,
        cat_seq_len=cat_seq_len,
        vocab_size=vocab_size,
        embed_dim=Config.EMBED_DIM,
        transformer_layers=Config.TRANSFORMER_LAYERS,
        transformer_heads=Config.TRANSFORMER_HEADS,
        backbone_stages=Config.BACKBONE_STAGES,
        dropout_transformer=Config.DROPOUT_TRANSFORMER,
        dropout_backbone=Config.DROPOUT_BACKBONE,
    ).to(device)

    # Weight Initialization (Context-Aware)
    model.apply(custom_weight_init)

    # 3. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Aggressive Step Learning Rate Scheduler (Decay 0.1 every 10 epochs)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_auc = 0.0
    best_model_path = Config.BEST_MODEL_PATH

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for i, (x_num, x_cat, y) in enumerate(train_loader):
            x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x_num, x_cat).squeeze()
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if debug and i > 10:
                break  # Fast debug

        # Step the scheduler
        scheduler.step()

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for x_num, x_cat, y in val_loader:
                x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
                logits = model(x_num, x_cat).squeeze()
                preds = torch.sigmoid(logits)
                val_preds.append(preds.cpu().numpy())
                val_targets.append(y.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_auc = roc_auc_score(val_targets, val_preds)

        avg_loss = train_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.6f} | Val AUC: {val_auc:.10f} | LR: {current_lr:.2e}"
        )

        # Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best Validation AUC: {best_auc:.10f}")

    # 5. Inference
    print("Generating submission...")
    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_preds = []
    # We need test IDs. They are returned by process_data but get_dataloaders doesn't return them directly.
    # However, we can reload them or assume the order matches.
    # To be safe, we load test_ids from cache or metadata.
    # The cache file is processed_data.npz.
    data_cache = np.load(
        os.path.join(Config.CACHE_DIR, "processed_data.npz"), allow_pickle=True
    )
    test_ids = data_cache["test_ids"]

    with torch.no_grad():
        for x_num, x_cat in test_loader:
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            logits = model(x_num, x_cat).squeeze()
            preds = torch.sigmoid(logits)
            test_preds.append(preds.cpu().numpy())

    test_preds = np.concatenate(test_preds)

    # 6. Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub = pd.DataFrame({"id": test_ids, "target": test_preds})
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
