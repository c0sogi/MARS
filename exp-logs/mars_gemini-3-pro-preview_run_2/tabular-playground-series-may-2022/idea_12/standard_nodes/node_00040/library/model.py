import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
from library.config import Config

# ------------------------------------------------------------------------------
# Model Components
# ------------------------------------------------------------------------------


class GatedResidualBlock(nn.Module):
    """
    Residual Block with Gated Linear Unit (GLU), Dropout, and Batch Normalization.
    Handles dimension changes via projected residual connections.
    """

    def __init__(self, in_dim, out_dim, dropout_rate):
        super(GatedResidualBlock, self).__init__()

        # Signal path: Linear -> GLU -> Dropout -> BN
        # GLU requires output_dim * 2
        self.linear = nn.Linear(in_dim, out_dim * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.BatchNorm1d(out_dim)

        # Residual path: Projection if dimensions change
        if in_dim != out_dim:
            self.project = nn.Linear(in_dim, out_dim)
            self.project_norm = nn.BatchNorm1d(out_dim)
        else:
            self.project = None
            self.project_norm = None

    def forward(self, x):
        # 1. Residual Path
        if self.project is not None:
            residual = self.project_norm(self.project(x))
        else:
            residual = x

        # 2. Signal Path
        x_signal = self.linear(x)

        # GLU Operation: Split and Gate
        # x_signal shape: (B, out_dim * 2)
        out_dim = x_signal.size(-1) // 2
        a, b = x_signal.split(out_dim, dim=-1)
        # GLU with Sigmoid gating as per Idea
        x_signal = a * torch.sigmoid(b)

        # Regularization
        x_signal = self.dropout(x_signal)
        x_signal = self.norm(x_signal)

        # 3. Add Residual
        return residual + x_signal


class GlobalContextTransformerResFunnel(nn.Module):
    """
    Global-Context Transformer-ResFunnel Hybrid Architecture.
    Fuses categorical sequences with continuous context via Attention,
    then exploits features using a deep ResFunnel backbone.
    """

    def __init__(self):
        super(GlobalContextTransformerResFunnel, self).__init__()

        # --- Hyperparameters ---
        self.seq_len = Config.CHAR_SEQ_LEN  # 10
        self.vocab_size = Config.VOCAB_SIZE  # 30
        self.embed_dim = Config.EMBED_DIM  # 32
        self.num_cont = Config.NUM_CONTINUOUS_FEATURES  # 30

        # --- Stream 1: Categorical Sequence ---
        self.char_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        # Learnable Positional Encoding: Length 11 (1 Context + 10 Chars)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len + 1, self.embed_dim))

        # --- Stream 2: Continuous Context ---
        # Project 30 continuous features to 1 Context Token (32 dim)
        self.context_proj = nn.Linear(self.num_cont, self.embed_dim)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=Config.TRANSFORMER_HEADS,
            dim_feedforward=Config.TRANSFORMER_FF_DIM,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.TRANSFORMER_LAYERS
        )

        # --- Stream 3: Fusion & Backbone ---
        # Flattened Transformer Output: 11 tokens * 32 dim = 352
        flattened_dim = (self.seq_len + 1) * self.embed_dim
        # Concatenate with Raw Continuous Features (30)
        fusion_dim = flattened_dim + self.num_cont  # 382

        # Projection to Backbone Width
        backbone_widths = Config.BACKBONE_WIDTHS  # [512, 256, 128]
        self.fusion_proj = nn.Linear(fusion_dim, backbone_widths[0])
        self.fusion_norm = nn.BatchNorm1d(backbone_widths[0])

        # ResFunnel Backbone (3 Stages)
        self.blocks = nn.ModuleList()

        # Stage 1: 512 -> 512
        self.blocks.append(
            GatedResidualBlock(
                backbone_widths[0], backbone_widths[0], Config.DROPOUT_RATE
            )
        )

        # Stage 2: 512 -> 256
        self.blocks.append(
            GatedResidualBlock(
                backbone_widths[0], backbone_widths[1], Config.DROPOUT_RATE
            )
        )

        # Stage 3: 256 -> 128
        self.blocks.append(
            GatedResidualBlock(
                backbone_widths[1], backbone_widths[2], Config.DROPOUT_RATE
            )
        )

        # --- Output Head ---
        self.head = nn.Linear(backbone_widths[2], 1)

        self._init_weights()

    def _init_weights(self):
        # Initialize positional embeddings and linear layers
        nn.init.normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, continuous, sequence):
        # continuous: (B, 30)
        # sequence: (B, 10)

        B = continuous.size(0)

        # --- 1. Sequence Construction ---
        # Embed characters: (B, 10, 32)
        char_embs = self.char_embed(sequence)

        # Create Context Token: (B, 30) -> (B, 32) -> (B, 1, 32)
        context_token = self.context_proj(continuous).unsqueeze(1)

        # Concatenate: [Context, Chars] -> (B, 11, 32)
        transformer_input = torch.cat([context_token, char_embs], dim=1)

        # Add Positional Embeddings
        transformer_input = transformer_input + self.pos_embed

        # --- 2. Transformer Encoding ---
        # (B, 11, 32)
        trans_out = self.transformer(transformer_input)

        # Flatten: (B, 352)
        trans_flat = trans_out.reshape(B, -1)

        # --- 3. Fusion with Raw Signal ---
        # Concatenate: [Flattened Trans, Raw Continuous] -> (B, 382)
        fused = torch.cat([trans_flat, continuous], dim=1)

        # --- 4. Backbone ---
        x = self.fusion_proj(fused)
        x = self.fusion_norm(x)
        x = F.relu(x)  # Activation before entering blocks

        for block in self.blocks:
            x = block(x)

        # --- 5. Output ---
        logits = self.head(x)
        return torch.sigmoid(logits)


# ------------------------------------------------------------------------------
# Training & Inference Logic
# ------------------------------------------------------------------------------


def train_model(train_loader, val_loader):
    """
    Executes the training loop with AdamW, StepLR, and Early Stopping.
    """
    device = Config.DEVICE
    model = GlobalContextTransformerResFunnel().to(device)

    # Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    criterion = nn.BCELoss()

    # Tracking
    best_val_auc = 0.0
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            cont = batch["continuous"].to(device)
            seq = batch["sequence"].to(device)
            target = batch["target"].to(device).unsqueeze(1)

            optimizer.zero_grad()
            output = model(cont, seq)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                cont = batch["continuous"].to(device)
                seq = batch["sequence"].to(device)
                target = batch["target"].to(device).unsqueeze(1)

                output = model(cont, seq)
                loss = criterion(output, target)
                val_loss += loss.item()

                val_preds.append(output.cpu().numpy())
                val_targets.append(target.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)

        val_auc = roc_auc_score(val_targets, val_preds)

        # Update Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"Val AUC: {val_auc:.10f} | "
            f"LR: {current_lr:.2e}"
        )

        # --- Early Stopping & Checkpointing ---
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"--> Best model saved! AUC: {best_val_auc:.10f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    return best_val_auc


def predict_and_submit(test_loader):
    """
    Loads the best model, performs inference on the test set, and saves the submission file.
    """
    device = Config.DEVICE
    model = GlobalContextTransformerResFunnel().to(device)

    # Load best weights
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    print("Generating predictions on test set...")

    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            cont = batch["continuous"].to(device)
            seq = batch["sequence"].to(device)
            ids = batch["id"]

            output = model(cont, seq)

            ids_list.append(ids.numpy())
            preds_list.append(output.cpu().numpy())

    # Aggregate results
    all_ids = np.concatenate(ids_list)
    all_preds = np.concatenate(preds_list).flatten()

    # Create submission DataFrame
    df_sub = pd.DataFrame({"id": all_ids, "target": all_preds})

    # Save
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved successfully to {Config.SUBMISSION_PATH}")
