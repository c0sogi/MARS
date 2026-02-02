import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import os
import numpy as np
import pandas as pd
from library.config import Config
from library.dataset import get_dataloaders
from library.loss import MaskedMSELoss
from library.utils import set_seed, mcrmse_metric


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Encodes signed distances using fixed sinusoidal functions.
    Preserves the sign information (upstream vs downstream).
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: (B, L) signed distances
        half_dim = self.dim // 2
        # Use log scale for frequencies
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)

        x_ex = x.unsqueeze(-1)  # (B, L, 1)
        emb_ex = emb.view(1, 1, -1)  # (1, 1, half_dim)

        args = x_ex * emb_ex
        # Concatenate sin and cos to form the full embedding
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class VectorScaledResidualBlock(nn.Module):
    """
    A Wide-Stream BiGRU block with Channel-Wise Residual Scaling.
    Structure: Pre-LN -> BiGRU -> Dropout -> Vector Scale -> Add.
    """

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(
            hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        # Vector Scaling: Learnable diagonal matrix initialized to Identity (ones)
        # This allows the model to independently regulate feature channels.
        self.scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x):
        residual = x
        out = self.ln(x)
        out, _ = self.gru(out)
        out = self.dropout(out)
        # Apply channel-wise scaling
        out = out * self.scale
        return residual + out


class ScalarMixture(nn.Module):
    """
    Aggregates outputs from multiple layers using a learnable weighted sum.
    """

    def __init__(self, num_layers):
        super().__init__()
        # Initialize weights to zero (softmax will make them uniform initially)
        self.mix_weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, tensors):
        # tensors: list of (B, L, H)
        stacked = torch.stack(tensors, dim=-1)  # (B, L, H, N_Layers)
        weights = F.softmax(self.mix_weights, dim=0)
        # Weighted sum across the last dimension
        return torch.sum(stacked * weights, dim=-1)


class VectorScaledHighCapacityBiGRU(nn.Module):
    """
    Main Architecture:
    1. Heterogeneous Embeddings (Seq + Loop + Dist)
    2. High-Fidelity Stem (BiGRU, No Dropout)
    3. 6 Vector-Scaled Wide Blocks
    4. Scalar Mixture Aggregation
    5. Output Head
    """

    def __init__(self):
        super().__init__()

        # 1. Embeddings
        self.seq_emb = nn.Embedding(4, Config.EMBED_DIM_SEQ)
        self.loop_emb = nn.Embedding(7, Config.EMBED_DIM_LOOP)
        self.dist_emb = SinusoidalPositionalEmbedding(Config.EMBED_DIM_DIST)

        input_dim = Config.EMBED_DIM_SEQ + Config.EMBED_DIM_LOOP + Config.EMBED_DIM_DIST

        # 2. Stem (Project to Hidden Dim)
        self.stem_gru = nn.GRU(
            input_dim, Config.HIDDEN_DIM // 2, batch_first=True, bidirectional=True
        )

        # 3. Backbone (Vector Scaled Blocks)
        self.blocks = nn.ModuleList(
            [
                VectorScaledResidualBlock(Config.HIDDEN_DIM, Config.DROPOUT)
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # 4. Aggregation (Stem + 6 Blocks)
        self.mixture = ScalarMixture(Config.NUM_LAYERS + 1)

        # 5. Head
        self.head = nn.Linear(Config.HIDDEN_DIM, len(Config.TARGET_COLS))

    def forward(self, seq, loop, dist):
        # Embed inputs
        x_seq = self.seq_emb(seq)
        x_loop = self.loop_emb(loop)
        x_dist = self.dist_emb(dist)

        # Fuse embeddings
        x = torch.cat([x_seq, x_loop, x_dist], dim=-1)

        # Stem Pass
        x, _ = self.stem_gru(x)
        outputs = [x]

        # Backbone Pass
        for block in self.blocks:
            x = block(x)
            outputs.append(x)

        # Aggregate
        x_agg = self.mixture(outputs)

        # Predict
        logits = self.head(x_agg)
        return logits


def train_and_predict():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # 2. Setup Model & Optimization
    model = VectorScaledHighCapacityBiGRU().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    criterion = MaskedMSELoss()

    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 3. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_sum = 0.0

        for batch in train_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            preds = model(seq, loop, dist)

            # Compute loss only on masked (valid) positions
            loss = criterion(preds, targets, mask)
            loss.backward()

            # Gradient Clipping is critical for stability in 512-width networks
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            optimizer.step()

            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)

        # 4. Validation Loop
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                seq = batch["seq"].to(device)
                loop = batch["loop"].to(device)
                dist = batch["dist"].to(device)
                targets = batch["targets"].to(device)

                preds = model(seq, loop, dist)

                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

        # Concatenate for metric calculation
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calculate MCRMSE
        val_mcrmse = mcrmse_metric(all_targets, all_preds, pred_len=Config.PRED_LEN)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        # Save Best Model
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best MCRMSE: {best_mcrmse:.6f}")

    # 5. Inference & Submission
    print("Generating submission...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    submission_data = []

    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            ids = batch["id"]

            preds = model(seq, loop, dist)
            preds = preds.cpu().numpy()  # (B, 107, 3)

            for i, sample_id in enumerate(ids):
                for pos in range(Config.SEQ_LEN):
                    # Format: id_seqpos
                    row_id = f"{sample_id}_{pos}"

                    # Extract predictions
                    p_react = preds[i, pos, 0]
                    p_mg_ph10 = preds[i, pos, 1]
                    p_mg_50c = preds[i, pos, 2]

                    # Fill row (unscored columns are 0.0)
                    submission_data.append(
                        {
                            "id_seqpos": row_id,
                            "reactivity": p_react,
                            "deg_Mg_pH10": p_mg_ph10,
                            "deg_pH10": 0.0,
                            "deg_Mg_50C": p_mg_50c,
                            "deg_50C": 0.0,
                        }
                    )

    # Save to CSV
    sub_df = pd.DataFrame(submission_data)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


# Execute the pipeline
if __name__ == "__main__":
    train_and_predict()
