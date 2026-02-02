import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from library.config import Config, sinusoidal_encoding
from library.dataset import prepare_data, RNADataset
from library.utils import set_seed, mcrmse_metric, build_submission_df


class RNAModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = Config()
        self.config = config

        # Embeddings
        self.char_emb = nn.Embedding(4, config.EMBED_DIM_CHAR)
        self.loop_emb = nn.Embedding(7, config.EMBED_DIM_LOOP)
        # Distance embedding is computed via sinusoidal_encoding

        # Total Input Embedding Dimension E
        self.embed_dim = (
            config.EMBED_DIM_CHAR + config.EMBED_DIM_LOOP + config.EMBED_DIM_DIST
        )

        # Backbone: Input-Injected Residual BiGRU
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # BiGRU hidden size per direction (Total hidden dim split by 2)
        gru_hidden_size = config.HIDDEN_DIM // 2

        for i in range(config.NUM_LAYERS):
            # Layer 0: Input is just the Embedding E
            # Layer >0: Input is Concat(Norm(h_prev), E) -- Input Injection
            if i == 0:
                layer_input_dim = self.embed_dim
            else:
                layer_input_dim = config.HIDDEN_DIM + self.embed_dim

            self.layers.append(
                nn.GRU(
                    input_size=layer_input_dim,
                    hidden_size=gru_hidden_size,
                    batch_first=True,
                    bidirectional=True,
                )
            )

            # Pre-LayerNorm
            # We apply Norm to the residual path input
            if i > 0:
                self.norms.append(nn.LayerNorm(config.HIDDEN_DIM))
            else:
                self.norms.append(nn.Identity())

        self.dropout = nn.Dropout(config.DROPOUT)

        # Output Head
        self.head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.Dropout(config.DROPOUT),
            nn.Linear(
                config.HIDDEN_DIM, 3
            ),  # Predicts: reactivity, deg_Mg_pH10, deg_Mg_50C
        )

    def forward(self, seq, loop, dist):
        # 1. Construct Embedding E
        x_char = self.char_emb(seq)
        x_loop = self.loop_emb(loop)
        x_dist = sinusoidal_encoding(dist, self.config.EMBED_DIM_DIST)

        # E: (B, L, D_total)
        E = torch.cat([x_char, x_loop, x_dist], dim=-1)

        # 2. Backbone with Input Injection
        current_h = None

        for i, layer in enumerate(self.layers):
            if i == 0:
                # First layer takes embedding directly
                out, _ = layer(E)
                current_h = out
            else:
                # Subsequent layers: Inject E into the input
                # h_l = h_{l-1} + GRU(Concat(Norm(h_{l-1}), E))

                h_norm = self.norms[i](current_h)
                gru_input = torch.cat([h_norm, E], dim=-1)

                out, _ = layer(gru_input)

                # Residual Connection
                current_h = current_h + out

        # 3. Output Head
        features = self.dropout(current_h)
        logits = self.head(features)  # (B, L, 3)

        return logits


def train_model():
    config = Config()
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare Data (using library function with caching)
    datasets = prepare_data(config, load_cached_data=True)

    train_loader = DataLoader(
        datasets["train"],
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    # Model Setup
    model = RNAModel(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    criterion = nn.MSELoss()

    best_mcrmse = float("inf")
    patience = 0
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {device}...")

    for epoch in range(config.EPOCHS):
        # Training Phase
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            targets = batch["targets"].to(device)  # (B, 68, 3)

            optimizer.zero_grad()
            preds = model(seq, loop, dist)  # (B, 107, 3)

            # Loss is calculated only on the first 68 scored positions
            preds_scored = preds[:, : config.SEQ_SCORED, :]

            loss = criterion(preds_scored, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # Validation Phase
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
                preds_scored = preds[:, : config.SEQ_SCORED, :]

                all_preds.append(preds_scored.cpu())
                all_targets.append(targets.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        val_mcrmse = mcrmse_metric(all_targets, all_preds)

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train MSE: {avg_train_loss:.6f} | Val MCRMSE: {val_mcrmse:.6f}"
        )

        # Checkpointing & Early Stopping
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)
            patience = 0
        else:
            patience += 1
            if patience >= config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Best Val MCRMSE: {best_mcrmse:.6f}")

    # Inference on Test Set
    print("Generating submission...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_loader = DataLoader(
        datasets["test"],
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    all_test_preds = []
    all_test_ids = []

    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            ids = batch["id"]

            preds = model(seq, loop, dist)  # (B, 107, 3)

            all_test_preds.append(preds.cpu())
            all_test_ids.extend(ids)

    all_test_preds = torch.cat(all_test_preds, dim=0)

    # Format and Save Submission
    submission_df = build_submission_df(
        all_test_ids, all_test_preds, seq_len=config.SEQ_LENGTH
    )
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")


# Execute the pipeline
train_model()
