import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library import config
from library import utils
from library import data

# =============================================================================
# Loss Function
# =============================================================================


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, pred, target, u_out):
        # pred, target: (B, L)
        # u_out: (B, L)
        loss = self.l1(pred, target)
        # Mask: 1 - u_out (1 for inspiratory, 0 for expiratory)
        mask = 1 - u_out
        loss = loss * mask
        # Sum over valid steps and divide by total valid steps
        return loss.sum() / (mask.sum() + 1e-8)


# =============================================================================
# Model Components
# =============================================================================


class MultiScaleConvStem(nn.Module):
    def __init__(self, input_dim, stem_dim, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(input_dim, stem_dim // len(kernel_sizes), k, padding="same")
                for k in kernel_sizes
            ]
        )
        # Projection to ensure exact stem_dim output
        concat_dim = (stem_dim // len(kernel_sizes)) * len(kernel_sizes)
        self.proj = nn.Conv1d(concat_dim, stem_dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, L, F) -> (B, F, L) for Conv1d
        x = x.transpose(1, 2)
        outs = [conv(x) for conv in self.convs]
        x = torch.cat(outs, dim=1)
        x = self.proj(x)
        x = self.act(x)
        # Back to (B, L, F)
        return x.transpose(1, 2)


class ExpansionBlock(nn.Module):
    def __init__(self, input_dim, context_dim, wide_dim, lstm_hidden, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim + context_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        # Project residual: input_dim -> wide_dim
        self.res_proj = nn.Linear(input_dim, wide_dim)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(wide_dim, wide_dim * 2),
            nn.GELU(),
            nn.Linear(wide_dim * 2, wide_dim),
            nn.Dropout(dropout),
        )
        # Removed LayerNorm as per Lesson 00044 for regression tasks

    def forward(self, x, context):
        # x: (B, L, input_dim)
        # context: (B, L, context_dim)

        # Context Injection (Lesson 00039, 00074)
        lstm_in = torch.cat([x, context], dim=-1)

        # LSTM
        lstm_out, _ = self.lstm(lstm_in)  # (B, L, wide_dim)

        # Residual connection with projection
        res = self.res_proj(x)
        out = lstm_out + self.dropout(res)

        # Pointwise FFN (Lesson 00040)
        ffn_out = self.ffn(out)
        out = out + self.dropout(ffn_out)

        return out


class IdentityBlock(nn.Module):
    def __init__(self, wide_dim, context_dim, lstm_hidden, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=wide_dim + context_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(wide_dim, wide_dim * 2),
            nn.GELU(),
            nn.Linear(wide_dim * 2, wide_dim),
            nn.Dropout(dropout),
        )
        # Removed LayerNorm as per Lesson 00044

    def forward(self, x, context):
        # x: (B, L, wide_dim)

        # Context Injection (Lesson 00039, 00074)
        lstm_in = torch.cat([x, context], dim=-1)

        lstm_out, _ = self.lstm(lstm_in)

        # Additive Residual (Lesson 00033, 00054)
        out = lstm_out + self.dropout(x)

        # Pointwise FFN (Lesson 00040)
        ffn_out = self.ffn(out)
        out = out + self.dropout(ffn_out)

        return out


class DeepContextInjectedNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # Stem takes x + u_out
        self.stem = MultiScaleConvStem(
            input_dim + 1, config.STEM_DIM, config.KERNEL_SIZES
        )

        # Explicit Context Dimensions (R, C)
        self.context_dim = 2

        # Block 1: Expansion (Bottleneck Init - Lesson 00073)
        self.block1 = ExpansionBlock(
            config.STEM_DIM,
            self.context_dim,
            config.WIDE_DIM,
            config.LSTM_HIDDEN,
            config.DROPOUT,
        )

        # Deep Backbone
        self.block2 = IdentityBlock(
            config.WIDE_DIM, self.context_dim, config.LSTM_HIDDEN, config.DROPOUT
        )
        self.block3 = IdentityBlock(
            config.WIDE_DIM, self.context_dim, config.LSTM_HIDDEN, config.DROPOUT
        )
        self.block4 = IdentityBlock(
            config.WIDE_DIM, self.context_dim, config.LSTM_HIDDEN, config.DROPOUT
        )

        self.aux_head = nn.Linear(config.WIDE_DIM, 1)
        self.head = nn.Linear(config.WIDE_DIM, 1)

    def forward(self, x, u_out):
        # x: (B, L, F)
        # u_out: (B, L)

        # 1. Explicit Context Extraction (Lesson 00074)
        # Assuming R and C are the first two features based on data pipeline
        # x[:, :, 0] is R, x[:, :, 1] is C
        context = x[:, :, : self.context_dim]

        # 2. Stem
        u_out_unsqueezed = u_out.unsqueeze(-1)
        stem_in = torch.cat([x, u_out_unsqueezed], dim=-1)
        h = self.stem(stem_in)

        # 3. Deep Context Injected Backbone
        h = self.block1(h, context)

        h = self.block2(h, context)
        # Deep Supervision (Lesson 00039)
        aux_pred = self.aux_head(h).squeeze(-1)

        h = self.block3(h, context)
        h = self.block4(h, context)

        final_pred = self.head(h).squeeze(-1)

        return final_pred, aux_pred


# =============================================================================
# Training Function
# =============================================================================


def train_model(dataloaders=None):
    utils.seed_everything()

    # Load data if not provided
    if dataloaders is None:
        train_loader, val_loader, _ = data.get_dataloaders()
    else:
        train_loader, val_loader, _ = dataloaders

    # Determine input dimension from a batch
    sample_x, _, _ = next(iter(train_loader))
    input_dim = sample_x.shape[-1]

    model = DeepContextInjectedNetwork(input_dim).to(config.DEVICE)

    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    criterion = MaskedL1Loss()

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
    )

    best_mae = float("inf")
    patience_counter = 0

    print(f"Starting training for {config.EPOCHS} epochs...")

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0

        for x, u_out, y in train_loader:
            x, u_out, y = (
                x.to(config.DEVICE),
                u_out.to(config.DEVICE),
                y.to(config.DEVICE),
            )

            optimizer.zero_grad()

            pred, aux_pred = model(x, u_out)

            loss_final = criterion(pred, y, u_out)
            loss_aux = criterion(aux_pred, y, u_out)

            loss = loss_final + config.AUX_WEIGHT * loss_aux

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_uouts = []

        with torch.no_grad():
            for x, u_out, y in val_loader:
                x, u_out, y = (
                    x.to(config.DEVICE),
                    u_out.to(config.DEVICE),
                    y.to(config.DEVICE),
                )
                pred, _ = model(x, u_out)

                val_preds.append(pred.cpu().numpy())
                val_targets.append(y.cpu().numpy())
                val_uouts.append(u_out.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_uouts = np.concatenate(val_uouts)

        mae = utils.compute_mae(val_preds, val_targets, val_uouts)

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val MAE: {mae:.9f}"
        )

        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Best Val MAE: {best_mae:.9f}")
    return model


# =============================================================================
# Prediction Function
# =============================================================================


def predict(dataloaders=None):
    utils.seed_everything()

    if dataloaders is None:
        _, _, test_loader = data.get_dataloaders(load_cached_data=True)
    else:
        _, _, test_loader = dataloaders

    # Determine input dimension
    sample_x, _, _ = next(iter(test_loader))
    input_dim = sample_x.shape[-1]

    model = DeepContextInjectedNetwork(input_dim).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.eval()

    predictions = []
    ids_list = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            x, u_out, ids = batch
            x, u_out = x.to(config.DEVICE), u_out.to(config.DEVICE)

            pred, _ = model(x, u_out)

            # Flatten predictions and ids
            predictions.extend(pred.cpu().numpy().flatten())
            ids_list.extend(ids.numpy().flatten())

    # Create submission dataframe
    sub_df = pd.DataFrame({"id": ids_list, "pressure": predictions})

    sub_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
