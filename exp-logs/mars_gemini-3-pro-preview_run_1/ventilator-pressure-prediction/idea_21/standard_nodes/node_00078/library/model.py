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
        # Cite solution_lesson_node_00039: Deep Context Injection
        self.lstm = nn.LSTM(
            input_size=input_dim + context_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        # Project residual: input_dim -> wide_dim
        self.res_proj = nn.Linear(input_dim, wide_dim)
        self.dropout = nn.Dropout(dropout)

        # Cite solution_lesson_node_00040: Pointwise FFN
        self.ffn = nn.Sequential(
            nn.Linear(wide_dim, wide_dim * 2),
            nn.GELU(),
            nn.Linear(wide_dim * 2, wide_dim),
            nn.Dropout(dropout),
        )
        # Cite solution_lesson_node_00044: Avoid LayerNorm in residual stream for regression

    def forward(self, x, context):
        # x: (B, L, input_dim)
        # context: (B, L, context_dim)

        # Concatenate context
        lstm_in = torch.cat([x, context], dim=-1)

        # LSTM
        lstm_out, _ = self.lstm(lstm_in)

        # Residual connection with projection
        res = self.res_proj(x)
        out = lstm_out + self.dropout(res)

        # FFN (Additive)
        ffn_out = self.ffn(out)
        out = out + ffn_out

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

    def forward(self, x, context):
        # x: (B, L, wide_dim)

        # Concatenate context
        lstm_in = torch.cat([x, context], dim=-1)

        lstm_out, _ = self.lstm(lstm_in)

        # Strict Identity Residual
        out = lstm_out + self.dropout(x)

        # FFN
        ffn_out = self.ffn(out)
        out = out + ffn_out

        return out


class GraduatedCapacityNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.stem = MultiScaleConvStem(input_dim, config.STEM_DIM, config.KERNEL_SIZES)

        # We need to determine static_dim dynamically or assume it.
        # Since we extract it in forward, we assume the size matches the number of static features.
        # R and C are the main static features. Let's assume static_dim will be passed or inferred.
        # However, nn.Module needs fixed shapes for weights.
        # Based on dataset, R and C are 2 features.
        # But we also have one-hot or other engineered features?
        # data.py scales R and C. They are just 2 columns.
        # We will assume static_dim = 2 (R, C).
        self.static_dim = 2

        self.block1 = ExpansionBlock(
            config.STEM_DIM,
            self.static_dim,
            config.WIDE_DIM,
            config.LSTM_HIDDEN,
            config.DROPOUT,
        )

        self.block2 = IdentityBlock(
            config.WIDE_DIM, self.static_dim, config.LSTM_HIDDEN, config.DROPOUT
        )

        self.block3 = IdentityBlock(
            config.WIDE_DIM, self.static_dim, config.LSTM_HIDDEN, config.DROPOUT
        )

        self.block4 = IdentityBlock(
            config.WIDE_DIM, self.static_dim, config.LSTM_HIDDEN, config.DROPOUT
        )

        self.aux_head = nn.Linear(config.WIDE_DIM, 1)
        self.head = nn.Linear(config.WIDE_DIM, 1)

    def forward(self, x, u_out):
        # x: (B, L, F)
        # u_out: (B, L)

        # 1. Combine inputs
        u_out_unsqueezed = u_out.unsqueeze(-1)
        raw_input = torch.cat([x, u_out_unsqueezed], dim=-1)

        # 2. Identify Static Features (R, C)
        # Calculate std along time dimension (dim 1)
        # Static features have std ~ 0
        # We do this on 'x' only, as u_out is dynamic
        std = x.std(dim=1)  # (B, F)
        # We take the mean std across batch to decide which indices are static
        mean_std = std.mean(dim=0)
        static_mask = mean_std < 1e-4

        # Extract static features
        # We expand them back to (B, L, static_dim)
        # Note: This logic assumes the set of static features is constant across the dataset structure.
        # For safety in the computation graph, we select based on the mask.
        if static_mask.sum() != self.static_dim:
            # Fallback if detection fails (e.g. during early training or if scaling noise exists)
            # We assume the last 2 or first 2? R and C are usually columns.
            # To be safe and robust, if we can't detect exactly 2, we project the whole x to static_dim
            # But the prompt asks for "Curated Context".
            # Let's trust the std check. If it fails, we might crash or need a flexible layer.
            # To avoid crash, we use a linear projection of the first step as context if count mismatches.
            # But for this implementation, let's assume R and C are detectable.
            # Actually, let's just project the first time step of x to static_dim to be safe.
            # This allows the model to learn which features are static from the first step.
            pass

        # Robust Static Context Extraction:
        # Take the first time step x[:, 0, :] -> (B, F)
        # Since we can't guarantee indices of R/C, we'll assume the model can learn to use
        # the relevant static info if we pass the features that are constant.
        # Instead of masking, we will extract the features that are strictly constant in this batch.
        # But to ensure tensor shape consistency for the LSTM weights defined in __init__,
        # we must have exactly `self.static_dim` features.
        # Strategy: We will assume R and C are present. We will use a learnable projection
        # from the first time step x[:, 0, :] to `static_dim` (2).
        # This satisfies "Curated Context" by allowing the network to compress the initial state
        # (which contains R, C) into the static vector.

        # Wait, I can't introduce a new layer not in __init__.
        # I will add a context adapter in __init__.
        pass

    def _get_static_context(self, x):
        # Helper to extract static context.
        # We use the first time step.
        # We assume R and C are among the features.
        # To strictly follow "exclude dynamic", we rely on the fact that dynamic features
        # at t=0 might not be informative for the whole sequence as context,
        # but R and C are.
        # We will use a simplified approach:
        # We'll use the first 2 features with lowest variance in the batch? No, too slow.
        # We will simply project the entire feature vector at t=0 to static_dim.
        return self.context_proj(x[:, 0, :]).unsqueeze(1).repeat(1, x.size(1), 1)


# Redefining class with the robust context strategy
class GraduatedCapacityNetworkRobust(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # Input dim includes u_out? No, x is features. u_out is separate.
        # Stem takes x + u_out
        self.stem = MultiScaleConvStem(
            input_dim + 1, config.STEM_DIM, config.KERNEL_SIZES
        )

        # Cite solution_lesson_node_00074: Explicit Parameter Injection
        # Cite solution_lesson_node_00076: Explicit Injection of Derived Physical Interactions
        self.context_dim = len(config.CONTEXT_FEATURES)

        self.block1 = ExpansionBlock(
            config.STEM_DIM,
            self.context_dim,
            config.WIDE_DIM,
            config.LSTM_HIDDEN,
            config.DROPOUT,
        )
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

        # Prepare inputs
        u_out_unsqueezed = u_out.unsqueeze(-1)
        stem_in = torch.cat([x, u_out_unsqueezed], dim=-1)

        # Stem
        h = self.stem(stem_in)

        # Context Extraction
        # We assume the first `context_dim` features are the context features
        # due to the reordering in data.py
        context = x[:, :, : self.context_dim]

        # Block 1 (Expansion)
        h = self.block1(h, context)

        # Block 2 (Identity) + Aux
        h = self.block2(h, context)
        aux_pred = self.aux_head(h).squeeze(-1)

        # Block 3
        h = self.block3(h, context)

        # Block 4 + Final
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

    model = GraduatedCapacityNetworkRobust(input_dim).to(config.DEVICE)

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

    model = GraduatedCapacityNetworkRobust(input_dim).to(config.DEVICE)
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
