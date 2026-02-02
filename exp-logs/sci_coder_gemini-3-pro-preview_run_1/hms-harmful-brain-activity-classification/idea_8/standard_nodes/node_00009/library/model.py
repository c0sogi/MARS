import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
import os
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import (
    seed_everything,
    AverageMeter,
    KLDivLossWithLogits,
    kl_divergence_score,
)
from library.data import get_dataloaders

# =========================================================================
# Model Components
# =========================================================================


class Inception1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernels=[3, 5, 7, 9]):
        super().__init__()
        self.branches = nn.ModuleList()
        n_branches = len(kernels)
        ch_per_branch = out_channels // n_branches
        remainder = out_channels % n_branches

        for i, k in enumerate(kernels):
            out_c = ch_per_branch + (1 if i < remainder else 0)
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_c, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(out_c),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], dim=1)


class ContinuousPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, x):
        # x: (B, T) - relative time in seconds
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(x.device)
        args = x.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)  # (B, T, Half)

        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if self.dim % 2 == 1:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :, :1])], dim=-1
            )

        return embedding


class EEGEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(config.EEG_CHANNELS, 32, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.blocks = nn.Sequential(
            Inception1d(32, 64, config.EEG_KERNELS),
            nn.MaxPool1d(4),
            Inception1d(64, 128, config.EEG_KERNELS),
            nn.MaxPool1d(4),
            Inception1d(128, 256, config.EEG_KERNELS),
            nn.MaxPool1d(2),
            Inception1d(256, config.EEG_MODEL_DIM, config.EEG_KERNELS),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return x.permute(2, 0, 1)  # (Seq, Batch, Dim)


class SpectrogramEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = timm.create_model(
            config.SPEC_BACKBONE,
            pretrained=config.SPEC_PRETRAINED,
            features_only=True,
            out_indices=(4,),
        )

        # Determine output channels dynamically
        dummy = torch.randn(1, 3, config.SPEC_SIZE[0], config.SPEC_SIZE[1])
        with torch.no_grad():
            feats = self.backbone(dummy)
            out_ch = feats[0].shape[1]

        self.proj = nn.Conv1d(out_ch, config.SPEC_EMBED_DIM, kernel_size=1)
        self.pos_encoder = ContinuousPositionalEmbedding(config.SPEC_EMBED_DIM)

    def forward(self, x, rel_indices):
        # x: (B, 3, H, W). H=Time, W=Freq
        feats = self.backbone(x)[0]  # (B, 1280, H/32, W/32)

        # Pool Frequency (Width) to keep Time (Height)
        x_time = feats.mean(dim=3)  # (B, 1280, H_feat)

        # Project
        x_proj = self.proj(x_time)  # (B, Dim, H_feat)

        # Align Relative Indices
        B, T_orig = rel_indices.shape
        T_feat = x_proj.shape[2]

        rel_down = F.adaptive_avg_pool1d(rel_indices.unsqueeze(1), T_feat).squeeze(
            1
        )  # (B, T_feat)

        # Positional Embedding
        pos_emb = self.pos_encoder(rel_down)  # (B, T_feat, Dim)
        pos_emb = pos_emb.permute(0, 2, 1)  # (B, Dim, T_feat)

        # Add
        x_out = x_proj + pos_emb

        return x_out.permute(2, 0, 1)  # (Seq, Batch, Dim)


class ChronologicallyEmbeddedDualStream(nn.Module):
    def __init__(self, config=Config):
        super().__init__()
        self.eeg_encoder = EEGEncoder(config)
        self.spec_encoder = SpectrogramEncoder(config)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.TRANSFORMER_DIM,
            nhead=config.TRANSFORMER_HEADS,
            dim_feedforward=config.TRANSFORMER_FF_DIM,
            dropout=config.DROPOUT,
            activation="gelu",
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.TRANSFORMER_LAYERS
        )

        self.head = nn.Sequential(
            nn.Linear(config.TRANSFORMER_DIM, 64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, config.NUM_CLASSES),
        )

    def forward(self, eeg, spec, rel_indices):
        eeg_feats = self.eeg_encoder(eeg)
        spec_feats = self.spec_encoder(spec, rel_indices)

        # Transformer Decoder: tgt=EEG, memory=Spec
        out = self.transformer_decoder(tgt=eeg_feats, memory=spec_feats)

        # Global Average Pooling
        out = out.mean(dim=0)

        logits = self.head(out)
        return logits


# =========================================================================
# Training & Inference
# =========================================================================


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    loss_meter = AverageMeter()

    for batch in loader:
        eeg, spec, rel, targets = batch
        eeg, spec, rel, targets = (
            eeg.to(device),
            spec.to(device),
            rel.to(device),
            targets.to(device),
        )

        optimizer.zero_grad()
        logits = model(eeg, spec, rel)
        loss = criterion(logits, targets)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        loss_meter.update(loss.item(), eeg.size(0))

    return loss_meter.avg


def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    kl_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            eeg, spec, rel, targets = batch
            eeg, spec, rel, targets = (
                eeg.to(device),
                spec.to(device),
                rel.to(device),
                targets.to(device),
            )

            logits = model(eeg, spec, rel)
            loss = criterion(logits, targets)

            probs = F.softmax(logits, dim=1).cpu().numpy()
            kl = kl_divergence_score(targets.cpu().numpy(), probs)

            loss_meter.update(loss.item(), eeg.size(0))
            kl_meter.update(kl, eeg.size(0))

    return loss_meter.avg, kl_meter.avg


def predict(model, loader, device):
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            eeg, spec, rel = batch
            eeg, spec, rel = eeg.to(device), spec.to(device), rel.to(device)

            logits = model(eeg, spec, rel)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


def run_training(config=Config):
    seed_everything(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        train_batch_size=config.BATCH_SIZE,
        val_batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        debug=config.DEBUG,
        debug_subset_size=config.DEBUG_SUBSET_SIZE,
    )

    model = ChronologicallyEmbeddedDualStream(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=config.PCT_START,
        div_factor=config.DIV_FACTOR,
        final_div_factor=config.FINAL_DIV_FACTOR,
    )

    criterion = KLDivLossWithLogits()

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_kl = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val KL: {val_kl}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("Saved Best Model.")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print("Early stopping triggered.")
                break

    # Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    probs = predict(model, test_loader, device)

    sub_df = pd.read_csv(config.TEST_CSV)
    for i, col in enumerate(config.CLASS_NAMES):
        sub_df[col] = probs[:, i]

    sub_df = sub_df[["eeg_id"] + config.CLASS_NAMES]
    sub_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
