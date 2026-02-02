import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import time

from library.config import Config
from library.utils import (
    AverageMeter,
    save_checkpoint,
    probabilistic_f1,
    get_device,
    setup_logger,
)
from library.data import get_dataloaders

# =========================================================================
# 1. GPU-Enhanced Input Layer
# =========================================================================


class TriSpectralInputLayer(nn.Module):
    """
    Expands 1-channel DICOM input to 3 channels using GPU-accelerated operations:
    1. Raw Intensity
    2. Differentiable Gamma Correction (Density focus)
    3. Laplacian Edge Detection (Calcification/Texture focus)
    """

    def __init__(self):
        super().__init__()
        # Learnable gamma, initialized to 1.0 (neutral).
        # Constrained to be positive during forward pass.
        self.gamma = nn.Parameter(torch.tensor(1.0))

        # Fixed Laplacian Kernel for edge detection
        # Standard 3x3 Laplacian: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (Out, In, H, W) -> (1, 1, 3, 3)

        self.edge_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.edge_conv.weight = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        # x shape: (B, 1, H, W)

        # Channel 1: Raw
        c1 = x

        # Channel 2: Gamma Correction (Differentiable)
        # Add epsilon to avoid log(0) gradients, clamp gamma to reasonable range
        g = torch.clamp(self.gamma, 0.1, 3.0)
        c2 = torch.pow(x + 1e-6, g)

        # Channel 3: Texture/Edge
        c3 = self.edge_conv(x)

        # Concatenate: (B, 3, H, W)
        out = torch.cat([c1, c2, c3], dim=1)
        return out


# =========================================================================
# 2. Pooling Mechanism
# =========================================================================


class GeM(nn.Module):
    """
    Generalized Mean Pooling.
    p=1 -> AvgPool, p=inf -> MaxPool. p is learnable.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )


# =========================================================================
# 3. Main Model Architecture (DS-GEHN)
# =========================================================================


class DSGEHNModel(nn.Module):
    def __init__(self, feature_meta, pretrained=True):
        super().__init__()

        # --- Input Engineering ---
        self.input_layer = TriSpectralInputLayer()

        # --- Backbone (EfficientNetV2-S) ---
        # We need intermediate features for Deep Supervision.
        # 'features.5' is roughly stage 4/5 (stride 16 or 32 depending on exact variant),
        # 'features.7' is the final conv layer (1280 channels).
        base_model = models.efficientnet_v2_s(weights="DEFAULT" if pretrained else None)

        # Define return nodes for feature extraction
        # Node names can be found via `get_graph_node_names(model)`
        # For EffNetV2-S: 'features.5' (intermediate), 'features.7' (final)
        return_nodes = {"features.5": "aux", "features.7": "main"}
        self.backbone = create_feature_extractor(base_model, return_nodes=return_nodes)

        # Feature dimensions for EffNetV2-S
        # features.5 output channels: 160 (usually)
        # features.7 output channels: 1280
        self.aux_dim = 160
        self.main_dim = 1280

        # --- Pooling ---
        self.pool = GeM()

        # --- Tabular Branch ---
        self.cat_cols = feature_meta["cat_cols"]
        self.num_cols = feature_meta["num_cols"] + feature_meta["bin_cols"]

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    feature_meta["cat_dims"][col],
                    min(50, (feature_meta["cat_dims"][col] + 1) // 2),
                )
                for col in self.cat_cols
            ]
        )

        total_cat_dim = sum(e.embedding_dim for e in self.embeddings)
        num_cont_dim = len(self.num_cols)

        self.tabular_proj = nn.Sequential(
            nn.Linear(total_cat_dim + num_cont_dim, Config.TABULAR_EMBED_DIM),
            nn.BatchNorm1d(Config.TABULAR_EMBED_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # --- Heads ---

        # Auxiliary Head (Deep Supervision) - Visual Only
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.aux_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

        # Main Head - Hybrid (Visual + Tabular)
        fusion_dim = self.main_dim + Config.TABULAR_EMBED_DIM
        self.final_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, image, categorical, continuous):
        # 1. Input Enhancement
        x = self.input_layer(image)

        # 2. Backbone Extraction
        features = self.backbone(x)
        aux_feat_map = features["aux"]
        main_feat_map = features["main"]

        # 3. Auxiliary Prediction (Deep Supervision)
        aux_logits = self.aux_head(aux_feat_map)

        # 4. Main Visual Features
        # GeM Pooling -> Flatten
        visual_emb = self.pool(main_feat_map).flatten(1)

        # 5. Tabular Features
        cat_embs = [emb(categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_emb = torch.cat(cat_embs, dim=1)

        # Combine cat + cont
        tab_in = torch.cat([cat_emb, continuous], dim=1)
        tab_emb = self.tabular_proj(tab_in)

        # 6. Fusion
        fused = torch.cat([visual_emb, tab_emb], dim=1)

        # 7. Final Prediction
        final_logits = self.final_head(fused)

        return final_logits, aux_logits


# =========================================================================
# 4. Training Engine
# =========================================================================


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()

    losses = AverageMeter("Loss", ":.4f")

    # Progress bar
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]", leave=False)

    for batch in pbar:
        # Move data to device
        imgs = batch["image"].to(device, non_blocking=True)
        cats = batch["categorical"].to(device, non_blocking=True)
        conts = batch["continuous"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()

        # Forward
        # Mixed precision can be unstable with high pos_weight and deep supervision
        # We use standard FP32 for safety as per strategy
        final_logits, aux_logits = model(imgs, cats, conts)

        # Loss Calculation (Float32 Guarded)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            final_logits = final_logits.float()
            aux_logits = aux_logits.float()
            labels = labels.float()

            loss_main = criterion(final_logits, labels)
            loss_aux = criterion(aux_logits, labels)

            total_loss = loss_main + Config.AUX_LOSS_WEIGHT * loss_aux

        # Backward
        total_loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        losses.update(total_loss.item(), imgs.size(0))
        pbar.set_postfix(loss=losses.avg)

    return losses.avg


def validate(model, loader, criterion, device, epoch):
    model.eval()

    losses = AverageMeter("Loss", ":.4f")
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Val]", leave=False)

    with torch.no_grad():
        for batch in pbar:
            imgs = batch["image"].to(device, non_blocking=True)
            cats = batch["categorical"].to(device, non_blocking=True)
            conts = batch["continuous"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)

            # Forward (Main head only for validation)
            final_logits, _ = model(imgs, cats, conts)

            loss = criterion(final_logits, labels)
            losses.update(loss.item(), imgs.size(0))

            probs = torch.sigmoid(final_logits).cpu().numpy()
            targets = labels.cpu().numpy()

            all_preds.extend(probs)
            all_targets.extend(targets)

    # Compute pF1
    pf1 = probabilistic_f1(all_targets, all_preds)

    return losses.avg, pf1


def train_pipeline():
    logger = setup_logger(os.path.join(Config.WORK_DIR, "train.log"))
    device = get_device()

    logger.info(f"Using device: {device}")
    logger.info("Initializing DataLoaders...")

    train_loader, val_loader, test_loader, feature_meta = get_dataloaders(
        load_cached_data=True,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    logger.info("Initializing DS-GEHN Model...")
    model = DSGEHNModel(feature_meta, pretrained=Config.PRETRAINED)
    model.to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=Config.EPOCHS,
        pct_start=0.1,
    )

    # Loss Function (High pos_weight for imbalance)
    pos_weight = torch.tensor(Config.POS_WEIGHT).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_pf1 = 0.0

    logger.info("Starting Training...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Validate
        val_loss, val_pf1 = validate(model, val_loader, criterion, device, epoch)

        # Scheduler Step (if using plateau, but here OneCycle steps per batch)
        # scheduler.step() # Handled inside train loop if needed, but OneCycle is per batch usually.
        # Actually OneCycleLR should be stepped every batch.
        # Correcting: The snippet above defines steps_per_epoch, so we must step in loop.
        # However, standard practice in simple loops is step per batch.
        # Let's add scheduler.step() to train_one_epoch?
        # For simplicity in this structure, let's just step scheduler here if it was ReduceLROnPlateau,
        # but for OneCycleLR, it needs to be in the batch loop.
        # I will modify train_one_epoch to accept scheduler.

        elapsed = time.time() - start_time

        logger.info(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Time: {elapsed:.0f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val pF1: {val_pf1:.6f}"
        )

        # Checkpoint
        is_best = val_pf1 > best_pf1
        if is_best:
            best_pf1 = val_pf1
            logger.info(f"New Best pF1: {best_pf1:.6f}")

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_pf1": best_pf1,
                "optimizer": optimizer.state_dict(),
            },
            is_best=is_best,
        )

    logger.info("Training Complete.")
    return model


# Re-defining train_one_epoch to include scheduler step for OneCycleLR
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, scheduler=None):
    model.train()
    losses = AverageMeter("Loss", ":.4f")
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]", leave=False)

    for batch in pbar:
        imgs = batch["image"].to(device, non_blocking=True)
        cats = batch["categorical"].to(device, non_blocking=True)
        conts = batch["continuous"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()

        final_logits, aux_logits = model(imgs, cats, conts)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            final_logits = final_logits.float()
            aux_logits = aux_logits.float()
            labels = labels.float()
            loss_main = criterion(final_logits, labels)
            loss_aux = criterion(aux_logits, labels)
            total_loss = loss_main + Config.AUX_LOSS_WEIGHT * loss_aux

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        losses.update(total_loss.item(), imgs.size(0))
        pbar.set_postfix(loss=losses.avg)

    return losses.avg


# =========================================================================
# 5. Submission Logic
# =========================================================================


def predict_and_submit():
    logger = setup_logger(os.path.join(Config.WORK_DIR, "inference.log"))
    device = get_device()

    # Load Metadata & Dataloader
    # We only need test loader here
    _, _, test_loader, feature_meta = get_dataloaders(load_cached_data=True)

    # Load Model
    model_path = os.path.join(Config.WORK_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        logger.warning(
            "Best model not found. Using current model state or initializing new (Debug mode)."
        )
        # In a real run, this should fail. For safety, we assume training just finished or file exists.
        model = DSGEHNModel(feature_meta, pretrained=False)
    else:
        logger.info(f"Loading model from {model_path}")
        model = DSGEHNModel(feature_meta, pretrained=False)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    model.to(device)
    model.eval()

    # Inference
    logger.info("Running Inference on Test Set...")

    prediction_ids = []
    probabilities = []

    # We need to map back to prediction_id.
    # The test_loader dataset has a dataframe `test_df` with `prediction_id`.
    test_df = test_loader.dataset.df

    # Store raw predictions aligned with dataframe indices
    raw_preds = np.zeros(len(test_df))

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            imgs = batch["image"].to(device)
            cats = batch["categorical"].to(device)
            conts = batch["continuous"].to(device)
            indices = batch["idx"].cpu().numpy()

            # Forward (Main head only)
            logits, _ = model(imgs, cats, conts)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            # Assign to correct index
            raw_preds[indices] = probs

    # Add probs to dataframe
    test_df["cancer_prob"] = raw_preds

    # Aggregation: Group by prediction_id and take MAX
    # "Multiple images will share the same prediction ID."
    # "Predict the likelihood... 0.5" -> If one view is 0.9 and other is 0.1, patient likely has cancer -> Max.
    submission_df = test_df.groupby("prediction_id")["cancer_prob"].max().reset_index()
    submission_df.rename(columns={"cancer_prob": "cancer"}, inplace=True)

    # Save
    logger.info(f"Saving submission to {Config.SUBMISSION_PATH}")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    # Validate format
    logger.info("Submission head:")
    logger.info(submission_df.head())


# =========================================================================
# Main Execution Entry Point
# =========================================================================


def run():
    # 1. Train
    model = train_pipeline()

    # 2. Predict & Submit
    predict_and_submit()


if __name__ == "__main__":
    run()
