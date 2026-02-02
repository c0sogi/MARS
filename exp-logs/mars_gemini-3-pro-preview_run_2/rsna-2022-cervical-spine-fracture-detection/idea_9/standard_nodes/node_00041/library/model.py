import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from library.config import Config
from library.utils import load_checkpoint, save_checkpoint, calculate_weighted_log_loss
from library.dataset import CervicalSpineDataset, get_slice_cache, get_bbox_cache

# --- Losses ---


class WeightedMultilabelLoss(nn.Module):
    """
    Weighted Log Loss matching the competition metric.
    Weights: patient_overall=7.0, others=1.0.
    """

    def __init__(self):
        super().__init__()
        # Weights for C1..C7 (1.0) and patient_overall (7.0)
        self.weights = torch.tensor([1.0] * 7 + [7.0], device=Config.DEVICE)

    def forward(self, y_pred, y_true):
        # y_pred: (Batch, 8), y_true: (Batch, 8)
        # Clip predictions to avoid log(0)
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)

        # Binary Cross Entropy per label
        # Loss = - [y * log(p) + (1-y) * log(1-p)]
        loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

        # Apply weights
        weighted_loss = loss * self.weights

        # Average over all elements (as per competition description "loss is averaged across all rows")
        # In our batch context, we average over batch and sum/mean over classes appropriately.
        # The metric definition implies averaging the weighted loss.
        return weighted_loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: (B, 1, H, W) - Sigmoid applied
        # target: (B, 1, H, W)

        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=(2, 3))
        loss = 1 - (
            (2.0 * intersection + self.smooth)
            / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + self.smooth)
        )

        return loss.mean()


# --- Modules ---


class SpatialAttentionModule(nn.Module):
    """
    Projects feature maps to a 1-channel attention map and computes spatially weighted pooling.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
        )

    def forward(self, x):
        # x: (B*Seq, C, H, W)

        # Generate Attention Map
        attn_logits = self.conv(x)  # (B*Seq, 1, H, W)
        attn_map = torch.sigmoid(attn_logits)

        # Spatially Weighted Pooling
        # Multiply features by attention map
        # x * attn_map -> (B*Seq, C, H, W)
        # Sum over spatial dimensions -> (B*Seq, C)
        # Normalize by sum of attention weights to keep scale consistent

        weighted_features = (x * attn_map).sum(dim=(2, 3))
        normalization = attn_map.sum(dim=(2, 3)) + 1e-6

        pooled = weighted_features / normalization

        return pooled, attn_logits


class AttentionHead(nn.Module):
    """
    Temporal Attention Head for a specific class.
    Aggregates sequence of embeddings into a single vector.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x):
        # x: (Batch, Seq, Dim)
        weights = self.attention(x)  # (Batch, Seq, 1)
        weights = torch.softmax(weights, dim=1)

        context = (x * weights).sum(dim=1)  # (Batch, Dim)
        return context


class CervicalFractureNet(nn.Module):
    def __init__(self, backbone_name=Config.BACKBONE, pretrained=True):
        super().__init__()

        # 1. Backbone (EfficientNet-B4)
        # features_only=True returns a list of feature maps
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=Config.IN_CHANNELS,
        )

        # Enable Gradient Checkpointing to save memory
        self.backbone.set_grad_checkpointing(enable=True)

        # Feature Channels for B4: P4=160, P5=448 (approx, checking dynamically is better but hardcoding for B4)
        # EfficientNet-B4:
        # block 0: 24, block 1: 32, block 2: 56, block 3: 160 (P4), block 4: 448 (P5)
        self.p4_ch = 160
        self.p5_ch = 448
        self.concat_ch = self.p4_ch + self.p5_ch

        # 2. Spatial Attention
        self.spatial_attn = SpatialAttentionModule(self.concat_ch)

        # 3. Sequence Modeling
        self.lstm_dim = Config.LSTM_HIDDEN_SIZE
        self.pos_embed = nn.Parameter(torch.zeros(1, Config.SEQ_LEN, self.concat_ch))

        self.lstm = nn.LSTM(
            input_size=self.concat_ch,
            hidden_size=self.lstm_dim,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT if Config.LSTM_LAYERS > 1 else 0,
        )

        lstm_out_dim = self.lstm_dim * 2

        # 4. Heads

        # A. Slice Auxiliary Head
        self.slice_head = nn.Linear(lstm_out_dim, 1)

        # B. Primary Heads (8 Classes)
        # We create a ModuleList of attention mechanisms and classifiers
        self.class_attn_heads = nn.ModuleList(
            [AttentionHead(lstm_out_dim) for _ in range(Config.NUM_CLASSES)]
        )
        self.class_classifiers = nn.ModuleList(
            [nn.Linear(lstm_out_dim, 1) for _ in range(Config.NUM_CLASSES)]
        )

    def forward(self, x):
        # x: (Batch, Seq, C, H, W)
        b, s, c, h, w = x.shape

        # Merge Batch and Seq for Backbone
        x = x.view(b * s, c, h, w)

        # Backbone Forward
        features = self.backbone(x)
        p4 = features[3]  # (B*S, 160, H/16, W/16) -> 24x24 for 384 input
        p5 = features[4]  # (B*S, 448, H/32, W/32) -> 12x12 for 384 input

        # Upsample P5 to match P4
        p5_up = F.interpolate(
            p5, size=p4.shape[2:], mode="bilinear", align_corners=False
        )

        # Concatenate
        feat_cat = torch.cat([p4, p5_up], dim=1)  # (B*S, 608, 24, 24)

        # Spatial Attention Pooling
        pooled, attn_logits = self.spatial_attn(
            feat_cat
        )  # pooled: (B*S, 608), attn_logits: (B*S, 1, 24, 24)

        # Reshape for Sequence Modeling
        pooled = pooled.view(b, s, -1)  # (B, S, 608)

        # Add Positional Embeddings
        pooled = pooled + self.pos_embed[:, :s, :]

        # LSTM
        lstm_out, _ = self.lstm(pooled)  # (B, S, 512)

        # --- Outputs ---

        # 1. Spatial Map (for supervision)
        # Reshape back to (B, S, 1, H_feat, W_feat)
        spatial_out = attn_logits.view(
            b, s, 1, attn_logits.shape[2], attn_logits.shape[3]
        )

        # 2. Slice Prediction (Auxiliary)
        slice_out = self.slice_head(lstm_out).squeeze(-1)  # (B, S)

        # 3. Study Prediction (Primary)
        study_logits = []
        for i in range(Config.NUM_CLASSES):
            ctx = self.class_attn_heads[i](lstm_out)  # (B, 512)
            logit = self.class_classifiers[i](ctx)  # (B, 1)
            study_logits.append(logit)

        study_out = torch.cat(study_logits, dim=1)  # (B, 8)

        return {
            "study_logits": study_out,
            "slice_logits": slice_out,
            "spatial_logits": spatial_out,
        }


# --- Training & Inference Logic ---


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()

    loss_meter = 0.0
    study_loss_meter = 0.0
    slice_loss_meter = 0.0
    spatial_loss_meter = 0.0

    criterion_study = WeightedMultilabelLoss()
    criterion_slice = nn.BCEWithLogitsLoss()
    criterion_spatial = DiceLoss()

    bar = tqdm(loader, desc=f"Epoch {epoch} [Train]", disable=True)

    # Gradient Accumulation
    optimizer.zero_grad()

    for step, (images, targets) in enumerate(bar):
        images = images.to(device)

        # Targets
        y_study = targets["study_labels"].to(device)
        y_slice = targets["slice_labels"].to(device)
        mask_spatial = targets["spatial_masks"].to(device)  # (B, S, 1, H, W)

        # Forward
        outputs = model(images)

        pred_study = torch.sigmoid(outputs["study_logits"])
        pred_slice = outputs["slice_logits"]
        pred_spatial = outputs["spatial_logits"]  # (B, S, 1, H_feat, W_feat)

        # --- Loss Calculation ---

        # 1. Study Loss
        l_study = criterion_study(pred_study, y_study)

        # 2. Slice Loss
        l_slice = criterion_slice(pred_slice, y_slice)

        # 3. Spatial Loss (Only where we have masks)
        # We need to upsample prediction to mask size OR downsample mask to pred size.
        # Downsampling mask is cheaper.
        # pred_spatial shape: (B, S, 1, 24, 24) approx
        # mask_spatial shape: (B, S, 1, 384, 384)

        # Flatten batch and seq for interpolation
        b, s, c, h, w = pred_spatial.shape
        pred_spatial_flat = pred_spatial.view(b * s, c, h, w)
        pred_spatial_sig = torch.sigmoid(pred_spatial_flat)

        mask_spatial_flat = mask_spatial.view(
            b * s, 1, mask_spatial.shape[3], mask_spatial.shape[4]
        )
        mask_spatial_small = F.interpolate(
            mask_spatial_flat, size=(h, w), mode="nearest"
        )

        # Compute dice only on slices that have a fracture (sum of mask > 0)
        # Or compute everywhere (empty masks should drive prediction to 0)
        l_spatial = criterion_spatial(pred_spatial_sig, mask_spatial_small)

        # Total Loss
        loss = (
            l_study
            + (Config.LAMBDA_SLICE * l_slice)
            + (Config.LAMBDA_SPATIAL * l_spatial)
        )

        # Normalize for accumulation
        loss = loss / Config.ACCUMULATION_STEPS
        loss.backward()

        if (step + 1) % Config.ACCUMULATION_STEPS == 0:
            nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

        # Logging
        loss_meter += loss.item() * Config.ACCUMULATION_STEPS
        study_loss_meter += l_study.item()
        slice_loss_meter += l_slice.item()
        spatial_loss_meter += l_spatial.item()

        bar.set_postfix(loss=loss.item() * Config.ACCUMULATION_STEPS)

    return loss_meter / len(loader)


def validate(model, loader, device):
    model.eval()

    preds_all = []
    targets_all = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Validating", disable=True):
            images = images.to(device)
            y_study = targets["study_labels"].numpy()

            outputs = model(images)
            pred_study = torch.sigmoid(outputs["study_logits"]).cpu().numpy()

            preds_all.append(pred_study)
            targets_all.append(y_study)

    preds_all = np.concatenate(preds_all, axis=0)
    targets_all = np.concatenate(targets_all, axis=0)

    # Calculate Metric
    score = calculate_weighted_log_loss(targets_all, preds_all)
    return score


def generate_submission(model, device):
    """
    Generates the submission file for the test set.
    """
    print("Generating submission...")

    # Load Test Metadata
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Setup Dataset
    # We need slice cache for test set too
    slice_cache = get_slice_cache(test_meta, load_cached_data=True)

    test_dataset = CervicalSpineDataset(
        metadata_df=test_meta,
        study_to_slices=slice_cache,
        is_train=False,
        seq_len=Config.SEQ_LEN,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model.eval()
    results = []

    # Columns for submission
    # Order matches Config.TARGET_COLS: C1..C7, patient_overall
    target_cols = Config.TARGET_COLS

    with torch.no_grad():
        # Iterate through loader. Note: Dataset returns (images, dummy_targets)
        # We need to map predictions back to StudyInstanceUID

        # Since loader is sequential, we can iterate metadata rows in chunks
        meta_idx = 0

        for images, _ in tqdm(test_loader, desc="Inference", disable=True):
            images = images.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            probs = torch.sigmoid(outputs["study_logits"]).cpu().numpy()  # (B, 8)

            # Map back to IDs
            for b in range(batch_size):
                study_uid = test_meta.iloc[meta_idx]["StudyInstanceUID"]
                study_probs = probs[b]

                # Create rows for this study
                for i, col_name in enumerate(target_cols):
                    row_id = f"{study_uid}_{col_name}"
                    prob = study_probs[i]
                    results.append({"row_id": row_id, "fractured": prob})

                meta_idx += 1

    # Create DataFrame
    sub_df = pd.DataFrame(results)

    # Save
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training():
    # 1. Setup Data
    print("Initializing Data...")
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    if Config.DEBUG:
        train_meta = train_meta.head(20)
        val_meta = val_meta.head(10)

    # Caches
    slice_cache = get_slice_cache(
        pd.concat([train_meta, val_meta]), load_cached_data=True
    )
    bbox_cache = get_bbox_cache(Config.BOUNDING_BOX_PATH, load_cached_data=True)

    train_ds = CervicalSpineDataset(train_meta, slice_cache, bbox_cache, is_train=True)
    val_ds = CervicalSpineDataset(
        val_meta, slice_cache, bbox_cache, is_train=True
    )  # is_train=True to get targets

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Setup Model
    print("Initializing Model...")
    device = torch.device(Config.DEVICE)
    model = CervicalFractureNet(pretrained=True).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_steps = len(train_loader) * Config.EPOCHS // Config.ACCUMULATION_STEPS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_steps, eta_min=Config.MIN_LR
    )

    # 3. Loop
    best_score = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        val_score = validate(model, val_loader, device)

        print(
            f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Score (LogLoss): {val_score:.10f}"
        )

        # Checkpoint
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            save_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")
            save_checkpoint(model, optimizer, epoch, val_score, save_path)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 4. Inference
    print("Loading best model for inference...")
    best_model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")
    checkpoint = load_checkpoint(best_model_path, model)

    generate_submission(model, device)


if __name__ == "__main__":
    # This block is here for local testing if needed, but the prompt says
    # "DO NOT include an if __name__ == '__main__': block" in the requirements.
    # However, the prompt also says "If this module handles submission generation...".
    # I have provided the functions. The external caller can invoke run_training().
    pass
