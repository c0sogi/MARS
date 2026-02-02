import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from transformers import get_cosine_schedule_with_warmup

from library.config import Config
from library.data import get_dataloaders
from library.loss import FocalLoss
from library.utils import AverageMeter, get_roc_auc, seed_everything


class DeepHybridEfficientNet(nn.Module):
    """
    Deep Hybrid EfficientNet Architecture.

    Visual Backbone: EfficientNet-B0 (Pretrained, Unfrozen)
    Fusion: Concatenation of visual embeddings and metadata features.
    Head: MLP Projection Head (Linear -> ReLU -> Dropout -> Linear).
    """

    def __init__(
        self,
        meta_dim: int,
        model_name: str = Config.MODEL_NAME,
        pretrained: bool = Config.PRETRAINED,
        fusion_hidden_dim: int = Config.FUSION_HIDDEN_DIM,
        dropout_rate: float = Config.DROPOUT_RATE,
    ):
        super(DeepHybridEfficientNet, self).__init__()

        # 1. Visual Backbone
        # num_classes=0 returns the pooled features (Global Average Pooling output)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        self.n_visual_features = (
            self.backbone.num_features
        )  # e.g., 1280 for EfficientNet-B0

        # 2. Fusion Head
        input_dim = self.n_visual_features + meta_dim

        self.head = nn.Sequential(
            nn.Linear(input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_dim, 1),  # Output logits
        )

    def forward(self, images, meta):
        """
        Args:
            images: (Batch, 3, H, W)
            meta: (Batch, Meta_Dim)
        Returns:
            logits: (Batch, 1)
        """
        # Extract visual features
        visual_feats = self.backbone(images)  # (Batch, n_visual_features)

        # Concatenate with metadata
        combined_feats = torch.cat([visual_feats, meta], dim=1)

        # Pass through MLP head
        logits = self.head(combined_feats)

        return logits


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    loss_meter = AverageMeter()

    # Store predictions and targets for AUC calculation
    all_preds = []
    all_targets = []

    for batch_idx, (images, meta, targets) in enumerate(loader):
        images = images.to(device)
        meta = meta.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(images, meta)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_meter.update(loss.item(), images.size(0))

        # Apply sigmoid for metrics
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(targets.cpu().numpy())

    auc = get_roc_auc(all_targets, all_preds)
    return loss_meter.avg, auc


def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, meta, targets in loader:
            images = images.to(device)
            meta = meta.to(device)
            targets = targets.to(device)

            logits = model(images, meta)
            loss = criterion(logits, targets)

            loss_meter.update(loss.item(), images.size(0))

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy())

    auc = get_roc_auc(all_targets, all_preds)
    return loss_meter.avg, auc


def predict(model, loader, device):
    model.eval()
    all_preds = []
    image_names = []

    # We need image names for submission.
    # The loader returns (image, meta, target).
    # We can retrieve image_names from the dataset inside the loader.
    dataset = loader.dataset
    df = dataset.df

    with torch.no_grad():
        for images, meta, _ in loader:
            images = images.to(device)
            meta = meta.to(device)

            logits = model(images, meta)
            probs = torch.sigmoid(logits).cpu().numpy()

            # Flatten predictions
            all_preds.extend(probs.flatten())

    return df["image_name"].values, np.array(all_preds)


def run_training():
    """
    Main execution function to train the model and generate submission.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Using device: {device}")

    # 1. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
        debug=Config.DEBUG,
    )

    # Determine metadata dimension from a sample batch
    dummy_meta = next(iter(train_loader))[1]
    meta_dim = dummy_meta.shape[1]
    print(f"Metadata Dimension: {meta_dim}")

    # 2. Model Initialization
    print(f"Initializing {Config.MODEL_NAME}...")
    model = DeepHybridEfficientNet(meta_dim=meta_dim).to(device)

    # 3. Setup Training Components
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Cosine with Warmup
    num_training_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = len(train_loader) * Config.WARMUP_EPOCHS

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 4. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"  Train Loss: {train_loss:.6f} | Train AUC: {train_auc:.6f}")
        print(f"  Val Loss:   {val_loss:.6f} | Val AUC:   {val_auc:.6f}")

        # Checkpointing and Early Stopping
        if val_auc > best_auc + Config.EARLY_STOPPING_MIN_DELTA:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            print(f"  New best model saved! (AUC: {best_auc:.6f})")
        else:
            patience_counter += 1
            print(
                f"  No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # 5. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device))

    print("Generating predictions on test set...")
    image_names, predictions = predict(model, test_loader, device)

    # 6. Submission
    submission_df = pd.DataFrame({"image_name": image_names, "target": predictions})

    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
