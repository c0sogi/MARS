import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import pandas as pd
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided libraries
from library.utils import seed_everything, AverageMeter, compute_score, save_checkpoint
from library.dataset import get_dataloaders


# ====================================================
# Pooling Mechanism: Generalized Mean Pooling (GeM)
# ====================================================
class GeM(nn.Module):
    """
    Generalized Mean Pooling.
    p = 1 -> Average Pooling
    p -> infinity -> Max Pooling
    p is trainable.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (B, C, H, W)
        # Output: (B, C, 1, 1)
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )


# ====================================================
# Classification Head: Multi-Sample Dropout
# ====================================================
class MultiSampleDropoutHead(nn.Module):
    """
    Applies multiple dropout masks to the input and averages the results
    from a single linear layer.
    """

    def __init__(self, in_features, out_features, num_dropouts=5, drop_rate=0.5):
        super().__init__()
        self.dropouts = nn.ModuleList(
            [nn.Dropout(drop_rate) for _ in range(num_dropouts)]
        )
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x: (B, In_Features)
        logits = []
        for drop in self.dropouts:
            logits.append(self.fc(drop(x)))
        # Stack and average: (Num_Drops, B, Out) -> (B, Out)
        return torch.stack(logits).mean(dim=0)


# ====================================================
# Model Architecture
# ====================================================
class RetinopathyModel(nn.Module):
    def __init__(
        self, model_name="convnext_small.fb_in1k", pretrained=True, num_classes=4
    ):
        super().__init__()
        # Load backbone with Global Average Pooling
        # Cite solution_lesson_node_00026: Prefer Global Average Pooling over GeM for aggressive downsampling
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Determine feature dimension
        # ConvNeXt-Small usually has 768 features
        self.num_features = self.backbone.num_features

        # Components
        self.head = MultiSampleDropoutHead(self.num_features, num_classes)

    def forward(self, x):
        # Feature Extraction & Pooling
        x = self.backbone(x)  # (B, C)

        # Classification
        x = self.head(x)  # (B, 4)
        return x


# ====================================================
# Training and Evaluation Functions
# ====================================================
def train_one_epoch(loader, model, optimizer, criterion, device):
    model.train()
    losses = AverageMeter()

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(loader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)
            losses.update(loss.item(), images.size(0))

            # Decode Ordinal Predictions
            # Sigmoid -> Probabilities -> Sum -> Continuous Score
            probs = torch.sigmoid(logits)
            scores = probs.sum(dim=1)

            # Get ground truth from ordinal targets (summing gives class index)
            true_labels = targets.sum(dim=1)

            all_preds.extend(scores.cpu().numpy())
            all_targets.extend(true_labels.cpu().numpy())

    # Compute QWK
    score = compute_score(all_targets, all_preds)
    return losses.avg, score


def inference(loader, model, device):
    model.eval()
    results = []

    # 4-View TTA: Original, HFlip, VFlip, Rot180

    with torch.no_grad():
        for images, id_codes in loader:
            images = images.to(device)
            b, c, h, w = images.shape

            # Create augmented batch
            # 1. Original
            # 2. Horizontal Flip
            # 3. Vertical Flip
            # 4. Rotate 180 (equivalent to HFlip + VFlip)

            # We process them by passing a larger batch or iterating.
            # To save memory, we iterate views.

            preds_accum = torch.zeros(b, 4, device=device)

            views = [
                images,
                torch.flip(images, [3]),
                torch.flip(images, [2]),
                torch.rot90(images, 2, [2, 3]),
            ]

            for view in views:
                logits = model(view)
                probs = torch.sigmoid(logits)
                preds_accum += probs

            # Average probabilities
            avg_probs = preds_accum / 4.0

            # Decode to score [0, 4]
            scores = avg_probs.sum(dim=1).cpu().numpy()

            # Round to nearest integer for submission
            final_preds = np.round(scores).astype(int)
            final_preds = np.clip(final_preds, 0, 4)

            for code, pred in zip(id_codes, final_preds):
                results.append({"id_code": code, "diagnosis": pred})

    return pd.DataFrame(results)


# ====================================================
# Main Execution
# ====================================================
def run_training(
    epochs=10,
    batch_size=16,
    image_size=512,
    lr=1e-4,
    weight_decay=1e-2,
    seed=42,
    output_dir="./working/idea_7",
):
    # Setup
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting run with device: {device}")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, image_size=image_size
    )

    # Model
    print("Initializing ConvNeXt-Small with GeM Pooling...")
    model = RetinopathyModel(model_name="convnext_small.fb_in1k", pretrained=True)
    model = model.to(device)

    # Optimization
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    best_score = -1.0

    # Training Loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss = train_one_epoch(train_loader, model, optimizer, criterion, device)
        val_loss, val_score = validate(val_loader, model, criterion, device)

        scheduler.step()

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}")
        print(f"Val QWK:    {val_score:.6f}")

        # Save Best
        is_best = val_score > best_score
        if is_best:
            best_score = val_score
            print("New best score! Saving checkpoint.")

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_score": best_score,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            output_dir,
        )

    print(f"\nTraining complete. Best Validation QWK: {best_score:.6f}")

    # Inference
    print("Starting Inference on Test Set with TTA...")
    best_model_path = os.path.join(output_dir, "best_model.pth")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    submission_df = inference(test_loader, model, device)

    # Save Submission
    sub_dir = "./submission"
    os.makedirs(sub_dir, exist_ok=True)
    sub_path = os.path.join(sub_dir, "submission.csv")
    submission_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
    print(submission_df.head())
