import os
import math
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.config import Config
from library.utils import AverageMeter, mapk
from library.dataset import (
    HotelDataset,
    get_transforms,
    process_data,
    BalanceClassSampler,
)

# -------------------------------------------------------------------------
# Neural Network Architecture
# -------------------------------------------------------------------------


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM).
    Computes the generalized mean of each channel in the feature map.
    p > 1: Focuses on salient features (similar to MaxPool).
    p -> infinity: MaxPool.
    p = 1: AvgPool.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (B, C, H, W)
        # Clamp to avoid NaN in pow
        x = x.clamp(min=eps)
        # Average pooling on x^p
        x = F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1)))
        # Raise to 1/p
        return x.pow(1.0 / p)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class ArcMarginProduct(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) Head.
    """

    def __init__(
        self,
        in_features,
        out_features,
        s=30.0,
        m=0.50,
        easy_margin=False,
        ls_eps=0.0,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # Label smoothing epsilon (not used in standard ArcFace but good to have)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # --------------------------- cosine ---------------------------
        # Normalize input and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # If no label is provided (Inference), return scaled cosine similarity
        if label is None:
            return cosine * self.s

        # --------------------------- arcface ---------------------------
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), device=Config.DEVICE)
        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Efficient implementation without full one-hot tensor
        # Gather the cosine values corresponding to the ground truth classes
        # This part is tricky to vectorize purely with gather for the update,
        # so scatter is often used or advanced indexing.

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Calculate output: where one_hot is 1, use phi; else use cosine
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale
        output *= self.s

        return output


class HotelIdModel(nn.Module):
    """
    Main Model Class: EfficientNet Backbone + GeM Pooling + ArcFace Head.
    """

    def __init__(
        self,
        num_classes,
        backbone_name=Config.BACKBONE_NAME,
        embedding_size=Config.EMBEDDING_SIZE,
        pretrained=True,
    ):
        super(HotelIdModel, self).__init__()

        # Backbone
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained)

        # Determine input features for the neck
        # EfficientNet usually has .classifier, others might have .fc or .head
        if hasattr(self.backbone, "classifier"):
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, "fc"):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, "head"):
            # For some timm models
            in_features = (
                self.backbone.head.fc.in_features
                if hasattr(self.backbone.head, "fc")
                else self.backbone.num_features
            )
            self.backbone.head = nn.Identity()
        else:
            in_features = self.backbone.num_features

        # Remove global pooling from backbone to use GeM
        self.backbone.global_pool = nn.Identity()

        # Pooling
        self.gem = GeM()

        # Neck (Embedding Layer)
        self.neck = nn.Sequential(
            nn.Linear(in_features, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.PReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
        )

        # Head (ArcFace)
        self.head = ArcMarginProduct(
            embedding_size,
            num_classes,
            s=Config.ARCFACE_SCALE,
            m=Config.ARCFACE_MARGIN,
        )

    def forward(self, x, labels=None):
        # Backbone features: (B, C, H, W)
        features = self.backbone.forward_features(x)

        # Pooling: (B, C, 1, 1) -> (B, C)
        pooled_features = self.gem(features).flatten(1)

        # Embedding: (B, Emb)
        embeddings = self.neck(pooled_features)

        # ArcFace Head: (B, NumClasses)
        logits = self.head(embeddings, labels)

        return logits


# -------------------------------------------------------------------------
# Training & Evaluation Logic
# -------------------------------------------------------------------------


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    losses = AverageMeter()

    # Progress bar
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        # ArcFace requires labels during training
        outputs = model(images, labels)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))
        pbar.set_postfix(loss=losses.avg, lr=optimizer.param_groups[0]["lr"])

    # Step scheduler at the end of epoch
    if scheduler:
        scheduler.step()

    return losses.avg


def validate(model, loader, device, num_classes):
    model.eval()

    # Store predictions and targets for MAP@5 calculation
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc="[Val]", leave=False)

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)

            # Forward pass (Inference mode, labels=None)
            # Returns cosine similarities * s
            outputs = model(images, labels=None)

            # Get top 5 predictions
            # outputs shape: (Batch, NumClasses)
            _, topk_indices = torch.topk(outputs, Config.TOP_K, dim=1)

            all_preds.extend(topk_indices.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist())

    # Calculate MAP@5
    # targets are single integers, preds are lists of 5 integers
    # mapk expects list of ground truths (which can be lists), so we wrap targets
    all_targets_wrapped = [[t] for t in all_targets]
    score = mapk(all_targets_wrapped, all_preds, k=Config.TOP_K)

    return score


def inference(model, loader, device, label_encoder_classes):
    model.eval()
    results = []

    pbar = tqdm(loader, desc="[Inference]", leave=False)

    with torch.no_grad():
        for images, _ in pbar:
            images = images.to(device)

            # TTA: Original + Horizontal Flip
            if Config.USE_TTA:
                # Original
                out1 = model(images, labels=None)

                # Flip
                images_flipped = torch.flip(images, dims=[3])
                out2 = model(images_flipped, labels=None)

                outputs = (out1 + out2) / 2.0
            else:
                outputs = model(images, labels=None)

            # Get top 5
            _, topk_indices = torch.topk(outputs, Config.TOP_K, dim=1)
            topk_indices = topk_indices.cpu().numpy()

            # Decode labels
            for indices in topk_indices:
                decoded_labels = label_encoder_classes[indices]
                # Join with spaces
                prediction_str = " ".join(map(str, decoded_labels))
                results.append(prediction_str)

    return results


def run():
    """
    Main execution function.
    """
    print(f"Using device: {Config.DEVICE}")

    # 1. Data Preparation
    print("Processing metadata...")
    train_df, val_df, test_df, num_classes = process_data(load_cached_data=True)
    print(f"Number of classes: {num_classes}")

    # Load Label Encoder for inference decoding
    encoder_path = os.path.join(Config.WORKING_DIR, "label_encoder.npy")
    label_encoder_classes = np.load(encoder_path, allow_pickle=True)

    # Datasets
    train_dataset = HotelDataset(
        train_df, transforms=get_transforms("train"), root_dir=Config.INPUT_DIR
    )
    val_dataset = HotelDataset(
        val_df, transforms=get_transforms("val"), root_dir=Config.INPUT_DIR
    )
    test_dataset = HotelDataset(
        test_df,
        transforms=get_transforms("test"),
        root_dir=Config.INPUT_DIR,
        is_test=True,
    )

    # Sampler
    # Ensure we don't request more classes than available
    classes_per_batch = min(Config.CLASSES_PER_BATCH, num_classes)
    train_sampler = BalanceClassSampler(
        train_df["label"].values,
        classes_per_batch=classes_per_batch,
        samples_per_class=Config.SAMPLES_PER_CLASS,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Model Initialization
    model = HotelIdModel(num_classes=num_classes).to(Config.DEVICE)

    # 3. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    best_map = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, Config.DEVICE, epoch
        )

        # Validate
        val_map = validate(model, val_loader, Config.DEVICE, num_classes)

        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MAP@5: {val_map:.6f}"
        )

        # Checkpointing & Early Stopping
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New Best MAP@5: {best_map:.6f}. Model saved.")
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"  >>> No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 5. Inference
    print("Starting inference on test set...")
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))

    predictions = inference(model, test_loader, Config.DEVICE, label_encoder_classes)

    # 6. Submission
    print("Generating submission file...")
    submission_df = test_df[["image"]].copy()
    submission_df["hotel_id"] = predictions

    submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
