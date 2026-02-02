import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from library.config import Config
from library.utils import mapk

# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (B, C, H, W)
        # Apply clamping to avoid NaN with pow
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
        self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # input is (B, EmbeddingSize), weight is (NumClasses, EmbeddingSize)
        # Both should be normalized
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # We only modify the logits corresponding to the ground truth labels
        # Create a mask
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # Combine: use phi for target class, cosine for others
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


# --------------------------------------------------------------------------
# Model Architecture
# --------------------------------------------------------------------------


class HotelRecognitionModel(nn.Module):
    def __init__(
        self,
        n_classes=Config.NUM_CLASSES,
        model_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        embedding_size=Config.EMBEDDING_SIZE,
    ):
        super(HotelRecognitionModel, self).__init__()

        # Load Backbone
        # global_pool='' ensures we get spatial features (B, C, H, W)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine input features from backbone
        # For ConvNeXt-Tiny, this is typically 768
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            in_features = features.shape[1]

        self.pooling = GeM()

        # Neck: Linear Projection + Batch Normalization
        self.fc = nn.Linear(in_features, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

        # Head: ArcFace
        self.arcface = ArcMarginProduct(
            in_features=embedding_size,
            out_features=n_classes,
            s=Config.SCALE,
            m=Config.MARGIN,
        )

        # Initialization
        self._init_params()

    def _init_params(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, images, labels=None):
        # Feature Extraction
        features = self.backbone(images)  # (B, C, H, W)

        # Aggregation
        pooled = self.pooling(features)  # (B, C, 1, 1)
        pooled = pooled.flatten(1)  # (B, C)

        # Projection & Neck
        embedding = self.fc(pooled)
        embedding = self.bn(embedding)

        # Training vs Inference
        if labels is not None:
            # During training, return ArcFace logits
            return self.arcface(embedding, labels)
        else:
            # During inference, return embeddings
            return embedding

    def get_class_centers(self):
        """Returns the normalized class centers from the ArcFace head."""
        return F.normalize(self.arcface.weight)


# --------------------------------------------------------------------------
# Training & Evaluation Functions
# --------------------------------------------------------------------------


def train_fn(dataloader, model, criterion, optimizer, device, scheduler=None, epoch=0):
    model.train()
    scaler = GradScaler()

    running_loss = 0.0
    dataset_size = 0

    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images, labels)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD_NORM)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def eval_fn(dataloader, model, device, classes):
    """
    Evaluates the model on the validation set using MAP@5.

    Args:
        dataloader: Validation dataloader
        model: The model
        device: 'cuda' or 'cpu'
        classes: Array of hotel_ids (strings or ints) corresponding to class indices

    Returns:
        avg_loss (float): Validation loss (ArcFace loss)
        map5 (float): Mean Average Precision @ 5
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    dataset_size = 0

    # Store predictions and ground truths for MAP calculation
    all_preds = []
    all_targets = []

    # Pre-compute normalized class centers for fast similarity search
    # shape: (NumClasses, EmbeddingSize)
    class_centers = model.get_class_centers().detach()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            # 1. Compute Loss (requires passing labels to forward to get ArcFace logits)
            outputs = model(images, labels)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # 2. Compute Predictions for MAP@5
            # Get embeddings directly (pass labels=None)
            embeddings = model(images, labels=None)  # (B, EmbeddingSize)
            embeddings = F.normalize(embeddings)

            # Cosine Similarity: (B, Embed) @ (NumClasses, Embed).T -> (B, NumClasses)
            sims = torch.matmul(embeddings, class_centers.T)

            # Get Top 5 indices
            _, topk_indices = torch.topk(sims, k=5, dim=1)

            # Convert tensors to lists
            topk_indices = topk_indices.cpu().numpy()
            targets = labels.cpu().numpy()

            # For MAP@5, we need lists of ground truth and predictions
            # Since we are validating, we can just use the class indices 0..N-1
            for i in range(batch_size):
                all_preds.append(topk_indices[i].tolist())
                all_targets.append([targets[i]])  # apk expects a list of ground truths

    avg_loss = running_loss / dataset_size

    # Compute MAP@5
    map5 = mapk(all_targets, all_preds, k=5)

    print(f"Validation Loss: {avg_loss:.6f} | MAP@5: {map5:.6f}")

    return avg_loss, map5


def generate_submission(dataloader, model, device, classes, output_file):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()

    image_ids = []
    predictions_str = []

    # Pre-compute normalized class centers
    class_centers = model.get_class_centers().detach()

    print("Generating predictions...")
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)

            # Get embeddings
            embeddings = model(images, labels=None)
            embeddings = F.normalize(embeddings)

            # Cosine Similarity
            sims = torch.matmul(embeddings, class_centers.T)

            # Get Top 5 indices
            _, topk_indices = torch.topk(sims, k=5, dim=1)
            topk_indices = topk_indices.cpu().numpy()

            # Map indices back to hotel_ids
            for i in range(len(topk_indices)):
                indices = topk_indices[i]
                # Map index -> hotel_id
                pred_hotel_ids = [str(classes[idx]) for idx in indices]
                predictions_str.append(" ".join(pred_hotel_ids))

    # The test dataloader in this pipeline preserves order if shuffle=False
    # We need to get the image IDs from the dataset dataframe
    test_df = dataloader.dataset.df
    image_ids = test_df["image"].tolist()

    # Create DataFrame
    submission_df = pd.DataFrame({"image": image_ids, "hotel_id": predictions_str})

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    submission_df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
