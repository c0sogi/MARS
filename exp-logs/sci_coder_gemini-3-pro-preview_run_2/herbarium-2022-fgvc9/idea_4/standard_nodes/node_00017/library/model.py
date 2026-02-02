import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import timm
from sklearn.metrics import f1_score
from tqdm import tqdm

from library.config import Config
from library.utils import get_logger
from library.dataset import get_dataloaders

logger = get_logger(__name__)


# ==========================================
# ArcFace Layer
# ==========================================
class ArcFaceLayer(nn.Module):
    """
    Implementation of ArcFace (Additive Angular Margin Loss).
    Reference: https://arxiv.org/pdf/1801.07698.pdf
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFaceLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        # Weight shape: (out_features, in_features)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute constants
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # Normalize input and weights
        # input: (B, in_features)
        # weight: (out_features, in_features)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # If inference or no label provided, return scaled cosine similarity
        if label is None:
            return cosine * self.s

        # Training phase: Apply Additive Angular Margin
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        # Numerical stability: keep phi only when theta + m < pi
        # If cos(theta) > cos(pi - m), then theta < pi - m, so theta + m < pi.
        # Otherwise, use a Taylor expansion approximation or simple penalty (cosine - mm)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Create one-hot encoding to apply margin only to the ground truth class
        # We use scatter_ to create the mask efficiently
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # This replaces the target logit with phi, keeps others as cosine
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Rescale
        output *= self.s

        return output


# ==========================================
# Cascaded Taxonomic Network
# ==========================================
class CascadedTaxonomicNetwork(nn.Module):
    """
    Hierarchical model that predicts Family -> Genus -> Species in a cascaded manner.
    """

    def __init__(self, num_species, num_families, num_genera, pretrained=True):
        super(CascadedTaxonomicNetwork, self).__init__()

        # 1. Backbone
        self.backbone = timm.create_model(
            Config.MODEL_NAME, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        self.feature_dim = self.backbone.num_features  # 1792 for EfficientNet-B4

        # 2. Family Head
        # Input: Global Features
        self.fc_family = nn.Linear(self.feature_dim, num_families)

        # 3. Genus Head
        # Input: Global Features + Family Logits
        # This conditions the genus prediction on the family prediction
        self.fc_genus = nn.Linear(self.feature_dim + num_families, num_genera)

        # 4. Species Head (Embedding + ArcFace)
        # Input: Global Features + Genus Logits
        # This conditions the species prediction on the genus prediction
        input_dim_species = self.feature_dim + num_genera

        self.embedding_layer = nn.Sequential(
            nn.Linear(input_dim_species, Config.EMBEDDING_SIZE),
            nn.BatchNorm1d(Config.EMBEDDING_SIZE),
            nn.PReLU(),
            nn.Dropout(p=0.2),
        )

        self.arcface = ArcFaceLayer(
            Config.EMBEDDING_SIZE,
            num_species,
            s=Config.ARCFACE_SCALE,
            m=Config.ARCFACE_MARGIN,
        )

    def forward(self, x, labels=None):
        # Extract features
        features = self.backbone(x)  # (B, 1792)

        # --- Family Branch ---
        family_logits = self.fc_family(features)  # (B, num_families)

        # --- Genus Branch ---
        # Concatenate features with family logits
        genus_input = torch.cat([features, family_logits], dim=1)
        genus_logits = self.fc_genus(genus_input)  # (B, num_genera)

        # --- Species Branch ---
        # Concatenate features with genus logits
        species_input = torch.cat([features, genus_logits], dim=1)
        embedding = self.embedding_layer(species_input)  # (B, 512)

        # ArcFace Head
        # Pass labels only if available (training)
        species_label = labels[0] if labels is not None else None
        species_logits = self.arcface(embedding, species_label)

        return species_logits, genus_logits, family_logits


# ==========================================
# Training & Validation Functions
# ==========================================
def train_one_epoch(model, loader, optimizer, criterion_ce, device, epoch):
    model.train()
    running_loss = 0.0
    running_acc_sp = 0.0
    running_acc_gn = 0.0
    running_acc_fm = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for images, (species_labels, genus_labels, family_labels) in pbar:
        images = images.to(device)
        species_labels = species_labels.to(device)
        genus_labels = genus_labels.to(device)
        family_labels = family_labels.to(device)

        optimizer.zero_grad()

        # Forward pass (pass tuple of labels for ArcFace handling if needed,
        # though our ArcFaceLayer only takes one label tensor, passing tuple to forward for unpacking)
        # We modified forward to take tuple, but ArcFaceLayer takes single.
        # Let's align: model.forward takes 'labels' which is expected to be a tuple or list
        # where index 0 is species.

        sp_logits, gn_logits, fm_logits = model(
            images, labels=(species_labels, genus_labels, family_labels)
        )

        # Calculate Losses
        loss_sp = criterion_ce(sp_logits, species_labels)
        loss_gn = criterion_ce(gn_logits, genus_labels)
        loss_fm = criterion_ce(fm_logits, family_labels)

        total_loss = (
            loss_sp + (Config.LAMBDA_GENUS * loss_gn) + (Config.LAMBDA_FAMILY * loss_fm)
        )

        total_loss.backward()
        optimizer.step()

        # Metrics
        running_loss += total_loss.item() * images.size(0)

        # Accuracy
        _, preds_sp = torch.max(sp_logits, 1)
        _, preds_gn = torch.max(gn_logits, 1)
        _, preds_fm = torch.max(fm_logits, 1)

        running_acc_sp += torch.sum(preds_sp == species_labels.data)
        running_acc_gn += torch.sum(preds_gn == genus_labels.data)
        running_acc_fm += torch.sum(preds_fm == family_labels.data)

        pbar.set_postfix({"loss": total_loss.item()})

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc_sp = running_acc_sp.double() / len(loader.dataset)

    return epoch_loss, epoch_acc_sp.item()


def validate(model, loader, criterion_ce, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, (species_labels, genus_labels, family_labels) in tqdm(
            loader, desc="[Val]", leave=False
        ):
            images = images.to(device)
            species_labels = species_labels.to(device)
            genus_labels = genus_labels.to(device)
            family_labels = family_labels.to(device)

            # Inference mode: labels=None ensures ArcFace returns cosine similarity
            sp_logits, gn_logits, fm_logits = model(images, labels=None)

            # For validation loss, we can't use ArcFace margin logits because we don't pass labels.
            # However, to track convergence, we can compute CE on the raw cosine logits (scaled).
            # This is not the exact training loss but a proxy.
            loss_sp = criterion_ce(sp_logits, species_labels)
            loss_gn = criterion_ce(gn_logits, genus_labels)
            loss_fm = criterion_ce(fm_logits, family_labels)

            total_loss = (
                loss_sp
                + (Config.LAMBDA_GENUS * loss_gn)
                + (Config.LAMBDA_FAMILY * loss_fm)
            )
            running_loss += total_loss.item() * images.size(0)

            _, preds = torch.max(sp_logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(species_labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return epoch_loss, macro_f1


# ==========================================
# Pipeline Execution
# ==========================================
def run_training():
    logger.info("Initializing DataLoaders...")
    train_loader, val_loader, _, meta_counts = get_dataloaders(debug=Config.DEBUG)

    num_species = Config.NUM_CLASSES
    num_families = meta_counts["num_families"]
    num_genera = meta_counts["num_genera"]

    logger.info(
        f"Model Configuration: {num_species} Species, {num_genera} Genera, {num_families} Families"
    )

    model = CascadedTaxonomicNetwork(num_species, num_families, num_genera).to(
        Config.DEVICE
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR_START, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.LR_MIN
    )

    best_f1 = 0.0
    save_path = os.path.join(Config.WORK_DIR, "best_model.pth")

    logger.info("Starting Training...")

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE, epoch
        )
        val_loss, val_f1 = validate(model, val_loader, criterion, Config.DEVICE)

        scheduler.step()
        curr_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch}/{Config.EPOCHS} | LR: {curr_lr:.2e} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} F1: {val_f1:.6f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            logger.info(f"New Best Model Saved! F1: {best_f1:.6f}")

    logger.info(f"Training Complete. Best F1: {best_f1:.6f}")
    return model, meta_counts


def generate_submission(meta_counts):
    logger.info("Generating Submission...")

    # Re-initialize model structure
    num_species = Config.NUM_CLASSES
    num_families = meta_counts["num_families"]
    num_genera = meta_counts["num_genera"]

    model = CascadedTaxonomicNetwork(num_species, num_families, num_genera).to(
        Config.DEVICE
    )

    # Load weights
    load_path = os.path.join(Config.WORK_DIR, "best_model.pth")
    if not os.path.exists(load_path):
        logger.warning("Best model not found, skipping submission generation.")
        return

    model.load_state_dict(torch.load(load_path, map_location=Config.DEVICE))
    model.eval()

    # Get Test Loader
    _, _, test_loader, _ = get_dataloaders(debug=False)  # Ensure full test set

    predictions = []
    image_ids = []

    with torch.no_grad():
        for images, ids in tqdm(test_loader, desc="[Test Prediction]"):
            images = images.to(Config.DEVICE)

            # Forward pass (inference)
            sp_logits, _, _ = model(images, labels=None)

            # Argmax
            _, preds = torch.max(sp_logits, 1)

            predictions.extend(preds.cpu().numpy())
            image_ids.extend(ids)

    # Create DataFrame
    df_sub = pd.DataFrame({"Id": image_ids, "Predicted": predictions})

    # Ensure sorting if necessary (though Id is string in metadata, submission usually expects int sorted)
    # The sample submission shows Id as int.
    try:
        df_sub["Id"] = df_sub["Id"].astype(int)
        df_sub = df_sub.sort_values("Id")
    except:
        pass  # Keep as is if conversion fails

    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")


def run():
    """
    Main entry point for the pipeline.
    """
    try:
        _, meta_counts = run_training()
        generate_submission(meta_counts)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise e
