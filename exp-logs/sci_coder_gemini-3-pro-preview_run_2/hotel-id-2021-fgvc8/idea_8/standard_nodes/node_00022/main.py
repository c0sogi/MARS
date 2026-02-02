import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, calc_map5
from library.dataset import (
    get_dataloaders,
    HotelDataset,
    get_transforms,
    get_class_mapping,
)
from library.models import HotelRecognitionModel
from library.losses import SubCenterArcFaceLoss
from library.trainer import Trainer
from library.inference import (
    extract_features,
    fuse_embeddings,
    database_augmentation,
    query_expansion,
    run_inference,
)


def train_backbone(backbone_name):
    """
    Trains a single backbone using Progressive Resolution (Phase 1 & Phase 2).
    """
    print(f"\n{'='*40}\nTraining Backbone: {backbone_name}\n{'='*40}")

    device = Config.DEVICE
    checkpoint_path = Config.get_checkpoint_path(backbone_name)

    # --- Phase 1: Low Resolution (256x256) ---
    print("\n--- Phase 1: 256x256 ---")
    p1_cfg = Config.PHASE1_CONFIG
    # Overriding epochs for fast baseline execution
    p1_epochs = 1

    train_loader_p1, val_loader_p1, _, num_classes = get_dataloaders(
        img_size=p1_cfg["img_size"],
        batch_size=p1_cfg["batch_size"],
        load_cached_data=True,
    )

    model = HotelRecognitionModel(
        backbone_name, num_classes=num_classes, embedding_dim=Config.EMBEDDING_DIM
    )
    model.to(device)

    criterion = SubCenterArcFaceLoss(label_smoothing=0.0).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=p1_cfg["lr"], weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p1_epochs)

    trainer = Trainer(model, device, optimizer, scheduler, criterion, checkpoint_path)
    trainer.fit(train_loader_p1, val_loader_p1, epochs=p1_epochs, patience=2)

    # --- Phase 2: High Resolution (384x384) ---
    print("\n--- Phase 2: 384x384 ---")
    p2_cfg = Config.PHASE2_CONFIG
    # Overriding epochs for fast baseline execution
    p2_epochs = 1

    train_loader_p2, val_loader_p2, _, _ = get_dataloaders(
        img_size=p2_cfg["img_size"],
        batch_size=p2_cfg["batch_size"],
        load_cached_data=True,
    )

    # Update optimizer for fine-tuning (lower LR)
    # Note: Trainer.fit reloads the best model from Phase 1 at the start if it exists,
    # but since we are continuing with the same model instance in memory, we just update params.
    # However, to ensure we start from the best Phase 1 weights, we reload them explicitly.
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    optimizer = optim.AdamW(
        model.parameters(), lr=p2_cfg["lr"], weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p2_epochs)

    trainer = Trainer(model, device, optimizer, scheduler, criterion, checkpoint_path)
    trainer.fit(train_loader_p2, val_loader_p2, epochs=p2_epochs, patience=2)

    return checkpoint_path


def validate_ensemble():
    """
    Performs validation on the hold-out validation set using the Dual-Backbone Ensemble.
    Calculates MAP@5 and performs failure analysis.
    """
    print(f"\n{'='*40}\nValidating Ensemble\n{'='*40}")
    device = Config.DEVICE
    img_size = Config.PHASE2_CONFIG["img_size"]  # Use high res for validation

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Get Class Mapping
    class_mapping = get_class_mapping(train_df, load_cached_data=True)

    # Prepare DataLoaders for Feature Extraction
    # Gallery = Train Set (shuffle=False)
    gallery_dataset = HotelDataset(
        train_df,
        transform=get_transforms(img_size, mode="valid"),
        mode="valid",
        class_mapping=class_mapping,
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=Config.PHASE2_CONFIG["batch_size"] * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Query = Validation Set
    query_dataset = HotelDataset(
        val_df,
        transform=get_transforms(img_size, mode="valid"),
        mode="valid",
        class_mapping=class_mapping,
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=Config.PHASE2_CONFIG["batch_size"] * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Extract Features for each backbone
    gallery_embs_dict = {}
    query_embs_dict = {}

    # Ground truth for validation
    query_labels = val_df["hotel_id"].map(class_mapping).values
    # Ground truth for gallery (used for retrieval reference if needed, but here we just need embeddings)
    gallery_labels = train_df["hotel_id"].map(class_mapping).values

    for backbone in Config.BACKBONES:
        print(f"Extracting features for {backbone}...")

        # Load Model
        model = HotelRecognitionModel(
            backbone,
            num_classes=len(class_mapping),
            embedding_dim=Config.EMBEDDING_DIM,
            pretrained=False,
        )
        checkpoint_path = Config.get_checkpoint_path(backbone)
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print(f"Warning: Checkpoint for {backbone} not found!")

        model.to(device)
        model.eval()

        # Extract
        g_emb, _ = extract_features(gallery_loader, model, device)
        q_emb, _ = extract_features(query_loader, model, device)

        gallery_embs_dict[backbone] = g_emb
        query_embs_dict[backbone] = q_emb

        # Clear memory
        del model
        torch.cuda.empty_cache()

    # Fuse Embeddings
    print("Fusing embeddings...")
    gallery_fused = fuse_embeddings(gallery_embs_dict)
    query_fused = fuse_embeddings(query_embs_dict)

    # DBA on Gallery
    gallery_refined = database_augmentation(
        gallery_fused, k=Config.KNN_DBA, device=device
    )

    # QE on Query
    query_refined = query_expansion(
        query_fused, gallery_refined, k=Config.KNN_QE, device=device
    )

    # Compute Similarity and Retrieve
    print("Computing similarity matrix...")
    gallery_refined = gallery_refined.to(device)
    query_refined = query_refined.to(device)

    # Similarity: (N_query, N_gallery)
    sim_matrix = torch.mm(query_refined, gallery_refined.t())

    # Get Top 5 indices in Gallery
    _, top_inds = torch.topk(sim_matrix, k=5, dim=1)
    top_inds = top_inds.cpu().numpy()

    # Map Gallery Indices to Class Labels
    # gallery_labels is (N_gallery,) array of class indices
    predicted_classes = []
    for i in range(len(top_inds)):
        indices = top_inds[i]
        # Retrieve class indices of the top k gallery images
        pred_labels = gallery_labels[indices]
        predicted_classes.append(pred_labels.tolist())

    # Calculate MAP@5
    map5 = calc_map5(predicted_classes, query_labels.tolist())
    print(f"Final Validation Metric: {map5}")

    # --- Failure Analysis ---
    print("\n--- Failure Analysis ---")

    # 1. Determine correctness (Is ground truth in top 5?)
    is_correct = []
    for preds, target in zip(predicted_classes, query_labels):
        is_correct.append(1 if target in preds else 0)

    # 2. Feature: Class Frequency (Samples per class in training)
    # Calculate frequency map
    class_counts = train_df["hotel_id"].value_counts().to_dict()
    # Map validation samples to their class frequency
    val_class_freqs = [class_counts.get(hid, 0) for hid in val_df["hotel_id"]]

    # 3. Calculate Correlation
    # Point-biserial correlation between binary correctness and continuous frequency
    if len(is_correct) > 0:
        corr = np.corrcoef(is_correct, val_class_freqs)[0, 1]
        print(
            f"Correlation between Error (Correctness) and Class Frequency: {corr:.4f}"
        )

        # Additional Stats
        df_analysis = pd.DataFrame({"correct": is_correct, "freq": val_class_freqs})
        print(
            "Average Class Frequency for Correct Predictions:",
            df_analysis[df_analysis["correct"] == 1]["freq"].mean(),
        )
        print(
            "Average Class Frequency for Incorrect Predictions:",
            df_analysis[df_analysis["correct"] == 0]["freq"].mean(),
        )

    return map5


def main():
    seed_everything(Config.SEED)

    # 1. Train Backbones
    for backbone in Config.BACKBONES:
        train_backbone(backbone)

    # 2. Validate Ensemble
    val_metric = validate_ensemble()

    # 3. Submission
    threshold = 0.7120973100214514
    if val_metric > threshold:
        print(
            f"\nValidation metric {val_metric} > {threshold}. Generating submission..."
        )
        # Clear GPU memory before inference
        torch.cuda.empty_cache()
        run_inference(load_cached_data=True)
    else:
        print(
            f"\nValidation metric {val_metric} <= {threshold}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
