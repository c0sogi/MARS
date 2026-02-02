import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np
import os

# Import library modules
from library.config import Config
from library.utils import seed_everything, save_checkpoint, load_checkpoint
from library.dataset import get_label_mapping, get_transforms, HotelDataset
from library.model import HotelModel
from library.engine import train_one_epoch, evaluate, inference


def analyze_failures(model, data_loader, val_df, device, train_df_full):
    """
    Analyzes model failures by correlating error magnitude with class frequency.
    """
    print("Running Failure Analysis...")
    model.eval()
    all_ranks = []

    # Calculate ranks for each validation sample
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Get normalized embeddings (inference mode)
            embeddings = model(images, labels=None)
            features = torch.nn.functional.normalize(embeddings)

            # Get normalized head weights
            head = model.head
            weights = torch.nn.functional.normalize(head.weight)

            # Compute Cosine Similarity
            cosine = torch.nn.functional.linear(features, weights)
            cosine = cosine.view(-1, head.out_features, head.k)
            # Max similarity across sub-centers
            cosine, _ = torch.max(cosine, dim=2)

            # Get rank of ground truth
            # Gather cosine similarity of the target class
            target_cosine = cosine.gather(1, labels.view(-1, 1))
            # Rank is number of classes with similarity > target_cosine + 1
            ranks = (cosine > target_cosine).sum(dim=1) + 1
            all_ranks.extend(ranks.cpu().numpy())

    all_ranks = np.array(all_ranks)

    # Calculate Error Magnitude: 1 - Reciprocal Rank (MRR contribution)
    # Perfect prediction (Rank 1) -> Error 0.0
    # Rank infinity -> Error 1.0
    reciprocal_ranks = 1.0 / all_ranks
    errors = 1.0 - reciprocal_ranks

    # Get Class Frequency from Training Set
    class_counts = train_df_full["hotel_id"].value_counts().to_dict()

    # Map validation samples to their training class frequency
    # Note: val_loader is sequential, so it matches val_df order
    val_class_freqs = val_df["hotel_id"].map(class_counts).fillna(0).values

    # Calculate Correlation
    if len(errors) == len(val_class_freqs):
        corr = np.corrcoef(errors, val_class_freqs)[0, 1]
        print(f"Correlation between Error Magnitude and Class Frequency: {corr:.16f}")
    else:
        print("Error: Shape mismatch in failure analysis.")


def run():
    # 1. Config & Setup
    config = Config()
    seed_everything(config.seed)

    print(f"Device: {config.device}")

    # 2. Data Loading
    # Load full training data to generate complete label mapping
    full_train_df = pd.read_csv(config.train_csv_path)
    val_df = pd.read_csv(config.val_csv_path)

    # Generate label mapping using full dataset
    class_to_idx, idx_to_class = get_label_mapping(full_train_df, config.working_dir)

    # Subsample training data for fast baseline execution
    MAX_TRAIN_SAMPLES = 40000
    if len(full_train_df) > MAX_TRAIN_SAMPLES:
        print(f"Subsampling training data to {MAX_TRAIN_SAMPLES} samples...")
        train_df = full_train_df.sample(
            n=MAX_TRAIN_SAMPLES, random_state=config.seed
        ).reset_index(drop=True)
    else:
        train_df = full_train_df

    # 3. Model Initialization
    print("Initializing Model...")
    model = HotelModel(
        backbone_name=config.backbone_name,
        n_classes=config.n_classes,
        embedding_dim=config.embedding_dim,
        pretrained=config.pretrained,
        use_gem_pooling=config.use_gem_pooling,
        use_bn_neck=config.use_bn_neck,
        arcface_scale=config.arcface_scale,
        arcface_margin=config.arcface_margin,
        sub_centers_k=config.sub_centers_k,
    )
    model.to(config.device)

    # 4. Stage 1: Low Resolution Training
    print("\n=== Stage 1: Low Resolution Training (224x224) ===")

    train_transform_s1 = get_transforms(config.stage1_resolution, mode="train")
    val_transform_s1 = get_transforms(config.stage1_resolution, mode="val")

    train_dataset_s1 = HotelDataset(
        train_df,
        config.image_root_dir,
        transform=train_transform_s1,
        mode="train",
        class_to_idx=class_to_idx,
    )
    val_dataset_s1 = HotelDataset(
        val_df,
        config.image_root_dir,
        transform=val_transform_s1,
        mode="val",
        class_to_idx=class_to_idx,
    )

    train_loader_s1 = DataLoader(
        train_dataset_s1,
        batch_size=config.stage1_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader_s1 = DataLoader(
        val_dataset_s1,
        batch_size=config.stage1_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=config.stage1_lr, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.stage1_epochs)

    for epoch in range(config.stage1_epochs):
        train_one_epoch(model, optimizer, train_loader_s1, config.device, epoch + 1)
        scheduler.step()

    # Save Stage 1 Checkpoint
    save_checkpoint(
        {
            "epoch": config.stage1_epochs,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        is_best=False,
        filepath=os.path.join(config.working_dir, "stage1_checkpoint.pth"),
    )

    # 5. Stage 2: High Resolution Training
    print("\n=== Stage 2: High Resolution Training (384x384) ===")

    train_transform_s2 = get_transforms(config.stage2_resolution, mode="train")
    val_transform_s2 = get_transforms(config.stage2_resolution, mode="val")

    train_dataset_s2 = HotelDataset(
        train_df,
        config.image_root_dir,
        transform=train_transform_s2,
        mode="train",
        class_to_idx=class_to_idx,
    )
    val_dataset_s2 = HotelDataset(
        val_df,
        config.image_root_dir,
        transform=val_transform_s2,
        mode="val",
        class_to_idx=class_to_idx,
    )

    train_loader_s2 = DataLoader(
        train_dataset_s2,
        batch_size=config.stage2_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader_s2 = DataLoader(
        val_dataset_s2,
        batch_size=config.stage2_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Re-initialize optimizer with lower learning rate
    optimizer = optim.AdamW(
        model.parameters(), lr=config.stage2_lr, weight_decay=config.weight_decay
    )

    # Scheduler with Warmup
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, total_iters=config.stage2_warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=config.stage2_epochs - config.stage2_warmup_epochs
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.stage2_warmup_epochs],
    )

    best_map = 0.0

    for epoch in range(config.stage2_epochs):
        current_epoch = config.stage1_epochs + epoch + 1
        train_one_epoch(model, optimizer, train_loader_s2, config.device, current_epoch)
        scheduler.step()

        # Validation
        val_loss, map_score = evaluate(model, val_loader_s2, config.device)

        is_best = map_score > best_map
        best_map = max(map_score, best_map)

        save_checkpoint(
            {
                "epoch": current_epoch,
                "state_dict": model.state_dict(),
                "best_map": best_map,
            },
            is_best,
            filepath=os.path.join(
                config.working_dir, f"checkpoint_ep{current_epoch}.pth"
            ),
        )

    # 6. Final Validation & Failure Analysis
    print(f"Final Validation Metric: {best_map:.16f}")

    # Load best model for analysis and inference
    best_model_path = os.path.join(config.working_dir, "best_model.pth")
    load_checkpoint(best_model_path, model, device=config.device)

    analyze_failures(model, val_loader_s2, val_df, config.device, full_train_df)

    # 7. Submission
    THRESHOLD = 0.6236649771247449
    if best_map > THRESHOLD:
        print(
            f"Validation metric {best_map:.4f} > {THRESHOLD}. Generating submission..."
        )

        test_df = pd.read_csv(config.test_csv_path)
        test_transform = get_transforms(config.inference_resolution, mode="test")
        test_dataset = HotelDataset(
            test_df, config.image_root_dir, transform=test_transform, mode="test"
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.inference_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        inference(
            model,
            test_loader,
            config.device,
            idx_to_class,
            config.submission_path,
            use_tta=config.use_tta,
        )
    else:
        print(f"Validation metric {best_map:.4f} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    run()
