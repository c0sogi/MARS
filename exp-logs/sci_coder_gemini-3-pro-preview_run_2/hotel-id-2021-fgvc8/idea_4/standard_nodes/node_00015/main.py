import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import CFG
from library.utils import seed_everything
from library.dataset import process_metadata, HotelDataset, get_transforms
from library.model import HotelNet
from library.engine import train_loop
from library.inference import inference


def custom_validate(dataloader, model, device):
    """
    Performs validation using the model's embeddings and class centers.
    Computes MAP@5 using pure cosine similarity (no margin penalty).
    Returns the mean MAP@5, per-sample APs, and ground truth targets.
    """
    model.eval()
    embeddings = []
    targets = []

    # Extract Embeddings
    with torch.no_grad():
        for images, hotel_labels, _ in dataloader:
            images = images.to(device)
            # Forward in eval mode returns embeddings
            emb = model(images)
            emb = F.normalize(emb, p=2, dim=1)
            embeddings.append(emb.cpu())
            targets.append(hotel_labels)

    embeddings = torch.cat(embeddings, dim=0).to(device)
    targets = torch.cat(targets, dim=0).numpy()

    # Class Centers (Gallery)
    # Weights shape: (Num_Classes * SubCenter_K, Embedding_Dim)
    weights = model.hotel_head.weight
    weights = F.normalize(weights, p=2, dim=1)

    # Compute Similarity Matrix: (N_val, Num_Classes * SubCenter_K)
    sim = torch.matmul(embeddings, weights.T)

    # Handle SubCenter: Aggregate scores per class
    # Reshape to (N_val, Num_Classes, SubCenter_K)
    sim = sim.view(sim.size(0), CFG.num_classes, CFG.subcenter_k)

    # Max-pool over sub-centers to get the best score for each class
    scores, _ = torch.max(sim, dim=2)  # (N_val, Num_Classes)

    # Get Top 5 Predictions
    _, top_k_indices = torch.topk(scores, k=5, dim=1)
    predictions = top_k_indices.cpu().numpy()

    # Calculate MAP@5 and per-sample AP
    aps = []
    for i in range(len(targets)):
        target = targets[i]
        pred_row = predictions[i]
        ap = 0.0
        for rank, label in enumerate(pred_row):
            if label == target:
                ap = 1.0 / (rank + 1)
                break
        aps.append(ap)

    mean_map = np.mean(aps)
    return mean_map, aps, targets


def main():
    # 1. Setup
    seed_everything(CFG.seed)
    device = torch.device(CFG.device)

    # 2. Data Preparation
    # Load metadata and label encodings
    train_df, val_df, test_df, hotel_classes, chain_classes = process_metadata(
        load_cached_data=True
    )

    # Fast Baseline Strategy: Subsample training data
    # Limit to 5000 samples to ensure execution completes within the 2-hour limit
    if len(train_df) > 5000:
        print(
            f"Subsampling training data from {len(train_df)} to 5000 samples for fast baseline..."
        )
        train_df = train_df.sample(n=5000, random_state=CFG.seed).reset_index(drop=True)

    # Create Datasets
    train_dataset = HotelDataset(
        train_df, transform=get_transforms("train"), mode="train"
    )
    val_dataset = HotelDataset(val_df, transform=get_transforms("val"), mode="val")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = HotelNet()
    model.to(device)

    # 4. Training
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )

    # Override epochs for fast baseline execution
    epochs = 2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader)
    )

    print(f"Starting training for {epochs} epochs...")
    model = train_loop(
        train_loader, val_loader, model, optimizer, scheduler, device, epochs
    )

    # 5. Final Validation
    print("Performing final validation on the hold-out set...")
    val_map, val_aps, val_targets = custom_validate(val_loader, model, device)

    # Print the required metric
    print(f"Final Validation Metric: {val_map}")

    # 6. Failure Analysis
    print("Performing failure analysis...")

    # Load full training metadata to get accurate class frequencies (prior to subsampling)
    full_train_df = pd.read_csv(CFG.train_metadata_path)

    # Map hotel_ids to the indices used by the model
    hotel_to_idx = {hotel: idx for idx, hotel in enumerate(hotel_classes)}
    full_train_df["hotel_id_idx"] = full_train_df["hotel_id"].map(hotel_to_idx)

    # Calculate class frequencies in the training set
    class_counts = full_train_df["hotel_id_idx"].value_counts()

    # Map these counts to the validation samples
    # val_targets contains the hotel_id_idx for each validation image
    val_sample_counts = np.array([class_counts.get(t, 0) for t in val_targets])

    # Calculate Error (1.0 - AP)
    val_errors = 1.0 - np.array(val_aps)

    # Calculate Correlation
    if len(val_errors) > 1:
        corr = np.corrcoef(val_errors, val_sample_counts)[0, 1]
        print(
            f"Correlation between Error magnitude (1-AP) and Training Class Frequency: {corr}"
        )
    else:
        print("Insufficient validation samples for correlation analysis.")

    # 7. Submission
    threshold = 0.7120973100214514

    if val_map > threshold:
        print(
            f"Validation metric ({val_map}) exceeds threshold ({threshold}). Generating submission..."
        )

        test_dataset = HotelDataset(
            test_df, transform=get_transforms("test"), mode="test"
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )

        # Run inference (includes TTA and Query Expansion)
        inference(test_loader, model, device, hotel_classes)
    else:
        print(
            f"Validation metric ({val_map}) did not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
