import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import normalize

# Import from library
from library.config import Config
from library.utils import seed_everything, get_optimizer, get_scheduler
from library.dataset import HotelDataset, get_transforms, get_class_mapping
from library.model import HotelRecognitionModel
from library.engine import train_loop, extract_embeddings
from library.inference import run_inference, perform_dba, perform_qe


def calculate_map5(preds, targets):
    """
    Calculates Mean Average Precision @ 5.

    Args:
        preds (list): List of lists, where each inner list contains the top 5 predicted hotel_ids.
        targets (list): List of ground truth hotel_ids.

    Returns:
        float: The MAP@5 score.
    """
    score = 0.0
    n = len(targets)
    for p, t in zip(preds, targets):
        if t in p:
            # Rank is 1-based index
            rank = list(p).index(t) + 1
            score += 1.0 / rank
    return score / n


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline execution
    # We reduce epochs to ensure the script completes within the 2-hour limit
    # while still training on the full dataset for maximum coverage.
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 16

    print(f"Configuration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Image Size: {Config.IMG_SIZE}")

    # 2. Data Preparation
    print("\n[Data Preparation]")
    # Generate or load class mapping
    class_mapping = get_class_mapping(load_cached_data=True)

    # Training Data
    train_dataset = HotelDataset(
        Config.TRAIN_METADATA_PATH,
        transform=get_transforms("train"),
        class_mapping=class_mapping,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # Validation Data (for Loss monitoring)
    val_dataset = HotelDataset(
        Config.VAL_METADATA_PATH,
        transform=get_transforms("valid"),
        class_mapping=class_mapping,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    # 3. Model Initialization
    print("\n[Model Initialization]")
    device = torch.device(Config.DEVICE)
    model = HotelRecognitionModel()
    model.to(device)

    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    print("\n[Training]")
    train_loop(
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        device,
        epochs=Config.EPOCHS,
    )

    # 5. Validation (MAP@5 Calculation)
    print("\n[Validation Inference]")
    # Load best model weights
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"  Loading best model from {Config.BEST_MODEL_PATH}")
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("  Warning: Best model not found, using current weights.")

    model.eval()

    # Prepare datasets for embedding extraction (No shuffle, Test transforms)
    # Gallery = Train Set
    gallery_dataset = HotelDataset(
        Config.TRAIN_METADATA_PATH, transform=get_transforms("test"), is_test=True
    )
    gallery_loader = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # Query = Validation Set
    query_dataset = HotelDataset(
        Config.VAL_METADATA_PATH, transform=get_transforms("test"), is_test=True
    )
    query_loader = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    print("  Extracting Gallery (Train) embeddings...")
    gallery_emb = extract_embeddings(gallery_loader, model, device)

    # Save Gallery embeddings to cache for the Inference step to reuse
    # This prevents re-extracting the 70k training images during final submission generation
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.save(Config.GALLERY_EMBEDDINGS_PATH, gallery_emb)
    print(f"  Saved Gallery embeddings to {Config.GALLERY_EMBEDDINGS_PATH}")

    print("  Extracting Query (Validation) embeddings...")
    query_emb = extract_embeddings(query_loader, model, device)

    # Refinement (DBA & QE)
    print("  Refining embeddings (DBA/QE)...")

    # Normalize raw embeddings
    G = normalize(gallery_emb, axis=1)
    Q = normalize(query_emb, axis=1)

    # Database Augmentation on Gallery
    if Config.USE_DBA:
        G = perform_dba(G, k=Config.KNN)

    # Query Expansion on Validation Query
    if Config.USE_QE:
        Q = perform_qe(Q, G, k=Config.KNN)

    # Retrieval
    print("  Computing Similarity...")
    # Convert to Tensor for GPU calculation
    G_t = torch.from_numpy(G).to(device)
    Q_t = torch.from_numpy(Q).to(device)

    # Compute Cosine Similarity
    similarity = torch.matmul(Q_t, G_t.T)

    # Get Top 5
    _, indices = torch.topk(similarity, k=5, dim=1)
    indices = indices.cpu().numpy()

    # Map indices to hotel_ids
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    gallery_ids = train_df["hotel_id"].values
    val_targets = val_df["hotel_id"].values

    preds = []
    for idx_row in indices:
        preds.append([gallery_ids[i] for i in idx_row])

    # Calculate Metric
    val_map5 = calculate_map5(preds, val_targets)
    print(f"Final Validation Metric: {val_map5}")

    # 6. Failure Analysis
    print("\n[Failure Analysis]")
    # Calculate error per instance (1 - AP)
    errors = []
    for p, t in zip(preds, val_targets):
        score = 0.0
        if t in p:
            score = 1.0 / (list(p).index(t) + 1)
        errors.append(1.0 - score)

    val_df["error"] = errors

    # Feature: Class Frequency in Training (to check for long-tail performance issues)
    train_counts = train_df["hotel_id"].value_counts()
    val_df["train_samples"] = val_df["hotel_id"].map(train_counts).fillna(0)

    # Correlation
    corr = val_df["error"].corr(val_df["train_samples"])
    print(f"Correlation between Error and Class Frequency: {corr}")

    # 7. Submission
    threshold = 0.7120973100214514
    if val_map5 > threshold:
        print("\n[Submission]")
        print(f"Metric {val_map5} > {threshold}. Generating submission...")
        # Run inference using cached gallery embeddings (load_cached_data=True)
        # This will load the gallery embeddings we just saved, extract test embeddings,
        # perform DBA/QE, and generate the submission file.
        run_inference(load_cached_data=True)
    else:
        print(f"\n[Submission]")
        print(f"Metric {val_map5} <= {threshold}. Skipping submission.")


if __name__ == "__main__":
    main()
