import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, mapk
from library.dataset import get_dataloaders
from library.model import HotelConvNeXt
import library.post_processing as pp


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Training logic for one epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    # PKSampler determines the batch composition
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass with labels returns logits with ArcFace margin
        outputs = model(images, labels)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        count += images.size(0)

    return running_loss / count if count > 0 else 0.0


@torch.no_grad()
def extract_features(model, loader, device):
    """
    Extracts embeddings for a given loader.
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    all_names = []

    for images, labels, names in loader:
        images = images.to(device)

        # Forward pass without labels returns normalized embeddings
        embeddings = model(images, labels=None)

        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels)
        all_names.extend(names)

    return torch.cat(all_embeddings), torch.cat(all_labels), all_names


def validate(model, val_loader, gallery_loader, device):
    """
    Computes MAP@5 on the validation set using the training set as the gallery.
    """
    # 1. Extract Validation Embeddings (Query)
    val_embeddings, val_labels, _ = extract_features(model, val_loader, device)

    # 2. Extract Gallery Embeddings (Train)
    gal_embeddings, gal_labels, _ = extract_features(model, gallery_loader, device)

    # Ensure normalization (model output is already normalized, but good for safety)
    val_embeddings = F.normalize(val_embeddings, dim=1).to(device)
    gal_embeddings = F.normalize(gal_embeddings, dim=1).to(device)

    # 3. Compute Similarity Matrix
    # Shape: (N_val, N_gal)
    # Note: For very large datasets, this might need chunking, but fits in A100 for this task.
    sim_matrix = torch.matmul(val_embeddings, gal_embeddings.T)

    # 4. Top 5 Retrieval
    _, indices = torch.topk(sim_matrix, k=5, dim=1)
    indices = indices.cpu().numpy()

    # 5. Map indices to Hotel IDs
    gal_labels_np = gal_labels.numpy()
    val_labels_np = val_labels.numpy()

    predicted_lists = []
    for i in range(len(indices)):
        # Get the hotel_ids of the top 5 neighbors
        preds = gal_labels_np[indices[i]]
        predicted_lists.append(list(preds))

    actual_lists = [[label] for label in val_labels_np]

    # 6. Compute MAP@5
    score = mapk(actual_lists, predicted_lists, k=5)
    return score


def run_training():
    """
    Main execution function.
    """
    seed_everything(Config.seed)
    print(f"Device: {Config.device}")

    # Load Data
    # num_classes is the actual number of unique classes in the training set
    train_loader, val_loader, test_loader, gallery_loader, num_classes = (
        get_dataloaders(debug=Config.debug)
    )
    print(f"Number of classes: {num_classes}")

    # Initialize Model
    model = HotelConvNeXt(
        num_classes=num_classes,
        k_subcenters=Config.k_subcenters,
        margin=Config.margin,
        scale=Config.scale,
    )
    model = model.to(Config.device)

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.scheduler_T_max, eta_min=Config.min_lr
    )

    # Training Loop
    best_map = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(Config.epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.device
        )

        # Step Scheduler
        scheduler.step()

        # Validate
        val_map = validate(model, val_loader, gallery_loader, Config.device)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.epochs} - "
            f"Loss: {train_loss:.4f} - "
            f"Val MAP@5: {val_map} - "
            f"Time: {elapsed:.0f}s"
        )

        # Early Stopping & Checkpointing
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), Config.best_model_path)
            print(f"Saved Best Model (MAP@5: {best_map})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # --- Inference & Submission ---
    print("\nStarting Inference with Best Model...")

    # Load best model
    model.load_state_dict(torch.load(Config.best_model_path))
    model.to(Config.device)
    model.eval()

    # 1. Generate Gallery Embeddings (Full Training Set)
    print("Extracting Gallery Embeddings...")
    gal_emb, _, gal_names = extract_features(model, gallery_loader, Config.device)

    # 2. Generate Query Embeddings (Test Set)
    print("Extracting Query Embeddings...")
    qry_emb, _, qry_names = extract_features(model, test_loader, Config.device)

    # Save to Parquet for Post-Processing
    print("Saving embeddings to parquet...")
    gal_df = pd.DataFrame({"image": gal_names, "embedding": list(gal_emb.numpy())})
    gal_df.to_parquet(Config.gallery_embeddings_path, index=False)

    qry_df = pd.DataFrame({"image": qry_names, "embedding": list(qry_emb.numpy())})
    qry_df.to_parquet(Config.query_embeddings_path, index=False)

    # Run Post-Processing (DBA, QE, Submission Generation)
    print("Running Post-Processing...")
    pp.run_post_processing(load_cached_data=False)
