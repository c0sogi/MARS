import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from library.config import Config
from library.utils import mean_average_precision
from library.dataset import get_dataloaders
from library.model import HotelIdModel
from library.loss import SubCenterArcFaceLoss

# Use Mixed Precision for A100
scaler = torch.cuda.amp.GradScaler()


def train_one_epoch(
    epoch, model, loss_fn, optimizer, dataloader, device, scheduler=None
):
    """
    Performs one epoch of training.
    """
    model.train()
    loss_fn.train()

    running_loss = 0.0
    dataset_size = 0

    # Iterate over data
    # Using tqdm for progress tracking is standard but prompt asked to minimize prints
    # We will print summary at the end.

    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            embeddings = model(images)
            loss = loss_fn(embeddings, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def valid_one_epoch(epoch, model, loss_fn, dataloader, device):
    """
    Performs validation using ArcFace proxies (sub-centers) as class prototypes.
    This is faster than instance-to-instance retrieval for monitoring training.
    """
    model.eval()
    loss_fn.eval()

    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    # Get weights from loss function for proxy-based classification
    # Weights shape: (num_classes * k, embedding_size)
    with torch.no_grad():
        # Normalize weights once
        weights = F.normalize(loss_fn.weight, p=2, dim=1)

    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                embeddings = model(images)
                # Calculate loss for monitoring
                loss = loss_fn(embeddings, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # --- MAP@5 Calculation using Proxies ---
            # Normalize embeddings
            embeddings_norm = F.normalize(embeddings.float(), p=2, dim=1)

            # Cosine similarity: (Batch, Num_Classes * K)
            cosine_all = torch.matmul(embeddings_norm, weights.T)

            # Reshape to handle sub-centers: (Batch, Num_Classes, K)
            cosine_all = cosine_all.view(
                batch_size, Config.NUM_CLASSES, Config.K_SUB_CENTERS
            )

            # Max over sub-centers to get class score: (Batch, Num_Classes)
            cosine, _ = torch.max(cosine_all, dim=2)

            # Get top 5 predictions
            _, topk_indices = torch.topk(cosine, Config.TOP_K, dim=1)

            all_preds.extend(topk_indices.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    epoch_loss = running_loss / dataset_size

    # Calculate MAP@5
    val_map = mean_average_precision(all_preds, all_targets, k=Config.TOP_K)

    return epoch_loss, val_map


def train_model(model_name, train_loader, val_loader, device, epochs=Config.EPOCHS):
    """
    Orchestrates the training loop for a specific model architecture.
    """
    print(f"\n[Training] Starting training for {model_name}...")

    # Initialize Model
    model = HotelIdModel(model_name=model_name).to(device)

    # Initialize Loss
    loss_fn = SubCenterArcFaceLoss(
        num_classes=Config.NUM_CLASSES,
        embedding_size=Config.EMBEDDING_SIZE,
    ).to(device)

    # Optimizer (Parameters of both model and loss)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Scheduler
    # CosineAnnealingLR updates per epoch usually, but can be per step if T_max is steps.
    # Config says T_MAX = EPOCHS, so we step per epoch.
    # However, standard practice with AdamW often uses OneCycle or Cosine per step.
    # Given the loop structure, we'll step per epoch as implied by T_MAX=EPOCHS.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.MIN_LR
    )

    best_map = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, f"best_model_{model_name}.pth")

    # Early Stopping settings
    patience = 5
    counter = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None
        )

        val_loss, val_map = valid_one_epoch(epoch, model, loss_fn, val_loader, device)

        # Step scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MAP@5: {val_map:.6f}"
        )

        # Save best model
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> Model Saved! New Best MAP: {val_map:.6f}")
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(
                f"  >>> Early stopping triggered after {patience} epochs without improvement."
            )
            break

    print(f"Training finished for {model_name}. Best MAP: {best_map:.6f}")

    # Clean up to save memory
    del model, loss_fn, optimizer
    torch.cuda.empty_cache()

    return best_model_path


def extract_features(model, dataloader, device):
    """
    Extracts embeddings for a given model and dataloader.
    """
    model.eval()
    embeddings = []
    labels = []  # Can be hotel_id (int) or image_name (str) for test

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            with torch.cuda.amp.autocast():
                emb = model(images)

            embeddings.append(emb.cpu().numpy())

            # Targets might be tensors or tuples/lists depending on dataset
            if isinstance(targets, torch.Tensor):
                labels.extend(targets.cpu().numpy().tolist())
            else:
                labels.extend(targets)

    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings, np.array(labels)


def get_model_embeddings(model_name, split, dataloader, device, load_cached_data=True):
    """
    Handles caching logic for embedding extraction.
    split: 'train' or 'test'
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    emb_path = os.path.join(Config.WORKING_DIR, f"{split}_embeddings_{model_name}.npy")
    lbl_path = os.path.join(Config.WORKING_DIR, f"{split}_labels_{model_name}.npy")

    if load_cached_data and os.path.exists(emb_path) and os.path.exists(lbl_path):
        print(f"Loading cached {split} embeddings for {model_name}...")
        embeddings = np.load(emb_path)
        labels = np.load(lbl_path)
        return embeddings, labels

    print(f"Extracting {split} embeddings for {model_name}...")

    # Load model
    model_path = os.path.join(Config.WORKING_DIR, f"best_model_{model_name}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. Train the model first."
        )

    model = HotelIdModel(model_name=model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    embeddings, labels = extract_features(model, dataloader, device)

    # Save to cache
    np.save(emb_path, embeddings)
    np.save(lbl_path, labels)

    del model
    torch.cuda.empty_cache()

    return embeddings, labels


def generate_submission(load_cached_data=True):
    """
    Main inference pipeline:
    1. Load/Extract embeddings for both backbones (Train and Test).
    2. Concatenate embeddings.
    3. Perform Database Augmentation (DBA).
    4. Perform Query Expansion (QE).
    5. Generate predictions and save submission.
    """
    device = Config.DEVICE

    # Get DataLoaders
    # We need train_loader for Gallery and test_loader for Query
    # Note: We use the full training set (train + val) as gallery usually,
    # but here we use the 'train_loader' provided by get_dataloaders which covers ~80%.
    # Ideally, we should use a loader that covers all data, but we stick to provided utils.
    train_loader, val_loader, test_loader, class_to_idx, idx_to_class = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 1. Extract and Concatenate Embeddings
    gallery_embs_list = []
    query_embs_list = []
    gallery_labels = None
    query_ids = None

    for model_name in Config.MODEL_NAMES:
        # Gallery (Train)
        g_emb, g_lbl = get_model_embeddings(
            model_name, "train", train_loader, device, load_cached_data
        )
        # Normalize individual backbone features before concatenation
        g_emb = normalize(g_emb, norm="l2", axis=1)
        gallery_embs_list.append(g_emb)
        if gallery_labels is None:
            gallery_labels = g_lbl

        # Query (Test)
        q_emb, q_id = get_model_embeddings(
            model_name, "test", test_loader, device, load_cached_data
        )
        q_emb = normalize(q_emb, norm="l2", axis=1)
        query_embs_list.append(q_emb)
        if query_ids is None:
            query_ids = q_id

    # Concatenate
    gallery_embeddings = np.concatenate(gallery_embs_list, axis=1)
    query_embeddings = np.concatenate(query_embs_list, axis=1)

    print(f"Combined Gallery Shape: {gallery_embeddings.shape}")
    print(f"Combined Query Shape: {query_embeddings.shape}")

    # Normalize concatenated features
    gallery_embeddings = normalize(gallery_embeddings, norm="l2", axis=1)
    query_embeddings = normalize(query_embeddings, norm="l2", axis=1)

    # 2. Database Augmentation (DBA)
    # Refine gallery embeddings by aggregating neighbors
    print("Performing Database Augmentation (DBA)...")
    knn_dba = NearestNeighbors(n_neighbors=Config.DBA_K, metric="cosine", n_jobs=-1)
    knn_dba.fit(gallery_embeddings)
    dists, indices = knn_dba.kneighbors(gallery_embeddings)

    # Weighted average based on similarity (1 - distance) or just mean
    # Simple mean is robust enough
    gallery_embeddings_dba = np.zeros_like(gallery_embeddings)
    for i in range(len(gallery_embeddings)):
        neighbor_indices = indices[i]
        gallery_embeddings_dba[i] = np.mean(
            gallery_embeddings[neighbor_indices], axis=0
        )

    # Re-normalize
    gallery_embeddings = normalize(gallery_embeddings_dba, norm="l2", axis=1)

    # 3. Query Expansion (QE)
    # Refine query embeddings using top matches from gallery
    print("Performing Query Expansion (QE)...")
    knn_qe = NearestNeighbors(n_neighbors=Config.QE_K, metric="cosine", n_jobs=-1)
    knn_qe.fit(gallery_embeddings)
    _, indices = knn_qe.kneighbors(query_embeddings)

    query_embeddings_qe = np.zeros_like(query_embeddings)
    for i in range(len(query_embeddings)):
        neighbor_indices = indices[i]
        # Average original query with its top gallery matches
        # We can weight the original query higher, e.g., 0.5 * query + 0.5 * mean(neighbors)
        # Here we take the mean of (query + neighbors)
        vectors = np.vstack([query_embeddings[i], gallery_embeddings[neighbor_indices]])
        query_embeddings_qe[i] = np.mean(vectors, axis=0)

    # Re-normalize
    query_embeddings = normalize(query_embeddings_qe, norm="l2", axis=1)

    # 4. Final Retrieval
    print("Performing Final Retrieval...")
    knn_final = NearestNeighbors(n_neighbors=Config.TOP_K, metric="cosine", n_jobs=-1)
    knn_final.fit(gallery_embeddings)
    distances, indices = knn_final.kneighbors(query_embeddings)

    # 5. Generate Submission
    print("Generating Submission File...")
    predictions = []

    for i in range(len(query_ids)):
        img_id = query_ids[i]
        # Get indices of nearest gallery items
        neighbor_idxs = indices[i]
        # Get corresponding hotel_ids (labels)
        # gallery_labels contains the class indices (0..N-1)
        pred_class_idxs = gallery_labels[neighbor_idxs]

        # Convert class indices back to hotel_ids
        pred_hotel_ids = [str(idx_to_class[idx]) for idx in pred_class_idxs]

        predictions.append({"image": img_id, "hotel_id": " ".join(pred_hotel_ids)})

    df_sub = pd.DataFrame(predictions)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training_pipeline():
    """
    Trains all models defined in Config.
    """
    device = Config.DEVICE
    train_loader, val_loader, _, _, _ = get_dataloaders(load_cached_data=True)

    for model_name in Config.MODEL_NAMES:
        train_model(model_name, train_loader, val_loader, device)
