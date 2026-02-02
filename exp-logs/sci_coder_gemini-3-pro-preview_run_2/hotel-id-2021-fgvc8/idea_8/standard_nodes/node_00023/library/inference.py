import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import HotelDataset, get_transforms, get_class_mapping
from library.models import HotelRecognitionModel
from library.utils import seed_everything


def extract_features(loader, model, device):
    """
    Runs inference on a dataloader using the provided model.
    Returns raw embeddings and corresponding identifiers (labels or image IDs).
    """
    model.eval()
    embeddings = []
    identifiers = []

    with torch.no_grad():
        for batch in loader:
            # batch is (imgs, labels) for train/val or (imgs, ids) for test
            imgs = batch[0].to(device)
            targets = batch[1]

            # Model returns embeddings when labels=None
            embs = model(imgs, labels=None)
            embeddings.append(embs.cpu())

            if isinstance(targets, torch.Tensor):
                identifiers.append(targets.cpu().numpy())
            else:
                # targets is a tuple/list of strings for test set
                identifiers.extend(targets)

    embeddings = torch.cat(embeddings, dim=0).numpy()

    if len(identifiers) > 0:
        if isinstance(identifiers[0], np.ndarray):
            identifiers = np.concatenate(identifiers, axis=0)
        else:
            identifiers = np.array(identifiers)

    return embeddings, identifiers


def get_dataset_embeddings(
    backbone_name, mode, img_size, device, load_cached_data=True
):
    """
    Generates or loads cached embeddings for a specific backbone and dataset mode.

    Args:
        backbone_name (str): Name of the backbone (e.g., 'efficientnet_b4').
        mode (str): 'gallery' (train set) or 'query' (test set).
        img_size (int): Input image resolution.
        device (torch.device): Compute device.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (embeddings, identifiers)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_emb_path = os.path.join(
        Config.WORKING_DIR, f"{mode}_embeddings_{backbone_name}.npy"
    )
    cache_id_path = os.path.join(Config.WORKING_DIR, f"{mode}_ids_{backbone_name}.npy")

    # Try loading from cache
    if (
        load_cached_data
        and os.path.exists(cache_emb_path)
        and os.path.exists(cache_id_path)
    ):
        print(f"Loading cached {mode} embeddings for {backbone_name}...")
        embeddings = np.load(cache_emb_path)
        ids = np.load(cache_id_path)
        return embeddings, ids

    print(f"Generating {mode} embeddings for {backbone_name}...")

    # Setup Data
    if mode == "gallery":
        # Gallery uses the training set
        df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        # We use 'valid' mode for dataset to ensure deterministic transforms (Resize + Normalize)
        # We need class mapping to create the dataset
        class_mapping = get_class_mapping(df, load_cached_data=True)
        dataset = HotelDataset(
            df,
            transform=get_transforms(img_size, mode="valid"),
            mode="valid",
            class_mapping=class_mapping,
        )
    elif mode == "query":
        # Query uses the test set
        df = pd.read_csv(Config.TEST_METADATA_PATH)
        dataset = HotelDataset(
            df, transform=get_transforms(img_size, mode="test"), mode="test"
        )
    else:
        raise ValueError("mode must be 'gallery' or 'query'")

    # Use a larger batch size for inference if possible, but keep it safe
    batch_size = Config.PHASE2_CONFIG["batch_size"] * 2
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Setup Model
    model = HotelRecognitionModel(
        backbone_name,
        num_classes=Config.NUM_CLASSES,
        embedding_dim=Config.EMBEDDING_DIM,
        pretrained=False,
    )
    checkpoint_path = Config.get_checkpoint_path(backbone_name)

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Checkpoint {checkpoint_path} not found. Using random initialization (expect poor performance)."
        )

    model.to(device)

    # Extract
    embeddings, ids = extract_features(loader, model, device)

    # Cache results
    np.save(cache_emb_path, embeddings)
    np.save(cache_id_path, ids)

    return embeddings, ids


def fuse_embeddings(emb_dict):
    """
    Concatenates L2-normalized embeddings from multiple backbones.

    Args:
        emb_dict (dict): Dictionary mapping backbone names to numpy embedding arrays.

    Returns:
        torch.Tensor: Fused and normalized embeddings.
    """
    fused = []
    # Ensure consistent order based on backbone names
    keys = sorted(emb_dict.keys())

    for k in keys:
        emb = torch.from_numpy(emb_dict[k])
        # L2 Normalize individual backbone features first
        emb = F.normalize(emb, p=2, dim=1)
        fused.append(emb)

    # Concatenate along feature dimension
    fused_emb = torch.cat(fused, dim=1)

    # L2 Normalize the fused vector
    fused_emb = F.normalize(fused_emb, p=2, dim=1)

    return fused_emb


def database_augmentation(gallery_emb, k=5, device="cuda"):
    """
    Refines gallery embeddings using Database Augmentation (DBA).
    Replaces each embedding with a weighted average of itself and its k nearest neighbors.
    """
    print(f"Applying Database Augmentation (DBA) with k={k}...")
    gallery_emb = gallery_emb.to(device)
    n = gallery_emb.size(0)

    # Compute Similarity Matrix: (N, N)
    # Note: For N=70k, this fits in A100 40GB VRAM.
    sim_matrix = torch.mm(gallery_emb, gallery_emb.t())

    # Find Top K neighbors
    # top_vals: (N, k), top_inds: (N, k)
    top_vals, top_inds = torch.topk(sim_matrix, k=k, dim=1)

    # Use similarity scores as weights
    weights = top_vals.unsqueeze(2)  # (N, k, 1)

    # Gather neighbor embeddings
    flat_inds = top_inds.view(-1)
    neighbors = gallery_emb[flat_inds].view(n, k, -1)  # (N, k, D)

    # Weighted sum
    refined = (neighbors * weights).sum(dim=1)  # (N, D)

    # Normalize
    refined = F.normalize(refined, p=2, dim=1)

    return refined


def query_expansion(query_emb, gallery_emb, k=5, device="cuda"):
    """
    Refines query embeddings using Query Expansion (QE).
    Averages the query vector with its k nearest neighbors from the gallery.
    """
    print(f"Applying Query Expansion (QE) with k={k}...")
    query_emb = query_emb.to(device)
    gallery_emb = gallery_emb.to(device)

    # Compute Similarity: Query (N_q, D) x Gallery.T (D, N_g) -> (N_q, N_g)
    sim_matrix = torch.mm(query_emb, gallery_emb.t())

    # Find Top K neighbors in gallery
    top_vals, top_inds = torch.topk(sim_matrix, k=k, dim=1)

    # Gather neighbors
    n_q = query_emb.size(0)
    flat_inds = top_inds.view(-1)
    neighbors = gallery_emb[flat_inds].view(n_q, k, -1)  # (N_q, k, D)

    # Average query with neighbors
    # Strategy: refined = query + mean(neighbors)
    neighbor_mean = neighbors.mean(dim=1)
    refined = query_emb + neighbor_mean

    # Normalize
    refined = F.normalize(refined, p=2, dim=1)

    return refined


def generate_submission(
    query_emb, gallery_emb, gallery_labels, query_ids, top_k=5, device="cuda"
):
    """
    Performs final retrieval and generates the submission file.
    """
    print("Generating predictions and submission file...")
    query_emb = query_emb.to(device)
    gallery_emb = gallery_emb.to(device)

    # Final Similarity Search
    sim_matrix = torch.mm(query_emb, gallery_emb.t())

    # Retrieve Top K candidates
    _, top_inds = torch.topk(sim_matrix, k=top_k, dim=1)
    top_inds = top_inds.cpu().numpy()

    # Load Class Mapping to convert indices back to Hotel IDs
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    class_mapping = get_class_mapping(df_train, load_cached_data=True)
    # Invert mapping: class_idx -> hotel_id
    idx_to_hotel = {v: k for k, v in class_mapping.items()}

    preds = []
    for i in range(len(query_ids)):
        indices = top_inds[i]
        row_preds = []
        for idx in indices:
            # gallery_labels contains class indices
            class_idx = gallery_labels[idx]
            hotel_id = idx_to_hotel[class_idx]
            row_preds.append(str(hotel_id))

        preds.append(" ".join(row_preds))

    # Create Submission DataFrame
    sub_df = pd.DataFrame({"image": query_ids, "hotel_id": preds})

    # Save
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_inference(load_cached_data=True):
    """
    Main function to run the full inference pipeline.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE
    # Use the higher resolution defined for Phase 2 for inference
    img_size = Config.PHASE2_CONFIG["img_size"]

    print(f"Starting Inference Pipeline on {device}...")

    # 1. Extract/Load Embeddings for both backbones
    gallery_embs_dict = {}
    query_embs_dict = {}
    gallery_labels = None
    query_ids = None

    for backbone in Config.BACKBONES:
        # Gallery (Train Set)
        g_emb, g_ids = get_dataset_embeddings(
            backbone, "gallery", img_size, device, load_cached_data
        )
        gallery_embs_dict[backbone] = g_emb
        if gallery_labels is None:
            gallery_labels = g_ids  # These are class indices

        # Query (Test Set)
        q_emb, q_ids = get_dataset_embeddings(
            backbone, "query", img_size, device, load_cached_data
        )
        query_embs_dict[backbone] = q_emb
        if query_ids is None:
            query_ids = q_ids

    # 2. Feature Fusion
    print("Fusing embeddings from dual backbones...")
    gallery_fused = fuse_embeddings(gallery_embs_dict)
    query_fused = fuse_embeddings(query_embs_dict)

    # 3. Database Augmentation (DBA)
    # Refine the gallery manifold
    gallery_refined = database_augmentation(
        gallery_fused, k=Config.KNN_DBA, device=device
    )

    # 4. Query Expansion (QE)
    # Refine the query vectors based on the refined gallery
    query_refined = query_expansion(
        query_fused, gallery_refined, k=Config.KNN_QE, device=device
    )

    # 5. Prediction & Submission
    generate_submission(
        query_refined,
        gallery_refined,
        gallery_labels,
        query_ids,
        top_k=Config.TOP_K,
        device=device,
    )
