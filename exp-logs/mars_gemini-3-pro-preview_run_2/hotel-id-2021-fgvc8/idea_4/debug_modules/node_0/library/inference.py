import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from library.config import CFG


def get_embeddings(dataloader, model, device, use_tta=True, load_cached_data=False):
    """
    Generates embeddings for the dataset using the provided model.
    Implements Test-Time Augmentation (TTA) by averaging predictions of original
    and horizontally flipped images.
    Includes caching mechanism to save/load embeddings from disk.

    Args:
        dataloader: PyTorch DataLoader for the test set.
        model: The trained neural network model.
        device: Computation device (cpu or cuda).
        use_tta (bool): Whether to use Test-Time Augmentation.
        load_cached_data (bool): Whether to attempt loading embeddings from cache.

    Returns:
        tuple: (embeddings, image_names)
            - embeddings: Tensor of shape (N, Embedding_Dim)
            - image_names: List of image filenames
    """
    cache_dir = CFG.working_dir
    os.makedirs(cache_dir, exist_ok=True)
    emb_path = os.path.join(cache_dir, "test_embeddings.npy")
    names_path = os.path.join(cache_dir, "test_names.npy")

    # 1. Check Cache
    if load_cached_data and os.path.exists(emb_path) and os.path.exists(names_path):
        print(f"Loading cached embeddings from {cache_dir}...")
        embeddings = torch.from_numpy(np.load(emb_path)).to(device)
        image_names = np.load(names_path).tolist()
        return embeddings, image_names

    # 2. Compute Embeddings
    model.eval()
    embeddings = []
    image_names = []

    print("Extracting features for test set...")
    with torch.no_grad():
        for step, (images, names) in enumerate(dataloader):
            images = images.to(device)

            if use_tta:
                # TTA: Average of Original and Horizontal Flip
                images_flipped = torch.flip(images, dims=[3])  # Flip width

                # Model returns embeddings in eval mode
                emb_orig = model(images)
                emb_flip = model(images_flipped)

                emb = (emb_orig + emb_flip) / 2.0
            else:
                emb = model(images)

            # L2 Normalize
            emb = F.normalize(emb, p=2, dim=1)

            embeddings.append(emb.cpu())
            image_names.extend(names)

            if CFG.debug and step >= 5:
                break

    embeddings = torch.cat(embeddings, dim=0)

    # 3. Save to Cache
    print(f"Saving embeddings to {cache_dir}...")
    np.save(emb_path, embeddings.numpy())
    np.save(names_path, np.array(image_names))

    return embeddings.to(device), image_names


def query_expansion(query_embeddings, gallery_embeddings, k=3):
    """
    Refines query embeddings using Average Query Expansion (AQE).
    It retrieves the top-k nearest neighbors from the gallery (class centers)
    and updates the query by averaging it with these neighbors.

    Args:
        query_embeddings: Tensor (N, D) - The test image embeddings.
        gallery_embeddings: Tensor (M, D) - The reference embeddings (Class Centers).
        k (int): Number of neighbors to use for expansion.

    Returns:
        Tensor: Refined query embeddings.
    """
    print(f"Applying Query Expansion (k={k})...")

    # Ensure inputs are normalized
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    gallery_embeddings = F.normalize(gallery_embeddings, p=2, dim=1)

    # Compute Cosine Similarity (N, M)
    # Note: This is an efficient Nearest Neighbor search on GPU
    sim = torch.matmul(query_embeddings, gallery_embeddings.T)

    # Retrieve Top K Neighbors
    _, top_idxs = torch.topk(sim, k=k, dim=1)  # (N, k)

    # Gather Neighbor Vectors
    # neighbors shape: (N, k, D)
    neighbors = gallery_embeddings[top_idxs]

    # Average Query Expansion
    # New Query = (Old Query + Sum(Neighbors)) / (1 + k)
    expanded_queries = (query_embeddings + neighbors.sum(dim=1)) / (1.0 + k)

    # Re-normalize
    expanded_queries = F.normalize(expanded_queries, p=2, dim=1)

    return expanded_queries


def find_matches(
    query_embeddings, gallery_embeddings, hotel_classes, k=5, subcenter_k=1
):
    """
    Finds the top-k matches for each query embedding against the gallery.
    Handles Sub-Center logic by aggregating scores per class.

    Args:
        query_embeddings: Tensor (N, D) - Test embeddings.
        gallery_embeddings: Tensor (C*K, D) - Class center weights.
        hotel_classes: Array-like - Original hotel ID labels.
        k (int): Number of top predictions to return.
        subcenter_k (int): Number of sub-centers per class.

    Returns:
        list: A list of space-delimited strings containing the top-k hotel IDs.
    """
    print("Finding matches...")

    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    gallery_embeddings = F.normalize(gallery_embeddings, p=2, dim=1)

    # Compute Similarity (N, C*K)
    sim = torch.matmul(query_embeddings, gallery_embeddings.T)

    # Handle Sub-Centers
    if subcenter_k > 1:
        num_classes = len(hotel_classes)
        # Reshape to (N, Classes, SubCenters)
        # Weights in SubCenterArcFace are stored as [Class0_K0, Class0_K1, ..., Class1_K0...]
        sim = sim.view(sim.size(0), num_classes, subcenter_k)

        # Max-Pool over SubCenters to get best score per class
        scores, _ = torch.max(sim, dim=2)  # (N, C)
    else:
        scores = sim

    # Get Top K Indices
    _, top_indices = torch.topk(scores, k=k, dim=1)
    top_indices = top_indices.cpu().numpy()

    # Map Indices to Hotel IDs
    predictions = []
    for idx_list in top_indices:
        # Map integer index back to original Hotel ID string
        pred_hotels = [str(hotel_classes[i]) for i in idx_list]
        predictions.append(" ".join(pred_hotels))

    return predictions


def inference(test_loader, model, device, hotel_classes):
    """
    Main inference pipeline.
    Orchestrates embedding generation, query expansion, matching, and submission saving.
    """
    # 1. Generate Embeddings (with TTA and Caching)
    embeddings, image_names = get_embeddings(
        test_loader,
        model,
        device,
        use_tta=CFG.use_tta,
        load_cached_data=False,  # Force computation for fresh inference
    )

    # 2. Prepare Gallery
    # In this architecture, the gallery is the set of learned class centers (weights)
    # Shape: (Num_Classes * SubCenter_K, Embedding_Dim)
    gallery_weights = model.hotel_head.weight.detach()

    # 3. Query Expansion
    if CFG.use_qe:
        embeddings = query_expansion(embeddings, gallery_weights, k=CFG.qe_k)

    # 4. Find Matches
    predictions = find_matches(
        embeddings, gallery_weights, hotel_classes, k=5, subcenter_k=CFG.subcenter_k
    )

    # 5. Generate Submission File
    submission_df = pd.DataFrame({"image": image_names, "hotel_id": predictions})

    sub_dir = "./submission"
    os.makedirs(sub_dir, exist_ok=True)
    sub_path = os.path.join(sub_dir, "submission.csv")

    submission_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")

    return submission_df
