import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import scipy.sparse as sp
import timm
import gc

from library.config import Config
from library.data_utils import IndexMapper


# Set seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.SEED)


class HMImageDataset(Dataset):
    """
    Dataset to load images for articles.
    """

    def __init__(self, article_ids, img_dir, transform=None):
        self.article_ids = article_ids
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.article_ids)

    def __getitem__(self, idx):
        article_id = self.article_ids[idx]

        # Construct path: 0108775015 -> images/010/0108775015.jpg
        s = str(article_id).zfill(10)
        folder = s[:3]
        path = os.path.join(self.img_dir, folder, f"{s}.jpg")

        try:
            with open(path, "rb") as f:
                img = Image.open(f).convert("RGB")

            if self.transform:
                img = self.transform(img)
            return img, 1  # 1 indicates valid image

        except (FileNotFoundError, OSError, IOError):
            # Return zero tensor if image missing
            # Shape must match transform output: 3, 224, 224
            return torch.zeros((3, Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1])), 0


def extract_embeddings(mapper: IndexMapper, load_cached_data: bool = True):
    """
    Extracts image embeddings for all items in the mapper.
    Returns a numpy array of shape (n_items, embedding_dim).
    """
    cache_path = Config.CACHE_IMAGE_EMBEDDINGS

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading image embeddings from cache: {cache_path}")
        try:
            df_emb = pd.read_parquet(cache_path)
            # Ensure sorting by item_idx
            df_emb = df_emb.sort_values("item_idx")

            # Check if we have the correct number of items
            if len(df_emb) == mapper.get_num_items():
                # Convert list column to numpy matrix
                # Stacking lists is reasonably fast for 100k rows
                embeddings = np.stack(df_emb["embedding"].values)
                print(f"Loaded embeddings shape: {embeddings.shape}")
                return embeddings
            else:
                print("Cached embeddings count mismatch. Recomputing...")
        except Exception as e:
            print(f"Failed to load embedding cache: {e}. Recomputing...")

    # 2. Compute Embeddings
    print("Computing image embeddings...")

    # Setup Device
    device = torch.device(Config.DEVICE)

    # Setup Model (ResNet50 via timm)
    # num_classes=0 returns the pooled features (2048 dim)
    print(f"Loading model {Config.VISUAL_MODEL_NAME}...")
    model = timm.create_model(Config.VISUAL_MODEL_NAME, pretrained=True, num_classes=0)
    model = model.to(device)
    model.eval()

    # Setup Transforms
    transform = transforms.Compose(
        [
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.IMAGE_MEAN, std=Config.IMAGE_STD),
        ]
    )

    # Get all article IDs in order of indices 0..N-1
    n_items = mapper.get_num_items()
    all_indices = np.arange(n_items)
    all_article_ids = mapper.get_items_from_indices(all_indices)

    # Dataset & Loader
    dataset = HMImageDataset(
        all_article_ids, Config.PATH_IMAGES_DIR, transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    embeddings_list = []
    valid_mask = []

    print(f"Starting inference on {n_items} images...")

    with torch.no_grad():
        for i, (imgs, valids) in enumerate(dataloader):
            imgs = imgs.to(device)

            # Forward pass
            feats = model(imgs)  # (B, 2048)

            # Move to CPU
            feats = feats.cpu().numpy()

            # Zero out embeddings for missing images (just to be safe)
            # valids is (B,) tensor
            valids_np = valids.numpy()
            feats = feats * valids_np[:, None]

            embeddings_list.append(feats)

            if (i + 1) % 50 == 0:
                print(f"Processed batch {i+1}/{len(dataloader)}")

    # Concatenate
    all_embeddings = np.concatenate(embeddings_list, axis=0)

    # 3. Save Cache
    print(f"Saving embeddings to {cache_path}...")
    # Create DataFrame for Parquet
    # We store embeddings as lists to be compatible with simple parquet schemas
    df_emb = pd.DataFrame(
        {"item_idx": np.arange(len(all_embeddings)), "embedding": list(all_embeddings)}
    )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_emb.to_parquet(cache_path, index=False)

    print(f"Embeddings computed. Shape: {all_embeddings.shape}")
    return all_embeddings


def compute_visual_similarity_matrix(
    mapper: IndexMapper, load_cached_data: bool = True
):
    """
    Computes the sparse Item-Item visual similarity matrix.
    Returns scipy.sparse.csr_matrix of shape (n_items, n_items).
    """
    cache_path = Config.CACHE_SIM_VISUAL

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading visual similarity matrix from cache: {cache_path}")
        try:
            matrix = sp.load_npz(cache_path)
            if matrix.shape == (mapper.get_num_items(), mapper.get_num_items()):
                return matrix
            else:
                print("Cached matrix shape mismatch. Recomputing...")
        except Exception as e:
            print(f"Failed to load similarity cache: {e}. Recomputing...")

    # 2. Compute Matrix
    # Get embeddings (N, D)
    embeddings = extract_embeddings(mapper, load_cached_data=load_cached_data)
    n_items, dim = embeddings.shape

    print("Computing similarity matrix...")

    # Normalize embeddings for Cosine Similarity
    # L2 normalize rows
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1e-10
    embeddings_norm = embeddings / norms

    # Convert to Tensor for GPU acceleration
    device = torch.device(Config.DEVICE)
    # Process in chunks to avoid OOM
    # We need to compute (N, D) @ (D, N) -> (N, N)
    # But we only keep Top-K, so we don't store (N, N)

    emb_tensor = torch.from_numpy(embeddings_norm).to(device, dtype=torch.float32)

    topk = Config.TOP_K_SIMILAR
    # We retrieve topk + 1 because the item itself will be the closest (sim=1.0)
    k_retrieve = topk + 1

    rows = []
    cols = []
    vals = []

    chunk_size = 1000  # Adjust based on GPU memory

    print(f"Calculating Top-{topk} neighbors using block multiplication...")

    with torch.no_grad():
        for start_idx in range(0, n_items, chunk_size):
            end_idx = min(start_idx + chunk_size, n_items)

            # Current chunk of query items
            chunk = emb_tensor[start_idx:end_idx]  # (Batch, D)

            # Compute similarity against ALL items
            # (Batch, D) @ (N, D).T -> (Batch, N)
            sim_scores = torch.matmul(chunk, emb_tensor.T)

            # Get Top K
            # values: (Batch, K), indices: (Batch, K)
            top_vals, top_inds = torch.topk(sim_scores, k=k_retrieve, dim=1)

            # Move to CPU
            top_vals = top_vals.cpu().numpy()
            top_inds = top_inds.cpu().numpy()

            # Prepare sparse format
            # For each item in chunk
            for i in range(len(chunk)):
                global_row_idx = start_idx + i

                # Filter out self-similarity (where index == global_row_idx)
                # We want exactly top_k items that are NOT the item itself

                row_indices = top_inds[i]
                row_values = top_vals[i]

                # Mask for non-self items
                mask = row_indices != global_row_idx

                valid_inds = row_indices[mask][:topk]
                valid_vals = row_values[mask][:topk]

                # Append to lists
                # Create row indices array for this item
                rows.append(np.full(len(valid_inds), global_row_idx, dtype=np.int32))
                cols.append(valid_inds)
                vals.append(valid_vals)

            if (start_idx // chunk_size) % 10 == 0:
                print(f"Processed chunk {start_idx}/{n_items}")

            # Clean up GPU memory
            del sim_scores, top_vals, top_inds
            # torch.cuda.empty_cache() # Optional, can slow down loop

    # Construct Sparse Matrix
    print("Constructing CSR matrix...")
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)

    sim_matrix = sp.csr_matrix((vals, (rows, cols)), shape=(n_items, n_items))

    # 3. Save Cache
    print(f"Saving similarity matrix to {cache_path}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    sp.save_npz(cache_path, sim_matrix)

    print(
        f"Visual Similarity Matrix ready. Shape: {sim_matrix.shape}, NNZ: {sim_matrix.nnz}"
    )
    return sim_matrix
