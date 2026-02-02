import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from scipy import sparse
from library import config, data_manager

# Set deterministic behavior
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)


class ArticleImageDataset(Dataset):
    """
    Dataset to load images for articles.
    Returns a zero-tensor if the image is missing.
    """

    def __init__(self, article_ids, img_dir, transform=None):
        self.article_ids = article_ids
        self.img_dir = img_dir
        self.transform = transform

        # Pre-calculate paths to avoid doing it in __getitem__
        # article_id is int64, need to convert to string zfill(10)
        # folder is first 3 chars
        self.paths = []
        for aid in self.article_ids:
            s = str(aid).zfill(10)
            folder = s[:3]
            filename = f"{s}.jpg"
            self.paths.append(os.path.join(self.img_dir, folder, filename))

    def __len__(self):
        return len(self.article_ids)

    def __getitem__(self, idx):
        path = self.paths[idx]

        # Check existence
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    img = Image.open(f).convert("RGB")

                if self.transform:
                    img = self.transform(img)
                return img, 1  # 1 indicates valid image
            except Exception:
                # Corrupt file
                pass

        # Return zero tensor if missing or corrupt
        # Shape must match transform output (3, 224, 224)
        return (
            torch.zeros(
                (3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]), dtype=torch.float32
            ),
            0,
        )  # 0 indicates invalid


def get_image_transforms():
    """Returns the standard ImageNet normalization and resizing transforms."""
    return transforms.Compose(
        [
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def extract_embeddings(load_cached_data=True):
    """
    Generates or loads image embeddings for all articles.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        np.ndarray: Matrix of shape (num_articles, embedding_dim)
    """
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(config.IMAGE_EMBEDDINGS_PATH):
        print(f"Loading image embeddings from {config.IMAGE_EMBEDDINGS_PATH}...")
        return np.load(config.IMAGE_EMBEDDINGS_PATH)

    print("Generating image embeddings from scratch...")

    # 1. Get Article IDs (sorted)
    _, _, _, idx_to_article = data_manager.get_id_mappings(load_cached_data=True)

    # 2. Setup Model
    device = config.DEVICE
    print(f"Using device: {device}")

    # Load ResNet18
    model = models.resnet18(pretrained=True)

    # Remove classification head (fc layer)
    # ResNet18 structure: ... avgpool -> fc. We want output of avgpool.
    # We can replace fc with Identity or just take the features.
    # A cleaner way is to wrap it.
    class FeatureExtractor(nn.Module):
        def __init__(self, original_model):
            super(FeatureExtractor, self).__init__()
            self.features = nn.Sequential(*list(original_model.children())[:-1])

        def forward(self, x):
            x = self.features(x)
            return torch.flatten(x, 1)

    model = FeatureExtractor(model)
    model.to(device)
    model.eval()

    # 3. Setup DataLoader
    dataset = ArticleImageDataset(
        article_ids=idx_to_article,
        img_dir=config.IMAGES_DIR,
        transform=get_image_transforms(),
    )

    loader = DataLoader(
        dataset,
        batch_size=config.IMAGE_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Inference Loop
    embeddings = []
    valid_mask = []

    print(f"Starting inference on {len(dataset)} images...")

    with torch.no_grad():
        for i, (imgs, valids) in enumerate(loader):
            imgs = imgs.to(device)

            # Forward pass
            emb_batch = model(imgs)

            # Move to CPU
            embeddings.append(emb_batch.cpu().numpy())
            valid_mask.append(valids.numpy())

            if (i + 1) % 100 == 0:
                print(f"Processed batch {i + 1}/{len(loader)}")

    # Concatenate
    embeddings = np.concatenate(embeddings, axis=0)  # (N, 512)
    valid_mask = np.concatenate(valid_mask, axis=0)  # (N,)

    # Zero out embeddings for invalid images (just to be safe, though they were zero inputs)
    # ResNet on zero input might not produce exactly zero output due to bias/normalization.
    # We explicitly zero them out to ensure they don't affect cosine similarity.
    embeddings[valid_mask == 0] = 0.0

    print(f"Embeddings generated. Shape: {embeddings.shape}")
    print(f"Valid images found: {np.sum(valid_mask)} / {len(valid_mask)}")

    # 5. Save
    np.save(config.IMAGE_EMBEDDINGS_PATH, embeddings)
    print(f"Saved embeddings to {config.IMAGE_EMBEDDINGS_PATH}")

    return embeddings


def build_visual_graph(load_cached_data=True):
    """
    Constructs the sparse KNN graph based on visual similarity.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        scipy.sparse.csr_matrix: The visual transition matrix T_vis.
    """
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    path = config.VISUAL_GRAPH_PATH

    if load_cached_data and os.path.exists(path):
        print(f"Loading visual graph from {path}...")
        return sparse.load_npz(path)

    print("Building visual graph...")

    # 1. Load Embeddings
    embeddings = extract_embeddings(load_cached_data=True)
    n_items, dim = embeddings.shape

    # 2. Normalize for Cosine Similarity
    # L2 Normalize: x / ||x||
    # Handle zero vectors (missing images) to avoid division by zero
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Avoid div by zero, vector remains zero
    embeddings_norm = embeddings / norms

    # Convert to torch for GPU acceleration
    device = config.DEVICE
    tensor_emb = torch.from_numpy(embeddings_norm).to(
        device, dtype=torch.float16
    )  # FP16 for speed/mem

    # 3. Block-wise KNN
    # We can't compute (N, N) matrix. We do (Batch, N).
    k = config.VISUAL_KNN_K
    batch_size = 1024  # Adjust based on GPU memory

    row_indices = []
    col_indices = []
    values = []

    num_batches = (n_items + batch_size - 1) // batch_size
    print(f"Computing KNN in {num_batches} batches...")

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_items)

        # Current batch: (B, D)
        batch = tensor_emb[start_idx:end_idx]

        # Compute Similarity: (B, D) @ (D, N) -> (B, N)
        # Result is cosine similarity since vectors are normalized
        sim_matrix = torch.matmul(batch, tensor_emb.T)

        # Top-K
        # We want top K neighbors.
        # Note: The item itself will be the top-1 (sim=1.0).
        # Usually in retrieval graphs we might want to exclude self-loops,
        # but for propagation, self-loops can be handled by the ranker or ignored.
        # Let's keep top K+1 and remove self later if needed, or just keep top K including self.
        # Given the task description implies finding *other* items, but standard KNN includes self.
        # We will retrieve Top K.
        top_vals, top_inds = torch.topk(sim_matrix, k=k, dim=1)

        # Move to CPU
        top_vals = top_vals.float().cpu().numpy()
        top_inds = top_inds.cpu().numpy()

        # Prepare sparse matrix data
        # Rows: range(start, end) repeated K times
        # Cols: top_inds flattened
        # Data: top_vals flattened

        rows = np.arange(start_idx, end_idx).repeat(k)
        cols = top_inds.flatten()
        vals = top_vals.flatten()

        # Filter out zero-similarity edges (caused by missing images)
        # If an image is missing, its embedding is 0, dot product is 0.
        mask = vals > 1e-6

        row_indices.append(rows[mask])
        col_indices.append(cols[mask])
        values.append(vals[mask])

        if (i + 1) % 20 == 0:
            print(f"Processed KNN batch {i + 1}/{num_batches}")

    # 4. Construct CSR Matrix
    print("Constructing sparse matrix...")
    row_indices = np.concatenate(row_indices)
    col_indices = np.concatenate(col_indices)
    values = np.concatenate(values)

    # Shape (N, N)
    visual_graph = sparse.csr_matrix(
        (values, (row_indices, col_indices)), shape=(n_items, n_items)
    )

    print(f"Visual Graph constructed. Edges: {visual_graph.nnz}")

    # 5. Save
    sparse.save_npz(path, visual_graph)
    print(f"Saved visual graph to {path}")

    return visual_graph
