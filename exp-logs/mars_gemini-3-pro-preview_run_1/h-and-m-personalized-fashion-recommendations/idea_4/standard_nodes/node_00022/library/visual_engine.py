import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import scipy.sparse as sp
from tqdm import tqdm

from library.config import (
    IMAGES_DIR,
    VISUAL_EMBEDDINGS_PATH,
    VISUAL_MATRIX_PATH,
    EMBEDDING_DIM,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    TOP_K_VISUAL,
    SEED,
)

# Set fixed seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class ArticleImageDataset(Dataset):
    """
    Dataset class to load images for active articles.
    """

    def __init__(self, article_ids, input_dir, transform=None):
        self.article_ids = article_ids
        self.input_dir = input_dir
        self.transform = transform
        self.empty_tensor = torch.zeros(3, IMAGE_SIZE[0], IMAGE_SIZE[1])

    def __len__(self):
        return len(self.article_ids)

    def __getitem__(self, idx):
        article_id = self.article_ids[idx]

        # Format: 0108775015 (10 digits) -> folder 010 -> file 0108775015.jpg
        s_id = f"{int(article_id):010d}"
        folder = s_id[:3]
        img_path = os.path.join(self.input_dir, folder, f"{s_id}.jpg")

        if os.path.exists(img_path):
            try:
                with open(img_path, "rb") as f:
                    img = Image.open(f).convert("RGB")

                if self.transform:
                    return self.transform(img)
                return img
            except Exception:
                # Fallback for corrupt images
                pass

        # Return zero tensor (normalized later) if image missing
        # Note: Transform usually expects PIL Image, but if we return tensor here,
        # the collate_fn needs to handle it or we apply transform manually to a black image.
        # Safer to create a black PIL image.
        img = Image.new("RGB", IMAGE_SIZE)
        if self.transform:
            return self.transform(img)
        return transforms.ToTensor()(img)


class ImageEmbedder:
    """
    Extracts embeddings for items using ResNet50.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Standard ResNet preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _get_model(self):
        # Load ResNet50
        model = models.resnet50(weights="DEFAULT")
        # Remove classification head (fc layer) to get 2048-dim features
        model.fc = nn.Identity()
        model.to(self.device)
        model.eval()
        return model

    def extract_embeddings(self, mapper, load_cached_data=True):
        """
        Generates or loads embeddings for all items in the mapper.

        Args:
            mapper: IndexMapper instance containing item mappings.
            load_cached_data (bool): Whether to use cached .npy file.

        Returns:
            np.ndarray: Matrix of shape (n_items, 2048).
        """
        # Ensure working directory exists
        os.makedirs(os.path.dirname(VISUAL_EMBEDDINGS_PATH), exist_ok=True)

        if load_cached_data and os.path.exists(VISUAL_EMBEDDINGS_PATH):
            print(f"Loading visual embeddings from {VISUAL_EMBEDDINGS_PATH}...")
            embeddings = np.load(VISUAL_EMBEDDINGS_PATH)
            # Verify shape matches current mapper
            if embeddings.shape[0] == mapper.get_num_items():
                return embeddings
            print("Cached embeddings shape mismatch. Recomputing...")

        print("Extracting visual embeddings from scratch...")

        # Prepare list of article_ids sorted by item_idx (0 to N-1)
        n_items = mapper.get_num_items()
        sorted_article_ids = [mapper.idx2item[i] for i in range(n_items)]

        dataset = ArticleImageDataset(
            sorted_article_ids, IMAGES_DIR, transform=self.transform
        )

        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        model = self._get_model()
        embeddings_list = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting Embeddings"):
                batch = batch.to(self.device)
                features = model(batch)
                # Flatten if necessary (ResNet output before FC is (B, 2048))
                features = features.view(features.size(0), -1)
                embeddings_list.append(features.cpu().numpy())

        # Concatenate
        all_embeddings = np.vstack(embeddings_list)

        # Handle cases where image was missing (zero vectors)
        # We leave them as zeros; they will have 0 similarity later.

        print(f"Saving embeddings to {VISUAL_EMBEDDINGS_PATH}...")
        np.save(VISUAL_EMBEDDINGS_PATH, all_embeddings)

        return all_embeddings


class VisualSimilarityBuilder:
    """
    Constructs the sparse visual similarity matrix from embeddings.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_similarity_matrix(self, embeddings, load_cached_data=True):
        """
        Computes cosine similarity and returns a sparse matrix (Top-K).

        Args:
            embeddings (np.ndarray): (n_items, 2048) array.
            load_cached_data (bool): Whether to load from cache.

        Returns:
            scipy.sparse.csr_matrix: The visual similarity matrix.
        """
        # Ensure working directory exists
        os.makedirs(os.path.dirname(VISUAL_MATRIX_PATH), exist_ok=True)

        if load_cached_data and os.path.exists(VISUAL_MATRIX_PATH):
            print(f"Loading visual similarity matrix from {VISUAL_MATRIX_PATH}...")
            return sp.load_npz(VISUAL_MATRIX_PATH)

        print("Computing visual similarity matrix...")

        n_items = embeddings.shape[0]

        # Convert to torch for GPU acceleration
        tensor_embeddings = torch.from_numpy(embeddings).float().to(self.device)

        # Normalize embeddings to unit length for Cosine Similarity
        # (Cosine Sim = Dot product of normalized vectors)
        # Add epsilon to avoid div by zero for missing images (zero vectors)
        norm = torch.norm(tensor_embeddings, p=2, dim=1, keepdim=True)
        tensor_embeddings = tensor_embeddings / (norm + 1e-8)

        # We need to compute Top-K for each row.
        # Doing full N x N multiplication might OOM if N is large.
        # We process in chunks.

        chunk_size = 1000  # Adjust based on GPU memory

        row_indices = []
        col_indices = []
        values = []

        # Process rows in chunks
        for i in tqdm(range(0, n_items, chunk_size), desc="Computing Similarity"):
            end = min(i + chunk_size, n_items)

            # Chunk of queries: (B, D)
            query_chunk = tensor_embeddings[i:end]

            # Compute similarity: (B, D) @ (D, N) -> (B, N)
            sim_scores = torch.matmul(query_chunk, tensor_embeddings.t())

            # Mask self-similarity (diagonal)
            # The diagonal element for row r is at column r.
            # In the local chunk (size B x N), row k corresponds to global index i+k.
            # We want to set sim_scores[k, i+k] to -1 (or very low) so it's not picked.

            # Create a range for rows in this chunk
            local_rows = torch.arange(end - i, device=self.device)
            global_cols = torch.arange(i, end, device=self.device)

            sim_scores[local_rows, global_cols] = -1.0

            # Get Top-K
            # k is TOP_K_VISUAL
            top_vals, top_inds = torch.topk(sim_scores, k=TOP_K_VISUAL, dim=1)

            # Move to CPU and store
            top_vals = top_vals.cpu().numpy().flatten()
            top_inds = top_inds.cpu().numpy().flatten()

            # Generate row indices
            # For each row in chunk, we have K values.
            # rows: [i, i, ..., i+1, i+1, ...]
            rows = np.repeat(np.arange(i, end), TOP_K_VISUAL)

            row_indices.append(rows)
            col_indices.append(top_inds)
            values.append(top_vals)

        # Concatenate all parts
        row_indices = np.concatenate(row_indices)
        col_indices = np.concatenate(col_indices)
        values = np.concatenate(values)

        # Filter out negative correlations or zero similarities (from missing images)
        mask = values > 0.0
        row_indices = row_indices[mask]
        col_indices = col_indices[mask]
        values = values[mask]

        # Build CSR matrix
        print("Constructing sparse matrix...")
        sim_matrix = sp.csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(n_items, n_items),
            dtype=np.float32,
        )

        print(f"Saving visual similarity matrix to {VISUAL_MATRIX_PATH}...")
        sp.save_npz(VISUAL_MATRIX_PATH, sim_matrix)

        return sim_matrix
