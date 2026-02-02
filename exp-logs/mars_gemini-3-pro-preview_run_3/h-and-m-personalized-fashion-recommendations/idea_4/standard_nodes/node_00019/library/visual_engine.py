import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from scipy import sparse
from library.config import Config
from library.data_utils import load_articles


class ArticleImageDataset(Dataset):
    """
    PyTorch Dataset for loading article images.
    Only includes articles that have valid image files.
    """

    def __init__(self, df, transform=None):
        self.transform = transform
        # Filter for valid paths only to avoid runtime errors during iteration
        # We store tuples of (dense_idx, path)
        self.samples = []

        # Pre-check existence is expensive for 100k files if done one by one,
        # but necessary to avoid crashing the DataLoader.
        # We assume the paths in df are relative to INPUT_DIR or are full paths.
        # Config.IMAGES_DIR is INPUT_DIR/images.
        # The 'image_path' in df is like 'images/010/...' relative to input.

        input_dir = Config.INPUT_DIR

        # Vectorized check is hard with pathlib, so we iterate.
        # To speed up, we just try to open in __getitem__ and handle errors,
        # but for batching, it's better to filter first.
        # Given the constraints, we will filter.

        valid_paths = []
        indices = []

        paths = df["image_path"].values
        idxs = df["article_idx"].values

        for idx, rel_path in zip(idxs, paths):
            full_path = input_dir / rel_path
            # We trust the metadata generation step which checked paths,
            # but we add a safety check for the file existence.
            if os.path.exists(full_path):
                self.samples.append((idx, str(full_path)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        article_idx, path = self.samples[idx]
        try:
            with open(path, "rb") as f:
                img = Image.open(f).convert("RGB")

            if self.transform:
                img = self.transform(img)

            return article_idx, img
        except Exception:
            # Return None or handle gracefully.
            # Since we can't easily skip in map-style dataset, we return a zero tensor
            # and handle it in collate or just return a dummy.
            # Ideally this shouldn't happen if we filtered correctly.
            return article_idx, torch.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE))


class VisualGraphBuilder:
    """
    Handles the extraction of visual features and construction of the
    Visual KNN graph for the retrieval stage.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Standard ImageNet transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _get_model(self):
        """
        Initializes the ResNet model for feature extraction.
        """
        # Create model with no classification head (num_classes=0)
        # This returns the pooled feature vector (e.g., 512 dim for ResNet18)
        model = timm.create_model("resnet18", pretrained=True, num_classes=0)
        model = model.to(self.device)
        model.eval()
        return model

    def extract_embeddings(self, load_cached_data: bool = True) -> np.ndarray:
        """
        Extracts image embeddings for all articles.

        Args:
            load_cached_data (bool): If True, tries to load from disk.

        Returns:
            np.ndarray: Matrix of shape (n_articles, embedding_dim).
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        if load_cached_data and Config.ARTICLE_EMBEDDINGS_PATH.exists():
            print(
                f"Loading cached article embeddings from {Config.ARTICLE_EMBEDDINGS_PATH}"
            )
            return np.load(Config.ARTICLE_EMBEDDINGS_PATH)

        print("Extracting article embeddings (this may take a while)...")

        # 1. Load Article Metadata
        articles_df, _ = load_articles(load_cached_data=True)
        num_articles = len(articles_df)

        # 2. Setup Dataset and DataLoader
        dataset = ArticleImageDataset(articles_df, transform=self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        # 3. Inference
        model = self._get_model()

        # Initialize embedding matrix with zeros
        # Articles without images will remain zero vectors
        embeddings = np.zeros((num_articles, Config.EMBEDDING_DIM), dtype=np.float32)

        with torch.no_grad():
            for batch_idxs, batch_imgs in dataloader:
                batch_imgs = batch_imgs.to(self.device)

                # Forward pass
                features = model(batch_imgs)

                # Move to CPU and numpy
                features_np = features.cpu().numpy()

                # Assign to matrix
                # batch_idxs is a tensor of indices
                for i, idx in enumerate(batch_idxs):
                    embeddings[idx.item()] = features_np[i]

        # 4. Save
        print(f"Saving embeddings to {Config.ARTICLE_EMBEDDINGS_PATH}")
        np.save(Config.ARTICLE_EMBEDDINGS_PATH, embeddings)

        return embeddings

    def build_knn_graph(self, load_cached_data: bool = True) -> sparse.csr_matrix:
        """
        Constructs the Visual KNN graph.

        Args:
            load_cached_data (bool): If True, tries to load from disk.

        Returns:
            sparse.csr_matrix: Adjacency matrix where T_ij is the similarity.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        if load_cached_data and Config.VISUAL_KNN_GRAPH_PATH.exists():
            print(
                f"Loading cached Visual KNN graph from {Config.VISUAL_KNN_GRAPH_PATH}"
            )
            return sparse.load_npz(Config.VISUAL_KNN_GRAPH_PATH)

        print("Building Visual KNN graph...")

        # 1. Get Embeddings
        embeddings = self.extract_embeddings(load_cached_data=load_cached_data)

        # 2. Normalize for Cosine Similarity
        # L2 normalization allows us to use Euclidean distance to find Cosine neighbors
        # Cosine Sim(A, B) = Dot(A_norm, B_norm)
        # Euclidean Dist(A_norm, B_norm)^2 = 2 - 2 * Cosine Sim
        # So smallest Euclidean distance corresponds to largest Cosine Similarity
        embeddings_norm = normalize(embeddings, norm="l2", axis=1)

        # 3. Fit Nearest Neighbors
        # We use 'euclidean' on normalized data for efficiency
        nbrs = NearestNeighbors(
            n_neighbors=Config.KNN_NEIGHBORS,
            algorithm="auto",
            metric="euclidean",
            n_jobs=-1,
        )
        nbrs.fit(embeddings_norm)

        # 4. Find Neighbors
        # distances is euclidean distance, indices is article indices
        distances, indices = nbrs.kneighbors(embeddings_norm)

        # 5. Convert to Similarity
        # Convert Euclidean distance back to Cosine Similarity
        # dist^2 = 2 - 2*sim  =>  sim = 1 - 0.5 * dist^2
        # However, for sparse graph propagation, we just want a weight.
        # 1 - distance is a rough proxy, or we can compute exact dot products.
        # Let's compute exact dot products for the found neighbors to be precise.

        n_samples = embeddings.shape[0]
        row_ind = []
        col_ind = []
        data = []

        # Vectorized construction of CSR data
        # We iterate to compute dot products efficiently or just use the formula
        # sim = 1 - 0.5 * dist**2

        sims = 1.0 - 0.5 * (distances**2)

        # Flatten
        row_ind = np.repeat(np.arange(n_samples), Config.KNN_NEIGHBORS)
        col_ind = indices.flatten()
        data = sims.flatten()

        # Filter out self-loops or low similarity if needed?
        # Usually we keep them for the graph, but maybe zero out self-loops for retrieval?
        # The prompt implies "Sparse K-Nearest Neighbors graph".
        # We will keep it as is.

        # Handle cases where embeddings were zero (missing images)
        # These will have neighbors (likely other zero vectors or randoms depending on init),
        # but their similarity will be undefined or 0.
        # If embedding is 0, norm is 0. Dot product is 0.
        # We should mask out rows where the source embedding was zero.
        norms = np.linalg.norm(embeddings, axis=1)
        valid_mask = norms > 1e-6

        # Expand mask to match flattened data
        valid_rows = np.repeat(valid_mask, Config.KNN_NEIGHBORS)

        # Apply mask
        row_ind = row_ind[valid_rows]
        col_ind = col_ind[valid_rows]
        data = data[valid_rows]

        # 6. Create CSR Matrix
        knn_graph = sparse.csr_matrix(
            (data, (row_ind, col_ind)), shape=(n_samples, n_samples), dtype=np.float32
        )

        # 7. Save
        print(f"Saving Visual KNN graph to {Config.VISUAL_KNN_GRAPH_PATH}")
        sparse.save_npz(Config.VISUAL_KNN_GRAPH_PATH, knn_graph)

        return knn_graph
