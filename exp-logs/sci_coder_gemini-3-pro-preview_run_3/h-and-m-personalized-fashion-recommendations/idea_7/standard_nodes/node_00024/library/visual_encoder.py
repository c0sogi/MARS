import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import timm
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from library.config import Config
from library.data_utils import get_id_maps

# Set random seeds for reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.SEED)


class ArticleImageDataset(Dataset):
    """
    PyTorch Dataset to load images for articles.
    Returns a zero-tensor if the image is missing.
    """

    def __init__(self, article_ids, input_dir, transform=None):
        self.article_ids = article_ids
        self.input_dir = input_dir
        self.transform = transform

        # Pre-calculate paths logic to match metadata generation
        # Format: images/xxx/0xxxxxxxx.jpg
        self.paths = []
        for aid in self.article_ids:
            s = str(aid).zfill(10)
            folder = s[:3]
            filename = s + ".jpg"
            self.paths.append(os.path.join(self.input_dir, "images", folder, filename))

    def __len__(self):
        return len(self.article_ids)

    def __getitem__(self, idx):
        path = self.paths[idx]

        img = None
        if os.path.exists(path):
            try:
                # Read image using OpenCV
                img = cv2.imread(path)
                if img is not None:
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception:
                img = None

        if img is None:
            # Return zero tensor if image missing or corrupt
            # Shape 3, H, W (after transform it would be this shape)
            # We return a blank image that transform handles, or handle manually
            # Easiest is to return a black image of correct size
            img = np.zeros(
                (Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 3), dtype=np.uint8
            )
            is_valid = 0.0
        else:
            is_valid = 1.0

        if self.transform:
            img = self.transform(img)

        return img, is_valid


class ImageEmbedder:
    """
    Handles the generation of image embeddings using a pre-trained CNN.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = Config.IMAGE_BATCH_SIZE

        # Define Transforms
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(Config.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _load_model(self):
        print(f"Loading pre-trained {Config.IMAGE_MODEL_NAME} model...")
        model = timm.create_model(
            Config.IMAGE_MODEL_NAME, pretrained=True, num_classes=0
        )
        model = model.to(self.device)
        model.eval()
        return model

    def generate_embeddings(self, load_cached_data=True):
        """
        Generates embeddings for all articles in the ID map.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        if load_cached_data and Config.CACHE_ARTICLE_EMBEDDINGS.exists():
            print(
                f"Loading cached article embeddings from {Config.CACHE_ARTICLE_EMBEDDINGS}..."
            )
            return np.load(Config.CACHE_ARTICLE_EMBEDDINGS)

        print("Generating article embeddings from scratch...")

        # Get Article IDs to ensure alignment
        _, _, _, article_id_map = get_id_maps(load_cached_data=True)

        # Setup Dataset and Loader
        dataset = ArticleImageDataset(
            article_id_map, Config.INPUT_DIR, transform=self.transform
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        model = self._load_model()

        embeddings = []
        valid_flags = []

        print(f"Starting inference on {len(article_id_map)} articles...")
        with torch.no_grad():
            for imgs, is_valid in dataloader:
                imgs = imgs.to(self.device)
                features = model(imgs)

                # Move to CPU
                emb_batch = features.cpu().numpy()
                valid_batch = is_valid.numpy()

                # Zero out embeddings for invalid images explicitly (just in case model output noise on black img)
                # ResNet on black image might not be exactly zero vector
                for i in range(len(emb_batch)):
                    if valid_batch[i] == 0:
                        emb_batch[i] = np.zeros_like(emb_batch[i])

                embeddings.append(emb_batch)

        # Concatenate
        full_embeddings = np.concatenate(embeddings, axis=0)

        # Save
        print(f"Saving embeddings to {Config.CACHE_ARTICLE_EMBEDDINGS}...")
        np.save(Config.CACHE_ARTICLE_EMBEDDINGS, full_embeddings)

        return full_embeddings


def build_visual_graph(load_cached_data=True):
    """
    Constructs the sparse Visual Graph (KNN) based on image embeddings.

    Returns:
        scipy.sparse.csr_matrix: The visual transition matrix T_vis.
                                 Shape: (n_articles, n_articles)
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if load_cached_data and Config.CACHE_VISUAL_GRAPH.exists():
        print(f"Loading cached Visual Graph from {Config.CACHE_VISUAL_GRAPH}...")
        return sparse.load_npz(Config.CACHE_VISUAL_GRAPH)

    print("Building Visual Graph from scratch...")

    # 1. Get Embeddings
    embedder = ImageEmbedder()
    embeddings = embedder.generate_embeddings(load_cached_data=load_cached_data)

    num_articles = embeddings.shape[0]

    # 2. Filter Valid Embeddings
    # We only want to find neighbors for items that actually have images.
    # Zero vectors (missing images) should not participate in KNN as queries or targets.
    norms = np.linalg.norm(embeddings, axis=1)
    valid_mask = norms > 1e-6
    valid_indices = np.where(valid_mask)[0]

    print(
        f"Found {len(valid_indices)} articles with valid images out of {num_articles}."
    )

    if len(valid_indices) == 0:
        print("Warning: No valid images found. Returning empty graph.")
        visual_graph = sparse.csr_matrix((num_articles, num_articles), dtype=np.float32)
        sparse.save_npz(Config.CACHE_VISUAL_GRAPH, visual_graph)
        return visual_graph

    valid_embeddings = embeddings[valid_indices]

    # 3. Normalize for Cosine Similarity
    # Cosine Similarity is equivalent to dot product of L2-normalized vectors
    valid_embeddings_norm = normalize(valid_embeddings, axis=1, norm="l2")

    # 4. KNN Search
    print(f"Fitting NearestNeighbors (k={Config.VISUAL_KNN_K})...")
    # We use brute force or auto. With 100k items, brute force is acceptable on efficient implementations,
    # but 'algorithm="auto"' usually picks a tree for lower dims.
    # Metric='cosine' is often slower in sklearn than euclidean on normalized data.
    # Euclidean distance on normalized vectors ranks same as Cosine Similarity.
    # We use dot product via matrix multiplication if we want raw speed, but sklearn gives us the graph structure easily.
    # Let's use sklearn with cosine metric for simplicity of implementation.
    knn = NearestNeighbors(
        n_neighbors=Config.VISUAL_KNN_K,
        metric="cosine",
        algorithm="brute",
        n_jobs=Config.NUM_WORKERS,
    )
    knn.fit(valid_embeddings_norm)

    # Find neighbors
    # distances are cosine distances (1 - similarity)
    distances, neighbor_indices_local = knn.kneighbors(valid_embeddings_norm)

    # 5. Convert Distances to Similarities
    # Cosine Distance = 1 - Cosine Similarity
    # Similarity = 1 - Distance
    similarities = 1.0 - distances

    # 6. Construct Sparse Matrix (Map back to global indices)
    # We have neighbors in the space of 'valid_indices'. We need to map them to 'global_indices'.

    # Prepare COO format data
    row_indices = []
    col_indices = []
    data_values = []

    print("Constructing sparse adjacency matrix...")
    # Vectorized construction
    # row_indices_local: [0, 0, ..., 1, 1, ...]
    n_valid = len(valid_indices)
    k = Config.VISUAL_KNN_K

    # Create arrays of shape (n_valid, k)
    # The row index in the local space corresponds to valid_indices[row] in global space
    global_row_ids = valid_indices.repeat(k)

    # The neighbor index in local space corresponds to valid_indices[neighbor] in global space
    flat_neighbor_indices_local = neighbor_indices_local.flatten()
    global_col_ids = valid_indices[flat_neighbor_indices_local]

    flat_similarities = similarities.flatten()

    # Create the matrix
    visual_graph = sparse.csr_matrix(
        (flat_similarities, (global_row_ids, global_col_ids)),
        shape=(num_articles, num_articles),
        dtype=np.float32,
    )

    # 7. Save
    print(f"Saving Visual Graph to {Config.CACHE_VISUAL_GRAPH}...")
    sparse.save_npz(Config.CACHE_VISUAL_GRAPH, visual_graph)

    return visual_graph
