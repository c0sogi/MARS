import os
import torch
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from library.config import Config
from library.model import WhaleModel
from library.utils import save_submission


class InferenceEngine:
    """
    Handles the inference pipeline for Whale Identification.
    Includes feature extraction, caching, query expansion, and submission generation.
    """

    def __init__(self, checkpoint_path=None):
        """
        Args:
            checkpoint_path (str): Path to the model checkpoint.
                                   Defaults to Config.model_save_path.
        """
        self.device = Config.device
        self.checkpoint_path = (
            checkpoint_path if checkpoint_path else Config.model_save_path
        )

        # Initialize Model Architecture
        # We don't need the classification head for inference, just the backbone + neck
        self.model = WhaleModel(num_classes=None).to(self.device)
        self.model.eval()

        # Load Weights
        if os.path.exists(self.checkpoint_path):
            print(f"Loading model from {self.checkpoint_path}...")
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)

            # Handle potential mismatch if loading a model saved with a head into a model without one
            # or vice versa. The state_dict keys might need filtering.
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
        else:
            print(
                f"Warning: Checkpoint not found at {self.checkpoint_path}. Using random weights."
            )

    def extract_features(
        self, loader, cache_emb_path, cache_label_path, load_cached_data=True
    ):
        """
        Extracts embeddings from a data loader, with caching.

        Args:
            loader (DataLoader): The data loader to extract features from.
            cache_emb_path (str): Path to save/load embeddings .npy file.
            cache_label_path (str): Path to save/load labels/names .npy file.
            load_cached_data (bool): If True, attempts to load from cache first.

        Returns:
            embeddings (np.array): Normalized embeddings.
            targets (np.array): Labels or image names.
        """
        # 1. Check Cache
        if (
            load_cached_data
            and os.path.exists(cache_emb_path)
            and os.path.exists(cache_label_path)
        ):
            print(f"Loading cached embeddings from {cache_emb_path}")
            embeddings = np.load(cache_emb_path)
            targets = np.load(cache_label_path, allow_pickle=True)
            return embeddings, targets

        # 2. Compute Features
        print(f"Computing embeddings (Cache miss or force reload)...")
        embeddings = []
        targets = []

        with torch.no_grad():
            for batch in loader:
                images = batch[0].to(self.device)
                labels = batch[1]  # Can be tensor (int labels) or tuple (filenames)

                # Forward pass (returns normalized embeddings)
                emb = self.model(images)
                embeddings.append(emb.cpu().numpy())

                # Handle labels
                if isinstance(labels, torch.Tensor):
                    targets.append(labels.numpy())
                else:
                    targets.extend(labels)

        if len(embeddings) > 0:
            embeddings = np.vstack(embeddings)

            if len(targets) > 0:
                if isinstance(targets[0], (int, float, np.integer)):
                    targets = np.concatenate(targets)
                elif isinstance(targets[0], np.ndarray):
                    targets = np.concatenate(targets)
                else:
                    targets = np.array(targets)
        else:
            embeddings = np.array([])
            targets = np.array([])

        # 3. Save to Cache
        os.makedirs(os.path.dirname(cache_emb_path), exist_ok=True)
        np.save(cache_emb_path, embeddings)
        np.save(cache_label_path, targets)
        print(f"Saved embeddings to {cache_emb_path}")

        return embeddings, targets

    def predict_with_qe(
        self, test_loader, gallery_loader, encoder, load_cached_data=True
    ):
        """
        Performs inference using Query Expansion and Thresholding.

        Args:
            test_loader (DataLoader): DataLoader for test images.
            gallery_loader (DataLoader): DataLoader for gallery (train) images.
            encoder (LabelEncoder): Fitted encoder to map ints to whale IDs.
            load_cached_data (bool): Whether to use cached embeddings.
        """
        print("Starting Inference with Query Expansion...")

        # 1. Extract Embeddings for Gallery (Train)
        gallery_emb, gallery_labels = self.extract_features(
            gallery_loader,
            Config.train_embeddings_path,
            Config.train_labels_path,
            load_cached_data=load_cached_data,
        )

        # 2. Extract Embeddings for Query (Test)
        test_emb, test_names = self.extract_features(
            test_loader,
            Config.test_embeddings_path,
            Config.test_names_path,
            load_cached_data=load_cached_data,
        )

        if len(gallery_emb) == 0 or len(test_emb) == 0:
            print("Error: Empty embeddings. Cannot proceed with inference.")
            return

        # 3. Initialize KNN
        # Metric: Cosine Similarity (equivalent to Euclidean on normalized vectors)
        # We use cosine metric in NearestNeighbors for clarity
        knn = NearestNeighbors(n_neighbors=Config.knn_k, metric="cosine", n_jobs=-1)
        knn.fit(gallery_emb)

        # 4. Initial Query
        print("Performing initial neighbor search...")
        dists, indices = knn.kneighbors(test_emb)

        final_preds = []

        # 5. Query Expansion & Prediction Loop
        print(f"Processing {len(test_emb)} test queries...")
        for i in range(len(test_emb)):

            # --- Query Expansion ---
            if Config.use_qe:
                # Retrieve top K neighbors for expansion
                neighbor_indices = indices[i, : Config.qe_k]
                neighbor_embs = gallery_emb[neighbor_indices]

                # Compute average embedding (Query + Neighbors)
                # Note: test_emb[i] is already normalized, neighbor_embs are normalized.
                # A simple mean works as a centroid.
                expanded_query = test_emb[i] + np.mean(neighbor_embs, axis=0)

                # Re-normalize
                norm = np.linalg.norm(expanded_query)
                expanded_query = expanded_query / (norm + 1e-6)

                query_vec = expanded_query.reshape(1, -1)
            else:
                query_vec = test_emb[i].reshape(1, -1)

            # --- Re-Query with Refined Embedding ---
            # We need top 5 predictions, but we fetch a few more to handle duplicates
            q_dists, q_indices = knn.kneighbors(query_vec, n_neighbors=10)

            top_indices = q_indices[0]
            top_dists = q_dists[
                0
            ]  # Cosine distance (0 to 2, usually 0 to 1 for normalized)

            # Map indices to Class IDs
            top_class_ints = gallery_labels[top_indices]
            top_class_strs = encoder.inverse_transform(top_class_ints)

            # --- Thresholding & Formatting ---
            # Cosine Distance = 1 - Cosine Similarity
            # Similarity = 1 - Distance
            top_dist = top_dists[0]
            top_sim = 1.0 - top_dist

            current_preds = []

            # If the best match is weak, predict 'new_whale' first
            if top_sim < Config.unknown_threshold:
                current_preds.append("new_whale")

            # Add retrieved known whales
            for label in top_class_strs:
                if label not in current_preds:
                    current_preds.append(label)

            # If 'new_whale' wasn't added first, we can add it at the end if space permits
            # (The competition metric allows 5 predictions. 'new_whale' is a valid label)
            if "new_whale" not in current_preds:
                current_preds.append("new_whale")

            # Truncate to top 5
            final_preds.append(current_preds[:5])

        # 6. Save Submission
        save_submission(test_names, final_preds, Config.submission_path)
        print(f"Inference complete. Submission generated at {Config.submission_path}")
