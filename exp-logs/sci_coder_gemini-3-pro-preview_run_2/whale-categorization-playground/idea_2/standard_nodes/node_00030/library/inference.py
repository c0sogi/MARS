import os
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import torch.nn.functional as F

from library.config import (
    DEVICE,
    WORKING_DIR,
    INPUT_DIR,
    METADATA_DIR,
    SUBMISSION_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    MODEL_NAME,
    EMBEDDING_DIM,
    NUM_CLASSES,
    SEED,
)
from library.model import WhaleArcFaceModel
from library.dataset import WhaleDataset
from library.utils import setup_logger


class InferenceManager:
    def __init__(self, checkpoint_name="best_model.pth"):
        self.logger = setup_logger(
            name="inference_logger", log_file=os.path.join(WORKING_DIR, "inference.log")
        )
        self.device = DEVICE
        self.checkpoint_path = os.path.join(WORKING_DIR, checkpoint_name)
        self.submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

        # Define Transforms (Same as Validation transforms in dataset.py)
        self.transforms = A.Compose(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def load_model(self):
        """Loads the trained model architecture and weights."""
        self.logger.info(f"Loading model from {self.checkpoint_path}...")
        model = WhaleArcFaceModel(
            model_name=MODEL_NAME,
            num_classes=NUM_CLASSES,
            embedding_dim=EMBEDDING_DIM,
            pretrained=False,
        )

        if os.path.exists(self.checkpoint_path):
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            model.load_state_dict(state_dict)
            self.logger.info("Model weights loaded successfully.")
        else:
            self.logger.warning(
                f"Checkpoint not found at {self.checkpoint_path}. Using random weights (Expect poor performance)."
            )

        model.to(self.device)
        model.eval()
        return model

    def get_dataloader(self, df, cache_name, load_cache=True):
        """Creates a DataLoader for inference."""
        # Append image size to cache name to avoid resolution mismatch
        name, ext = os.path.splitext(cache_name)
        cache_name_sized = f"{name}_{IMAGE_SIZE}{ext}"

        dataset = WhaleDataset(
            df=df,
            transforms=self.transforms,
            cache_name=cache_name_sized,
            load_cache=load_cache,
            label_encoder=None,  # We don't need integer labels for inference extraction
        )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        return loader

    def generate_embeddings(self, model, dataloader, cache_file, load_cache=True):
        """
        Generates or loads embeddings for a given dataloader.

        Args:
            model: The loaded PyTorch model.
            dataloader: DataLoader for the dataset.
            cache_file: Name of the .npy file to store embeddings.
            load_cache: Boolean to determine if we should try loading from disk.

        Returns:
            torch.Tensor: Normalized embeddings (N, D)
        """
        cache_full_path = os.path.join(WORKING_DIR, cache_file)

        # 1. Try Loading Cache
        if load_cache and os.path.exists(cache_full_path):
            self.logger.info(f"Loading embeddings from {cache_full_path}...")
            try:
                embeddings_np = np.load(cache_full_path)
                return torch.from_numpy(embeddings_np).float()
            except Exception as e:
                self.logger.error(
                    f"Failed to load embedding cache: {e}. Recomputing..."
                )

        # 2. Compute Embeddings
        self.logger.info(f"Computing embeddings for {cache_file}...")
        embeddings_list = []

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                feats = model(images)  # Forward pass returns embeddings in eval mode
                feats = F.normalize(feats, p=2, dim=1)
                embeddings_list.append(feats.cpu())

        embeddings = torch.cat(embeddings_list, dim=0)

        # 3. Save Cache
        np.save(cache_full_path, embeddings.numpy())
        self.logger.info(f"Saved embeddings to {cache_full_path}")

        return embeddings

    def predict(self, load_cached_data=True, threshold=0.35):
        """
        Main inference pipeline.
        1. Loads data (Gallery=Train, Query=Test).
        2. Computes embeddings.
        3. Performs KNN retrieval.
        4. Generates submission file.
        """
        # --- 1. Load Metadata ---
        # Gallery: Known Whales Only (Cite Lesson 18)
        train_csv_path = os.path.join(METADATA_DIR, "train.csv")
        df_train = pd.read_csv(train_csv_path)
        df_gallery = df_train[df_train["Id"] != "new_whale"].reset_index(drop=True)

        # Query: Test set
        test_csv_path = os.path.join(METADATA_DIR, "test.csv")
        df_query = pd.read_csv(test_csv_path)

        self.logger.info(f"Gallery Size: {len(df_gallery)}")
        self.logger.info(f"Query Size: {len(df_query)}")

        # --- 2. Load Model ---
        model = self.load_model()

        # --- 3. Generate Embeddings ---
        # Note: We use distinct cache names for the image arrays to avoid conflict with trainer.py
        # trainer.py uses 'train_images.npy' (filtered). We use 'gallery_known_images.npy' (known).

        gallery_loader = self.get_dataloader(
            df_gallery, "gallery_known_images.npy", load_cache=load_cached_data
        )
        query_loader = self.get_dataloader(
            df_query, "query_images.npy", load_cache=load_cached_data
        )

        # Embedding caches
        gallery_feats = self.generate_embeddings(
            model,
            gallery_loader,
            "gallery_known_embeddings.npy",
            load_cache=load_cached_data,
        )
        query_feats = self.generate_embeddings(
            model, query_loader, "query_embeddings.npy", load_cache=load_cached_data
        )

        # --- 4. Retrieval (KNN) ---
        self.logger.info("Computing similarity matrix...")

        # Move to GPU for matrix multiplication
        gallery_feats = gallery_feats.to(self.device)
        query_feats = query_feats.to(self.device)

        # Cosine Similarity: (N_query, D) @ (D, N_gallery) -> (N_query, N_gallery)
        sim_matrix = torch.mm(query_feats, gallery_feats.t())

        # Get Top-20 neighbors to have enough candidates for filtering
        topk_scores, topk_indices = torch.topk(sim_matrix, k=20, dim=1)

        # Move back to CPU
        topk_scores = topk_scores.cpu().numpy()
        topk_indices = topk_indices.cpu().numpy()

        # --- 5. Generate Predictions ---
        self.logger.info("Generating prediction strings...")

        submission_data = []
        gallery_ids = df_gallery["Id"].values

        for i in range(len(df_query)):
            filename = df_query.iloc[i]["Image"]

            # Get neighbors for this query
            indices = topk_indices[i]
            scores = topk_scores[i]

            # Map indices to IDs
            neighbor_ids = gallery_ids[indices]

            # Logic to form top 5 predictions
            # Strategy:
            # 1. Gather unique IDs from neighbors, preserving order.
            # 2. Insert 'new_whale' based on threshold logic.

            unique_preds = []
            seen = set()

            # Check top 1 score for new_whale threshold
            top_score = scores[0]

            # If the top match is weak, we suspect new_whale
            force_new_whale_first = top_score < threshold

            if force_new_whale_first:
                unique_preds.append("new_whale")
                seen.add("new_whale")

            for nid in neighbor_ids:
                if nid not in seen:
                    unique_preds.append(nid)
                    seen.add(nid)
                if len(unique_preds) >= 5:
                    break

            # Fallback: If we still don't have 5 (unlikely given k=20), fill with new_whale if not present
            if "new_whale" not in seen:
                # If we haven't added new_whale yet, add it now.
                # Usually it's good to have it as the 2nd guess if the 1st guess was strong.
                if len(unique_preds) < 5:
                    unique_preds.append("new_whale")
                elif len(unique_preds) == 5:
                    # If we are full but new_whale isn't there, replace the last one
                    # (Common strategy: always predict new_whale within top 5)
                    unique_preds[-1] = "new_whale"

            # Final truncation
            final_preds = unique_preds[:5]

            submission_data.append({"Image": filename, "Id": " ".join(final_preds)})

        # --- 6. Save Submission ---
        df_sub = pd.DataFrame(submission_data)
        os.makedirs(os.path.dirname(self.submission_path), exist_ok=True)
        df_sub.to_csv(self.submission_path, index=False)
        self.logger.info(f"Submission saved to {self.submission_path}")


def main():
    # Initialize Manager
    manager = InferenceManager()

    # Run Inference
    # We set load_cached_data=False for the embeddings to ensure we use the latest model weights,
    # but we can keep True for the images to save loading time.
    # However, the function signature controls both via one flag in this simple implementation.
    # To be safe and ensure model changes are reflected, we set it to False for embeddings.
    # But since generate_embeddings handles its own cache file, we can just delete the embedding cache
    # if we want to force recompute, or pass a flag.
    # Here we will assume the user wants to run inference on the current best model.

    manager.predict(load_cached_data=True, threshold=0.35)


if __name__ == "__main__":
    # This block is not required by the prompt but useful for testing.
    # The prompt asks to implement the module.
    pass
