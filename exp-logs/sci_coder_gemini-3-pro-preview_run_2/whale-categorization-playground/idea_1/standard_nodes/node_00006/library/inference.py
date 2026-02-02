import os
import torch
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from library.config import Config
from library.utils import load_metadata, save_submission
from library.dataset import WhaleInferenceDataset
from library.engine import extract_embeddings
from library.model import EmbeddingNet, SiameseNet


def get_embeddings(model, df, device, cache_prefix, load_cached_data=True):
    """
    Generates or loads embeddings for a given dataframe using the provided model.

    Args:
        model (nn.Module): The neural network model.
        df (pd.DataFrame): Metadata dataframe.
        device (torch.device): Computation device.
        cache_prefix (str): Prefix for cache files (e.g., 'train_ref').
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        tuple: (embeddings, ids, images)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    emb_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_embeddings.npy")
    ids_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_ids.npy")
    imgs_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_imgs.npy")

    # Check if all cache files exist
    if (
        load_cached_data
        and os.path.exists(emb_path)
        and os.path.exists(ids_path)
        and os.path.exists(imgs_path)
    ):
        print(f"Loading cached embeddings from {emb_path}")
        try:
            embeddings = np.load(emb_path)
            ids = np.load(ids_path, allow_pickle=True)
            images = np.load(imgs_path, allow_pickle=True)
            return embeddings, ids, images
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    print(f"Computing embeddings for {cache_prefix}...")

    # Initialize Dataset
    # Use a specific cache name for images to avoid conflicts between train/test sets
    image_cache_name = f"{cache_prefix}_images_cache.npy"
    dataset = WhaleInferenceDataset(
        df,
        load_cached_data=load_cached_data,
        transform=None,  # Uses default val transform
        cache_name=image_cache_name,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Extract embeddings using engine function
    results = extract_embeddings(model, dataloader, device)

    embeddings = results["embeddings"]
    ids = np.array(results["ids"])
    images = np.array(results["images"])

    # Save to cache
    print(f"Saving embeddings to {emb_path}")
    np.save(emb_path, embeddings)
    np.save(ids_path, ids)
    np.save(imgs_path, images)

    return embeddings, ids, images


def predict_knn(
    test_embeddings,
    train_embeddings,
    train_labels,
    test_filenames,
    threshold=Config.NEW_WHALE_THRESHOLD,
    k=50,
):
    """
    Performs K-Nearest Neighbors search and applies thresholding logic for predictions.

    Args:
        test_embeddings (np.ndarray): Query embeddings.
        train_embeddings (np.ndarray): Reference embeddings.
        train_labels (np.ndarray): Reference labels.
        test_filenames (np.ndarray): Filenames for test images.
        threshold (float): Distance threshold for 'new_whale'.
        k (int): Number of neighbors to retrieve.

    Returns:
        list: List of lists containing top 5 predicted IDs.
    """
    print(f"Fitting NearestNeighbors with k={k}...")
    # Euclidean distance on L2-normalized embeddings is equivalent to ranking by Cosine Similarity
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    knn.fit(train_embeddings)

    print("Finding neighbors for test set...")
    distances, indices = knn.kneighbors(test_embeddings)

    final_predictions = []

    for i in range(len(test_filenames)):
        dists = distances[i]
        inds = indices[i]

        # Get the labels of the neighbors
        neighbor_labels = train_labels[inds]

        # Identify unique labels while preserving order (nearest first)
        unique_labels = []
        seen = set()
        for label in neighbor_labels:
            if label not in seen:
                unique_labels.append(label)
                seen.add(label)

        preds = []

        # Threshold Logic:
        # If the nearest neighbor is further away than the threshold,
        # assume it's a new whale not in the database.
        # Note: We check dists[0] (nearest neighbor distance)
        if dists[0] > threshold:
            preds.append("new_whale")

        # Fill the rest with the nearest unique neighbors
        for label in unique_labels:
            if label not in preds:
                preds.append(label)
                if len(preds) >= 5:
                    break

        # Fallback: If we still have fewer than 5 predictions,
        # append 'new_whale' if it's not already there.
        if len(preds) < 5 and "new_whale" not in preds:
            preds.append("new_whale")

        # Ensure we return exactly top 5 (or fewer if not enough candidates, but usually we have enough)
        final_predictions.append(preds[:5])

    return final_predictions


def run_inference(load_cached_data=True, threshold=Config.NEW_WHALE_THRESHOLD):
    """
    Main execution function for inference.

    Args:
        load_cached_data (bool): Whether to use cached data.
        threshold (float): Threshold for new_whale classification.
    """
    device = Config.DEVICE
    print(f"Running inference on {device}...")

    # 1. Load Metadata
    df_train = load_metadata("train")
    df_test = load_metadata("test")

    # 2. Initialize Model
    print("Initializing model...")
    # We need to reconstruct the model architecture to load weights
    embedding_net = EmbeddingNet()
    model = SiameseNet(embedding_net)
    model.to(device)

    # Load weights
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Loading model weights from {Config.MODEL_SAVE_PATH}")
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model weights not found at {Config.MODEL_SAVE_PATH}. Using initialized weights."
        )

    model.eval()

    # 3. Generate Embeddings
    # Reference Database (Train)
    # Cite solution_lesson_node_00005: Exclude new_whale from reference database
    df_train_ref = df_train[df_train["Id"] != "new_whale"]
    train_emb, train_ids, _ = get_embeddings(
        model, df_train_ref, device, "train_ref", load_cached_data=load_cached_data
    )

    # Query Set (Test)
    test_emb, _, test_imgs = get_embeddings(
        model, df_test, device, "test_query", load_cached_data=load_cached_data
    )

    # 4. Predict
    print(f"Predicting with threshold={threshold}...")
    predictions = predict_knn(
        test_emb,
        train_emb,
        train_ids,
        test_imgs,
        threshold=threshold,
        k=50,  # Retrieve enough neighbors to filter duplicates
    )

    # 5. Save Submission
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    save_submission(test_imgs, predictions, Config.SUBMISSION_PATH)

    print("Inference completed successfully.")
