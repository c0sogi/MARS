import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from library.config import Config
from library.dataset import get_dataloaders
from library.engine import get_model_embeddings


def extract_features(load_cached_data=True):
    """
    Extracts embeddings for both gallery (train) and query (test) sets
    using all models defined in Config.

    Args:
        load_cached_data (bool): Whether to load embeddings from cache if available.

    Returns:
        tuple: (gallery_embs_list, query_embs_list, gallery_labels, query_ids, idx_to_class)
    """
    device = Config.DEVICE
    # Load dataloaders to get access to data and class mappings
    # We use train_loader for Gallery and test_loader for Query
    train_loader, _, test_loader, _, idx_to_class = get_dataloaders(
        load_cached_data=load_cached_data
    )

    gallery_embs_list = []
    query_embs_list = []
    gallery_labels = None
    query_ids = None

    for model_name in Config.MODEL_NAMES:
        # Extract Gallery Embeddings (Train Set)
        # get_model_embeddings handles caching internally
        g_emb, g_lbl = get_model_embeddings(
            model_name, "train", train_loader, device, load_cached_data
        )
        gallery_embs_list.append(g_emb)

        # We assume labels are consistent across models for the same dataset split
        if gallery_labels is None:
            gallery_labels = g_lbl

        # Extract Query Embeddings (Test Set)
        q_emb, q_id = get_model_embeddings(
            model_name, "test", test_loader, device, load_cached_data
        )
        query_embs_list.append(q_emb)

        # We assume image IDs are consistent
        if query_ids is None:
            query_ids = q_id

    return gallery_embs_list, query_embs_list, gallery_labels, query_ids, idx_to_class


def concat_features(gallery_embs_list, query_embs_list):
    """
    Normalizes and concatenates embeddings from multiple models.

    Args:
        gallery_embs_list (list): List of numpy arrays containing gallery embeddings for each model.
        query_embs_list (list): List of numpy arrays containing query embeddings for each model.

    Returns:
        tuple: (combined_gallery_embeddings, combined_query_embeddings)
    """
    # Normalize individual model embeddings before concatenation
    gallery_norm_list = [normalize(emb, norm="l2", axis=1) for emb in gallery_embs_list]
    query_norm_list = [normalize(emb, norm="l2", axis=1) for emb in query_embs_list]

    # Concatenate features from all models
    gallery_concat = np.concatenate(gallery_norm_list, axis=1)
    query_concat = np.concatenate(query_norm_list, axis=1)

    # Normalize the concatenated vectors
    gallery_final = normalize(gallery_concat, norm="l2", axis=1)
    query_final = normalize(query_concat, norm="l2", axis=1)

    return gallery_final, query_final


def apply_dba(gallery_embeddings, k=Config.DBA_K):
    """
    Performs Database Augmentation (DBA) on gallery embeddings.
    Replaces each embedding with a weighted average of itself and its k-nearest neighbors.

    Args:
        gallery_embeddings (np.ndarray): The gallery embeddings matrix.
        k (int): Number of neighbors to use.

    Returns:
        np.ndarray: Refined gallery embeddings.
    """
    print(f"Applying Database Augmentation (DBA) with k={k}...")
    knn = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=-1)
    knn.fit(gallery_embeddings)
    _, indices = knn.kneighbors(gallery_embeddings)

    # Create a new array to store augmented embeddings
    augmented_gallery = np.zeros_like(gallery_embeddings)

    for i in range(len(gallery_embeddings)):
        neighbor_indices = indices[i]
        # Average the embeddings of neighbors (including self)
        # Since we used fit(gallery_embeddings), the point itself is included in neighbors at dist 0
        neighbors = gallery_embeddings[neighbor_indices]
        augmented_gallery[i] = np.mean(neighbors, axis=0)

    # Re-normalize after averaging
    return normalize(augmented_gallery, norm="l2", axis=1)


def apply_qe(query_embeddings, gallery_embeddings, k=Config.QE_K):
    """
    Performs Query Expansion (QE).
    Refines query embeddings by averaging with top-k retrieved gallery embeddings.

    Args:
        query_embeddings (np.ndarray): The query embeddings matrix.
        gallery_embeddings (np.ndarray): The gallery embeddings matrix.
        k (int): Number of neighbors to use.

    Returns:
        np.ndarray: Refined query embeddings.
    """
    print(f"Applying Query Expansion (QE) with k={k}...")
    knn = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=-1)
    knn.fit(gallery_embeddings)
    _, indices = knn.kneighbors(query_embeddings)

    augmented_query = np.zeros_like(query_embeddings)

    for i in range(len(query_embeddings)):
        neighbor_indices = indices[i]
        neighbors = gallery_embeddings[neighbor_indices]
        # Combine original query with neighbors
        # Strategy: Mean of (Query U Neighbors)
        vectors = np.vstack([query_embeddings[i], neighbors])
        augmented_query[i] = np.mean(vectors, axis=0)

    # Re-normalize
    return normalize(augmented_query, norm="l2", axis=1)


def predict(
    query_embeddings,
    gallery_embeddings,
    gallery_labels,
    query_ids,
    idx_to_class,
    top_k=Config.TOP_K,
):
    """
    Performs final retrieval and generates predictions.

    Args:
        query_embeddings (np.ndarray): Refined query embeddings.
        gallery_embeddings (np.ndarray): Refined gallery embeddings.
        gallery_labels (np.ndarray): Class indices for gallery images.
        query_ids (np.ndarray): Image IDs for query images.
        idx_to_class (dict): Mapping from class index to hotel_id.
        top_k (int): Number of predictions to return per query.

    Returns:
        pd.DataFrame: Submission dataframe.
    """
    print(f"Predicting top {top_k} matches...")
    knn = NearestNeighbors(n_neighbors=top_k, metric="cosine", n_jobs=-1)
    knn.fit(gallery_embeddings)
    _, indices = knn.kneighbors(query_embeddings)

    predictions = []
    for i in range(len(query_ids)):
        img_id = query_ids[i]
        neighbor_idxs = indices[i]

        # Map gallery indices to class indices
        pred_class_idxs = gallery_labels[neighbor_idxs]

        # Map class indices to hotel IDs (strings)
        pred_hotel_ids = [str(idx_to_class[idx]) for idx in pred_class_idxs]

        predictions.append({"image": img_id, "hotel_id": " ".join(pred_hotel_ids)})

    return pd.DataFrame(predictions)


def run_inference(load_cached_data=True):
    """
    Orchestrates the full inference pipeline:
    Extraction -> Concatenation -> DBA -> QE -> Prediction -> Save.
    """
    # 1. Extract Features
    gallery_embs_list, query_embs_list, gallery_labels, query_ids, idx_to_class = (
        extract_features(load_cached_data)
    )

    # 2. Concatenate Features
    gallery_embeddings, query_embeddings = concat_features(
        gallery_embs_list, query_embs_list
    )

    print(f"Combined Gallery shape: {gallery_embeddings.shape}")
    print(f"Combined Query shape: {query_embeddings.shape}")

    # 3. Database Augmentation (DBA)
    if Config.DBA_K > 0:
        gallery_embeddings = apply_dba(gallery_embeddings, k=Config.DBA_K)

    # 4. Query Expansion (QE)
    if Config.QE_K > 0:
        query_embeddings = apply_qe(query_embeddings, gallery_embeddings, k=Config.QE_K)

    # 5. Predict
    df_submission = predict(
        query_embeddings,
        gallery_embeddings,
        gallery_labels,
        query_ids,
        idx_to_class,
        top_k=Config.TOP_K,
    )

    # 6. Save Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
