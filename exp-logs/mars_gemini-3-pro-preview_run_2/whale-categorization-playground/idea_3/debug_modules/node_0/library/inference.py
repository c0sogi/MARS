import os
import torch
import numpy as np
import pandas as pd
import gc
from library.config import Config
from library.dataset import get_loaders
from library.model import WhaleModel
from library.utils import k_reciprocal_re_ranking


def extract_features(model, dataloader, device):
    """
    Extracts embeddings and labels from a dataloader using the provided model.

    Args:
        model (nn.Module): The loaded model.
        dataloader (DataLoader): DataLoader for inference.
        device (torch.device): Device to run inference on.

    Returns:
        embeddings (np.ndarray): Normalized embeddings (N, D).
        labels (np.ndarray): Labels (N,). None if dataloader has no labels.
        image_names (list): List of image filenames (if available/needed,
                            though loaders here return batches).
                            We rely on the order preservation of DataLoader(shuffle=False).
    """
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            emb = model(images)
            embeddings.append(emb.cpu().numpy())

            if "label" in batch:
                labels.append(batch["label"].numpy())

    embeddings = np.concatenate(embeddings)

    # L2 Normalize embeddings (Standard practice before distance calculations)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)

    if labels:
        labels = np.concatenate(labels)
        return embeddings, labels
    else:
        return embeddings, None


def generate_submission(load_cached_data=True):
    """
    Generates the submission file using the trained model and k-reciprocal re-ranking.

    Args:
        load_cached_data (bool): Whether to load cached *images*.
                                 Embeddings are always computed fresh.
    """
    print("Initializing Inference Pipeline...")

    # 1. Setup
    device = Config.device

    # 2. Load Data
    # We need gallery (all train) and test loaders.
    # We discard train/val loaders here.
    print("Loading DataLoaders...")
    _, gallery_loader, _, test_loader, id2label = get_loaders(
        debug=Config.debug, load_cached_data=load_cached_data
    )

    # Create Int -> String mapping
    # id2label maps 'new_whale' -> -1.
    int2str = {v: k for k, v in id2label.items()}

    # 3. Load Model
    print(f"Loading model from {Config.model_path}...")
    if not os.path.exists(Config.model_path):
        raise FileNotFoundError(
            f"Model file not found at {Config.model_path}. Please train the model first."
        )

    model = WhaleModel(
        pretrained=False
    )  # Pretrained weights not needed, we load state_dict
    state_dict = torch.load(Config.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # 4. Extract Embeddings
    # Explicitly NOT loading embeddings from cache to ensure freshness
    print("Extracting Gallery Embeddings...")
    gallery_feats, gallery_labels = extract_features(model, gallery_loader, device)

    print("Extracting Test Embeddings...")
    test_feats, _ = extract_features(model, test_loader, device)

    print(f"Gallery Shape: {gallery_feats.shape}")
    print(f"Test Shape: {test_feats.shape}")

    # 5. Compute Distances with Re-ranking
    print("Computing k-Reciprocal Re-ranking distances...")
    # Using default hyperparameters for re-ranking as defined in utils or tuned
    # k1=20, k2=6, lambda_value=0.3 are common defaults
    dist_matrix = k_reciprocal_re_ranking(
        test_feats,
        gallery_feats,
        k1=Config.knn_k,  # Use config value or default 20
        k2=6,
        lambda_value=0.3,
    )

    # 6. Generate Predictions
    print("Generating Top-5 Predictions...")

    # Get test image names from metadata to ensure alignment
    df_test = pd.read_csv(Config.meta_test_path)
    if Config.debug:
        df_test = df_test.iloc[: Config.debug_sample_size]
    test_image_names = df_test["Image"].values

    submission_data = []

    # Iterate over each test sample
    num_test = dist_matrix.shape[0]

    # We can process in chunks if memory is tight, but for 2600 samples it's fine.
    # We need to sort distances.
    # argsort on the whole matrix might be slow if gallery is huge, but here gallery is ~7k.
    # 2600 x 7000 is small enough for full argsort or argpartition.

    # Using numpy argsort
    # We want smallest distances
    sorted_indices = np.argsort(dist_matrix, axis=1)

    for i in range(num_test):
        # Get indices of neighbors sorted by distance
        indices = sorted_indices[i]

        # Map to Whale IDs
        neighbor_ids = gallery_labels[indices]

        # Collect top 5 unique IDs
        top_5_ids = []
        seen_ids = set()

        for nid in neighbor_ids:
            if nid not in seen_ids:
                label_str = int2str[nid]
                top_5_ids.append(label_str)
                seen_ids.add(nid)

            if len(top_5_ids) == 5:
                break

        # Fill with 'new_whale' if we somehow didn't find 5 (unlikely given gallery size)
        while len(top_5_ids) < 5:
            if "new_whale" not in top_5_ids:
                top_5_ids.append("new_whale")
            else:
                # Fallback if new_whale is already there (very unlikely edge case)
                # Just append the most frequent one or break
                break

        prediction_str = " ".join(top_5_ids)
        submission_data.append({"Image": test_image_names[i], "Id": prediction_str})

    # 7. Save Submission
    submission_df = pd.DataFrame(submission_data)

    # Ensure output directory exists
    os.makedirs(Config.submission_dir, exist_ok=True)

    print(f"Saving submission to {Config.submission_path}...")
    submission_df.to_csv(Config.submission_path, index=False)

    print("Submission generation complete.")

    # Cleanup
    del model, gallery_feats, test_feats, dist_matrix
    gc.collect()
    torch.cuda.empty_cache()
