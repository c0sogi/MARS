import os
import torch
import numpy as np
import pandas as pd
from library import config, utils
from library.model import WhaleEfficientNet
from library.data_loader import get_dataloaders


def generate_embeddings(
    model,
    loader,
    device,
    cache_path_emb,
    cache_path_meta,
    is_test=False,
    load_cached_data=True,
):
    """
    Generates or loads embeddings for a given dataloader.

    Args:
        model: The trained neural network.
        loader: DataLoader for the dataset.
        device: 'cuda' or 'cpu'.
        cache_path_emb: Path to save/load the embeddings numpy array.
        cache_path_meta: Path to save/load the metadata (labels or filenames).
        is_test: Boolean, True if processing test set (returns filenames), False for gallery (returns labels).
        load_cached_data: Boolean, whether to attempt loading from cache.

    Returns:
        embeddings: Numpy array of shape [N, Embedding_Size]
        meta: Numpy array of strings (Labels or Filenames)
    """
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # 1. Try Loading from Cache
    if (
        load_cached_data
        and os.path.exists(cache_path_emb)
        and os.path.exists(cache_path_meta)
    ):
        print(f"Loading cached features from {cache_path_emb}...")
        embeddings = np.load(cache_path_emb)
        meta = np.load(cache_path_meta)
        return embeddings, meta

    # 2. Generate from Scratch
    print(f"Generating features (is_test={is_test})...")
    model.eval()
    embeddings_list = []
    meta_list = []

    with torch.no_grad():
        for batch in loader:
            if is_test:
                images, filenames = batch
                meta_data = filenames
            else:
                images, _, label_strs = batch
                meta_data = label_strs

            images = images.to(device)

            # Forward pass with labels=None returns L2-normalized embeddings
            emb = model(images, labels=None)

            embeddings_list.append(emb.cpu().numpy())
            meta_list.extend(meta_data)

    embeddings = np.concatenate(embeddings_list, axis=0)
    meta = np.array(meta_list)

    # 3. Save to Cache
    np.save(cache_path_emb, embeddings)
    np.save(cache_path_meta, meta)
    print(f"Features saved to {config.WORKING_DIR}")

    return embeddings, meta


def predict(model, gallery_loader, test_loader, load_cached_data=True):
    """
    Performs the inference pipeline:
    1. Extract features for Gallery and Test sets.
    2. Compute Cosine Similarity Matrix.
    3. Apply Open-Set Recognition Threshold.
    4. Generate Submission CSV.
    """
    device = config.DEVICE

    # ---------------------------------------------------------
    # 1. Feature Extraction
    # ---------------------------------------------------------
    # Gallery (Known Whales)
    gallery_emb, gallery_ids = generate_embeddings(
        model,
        gallery_loader,
        device,
        config.TRAIN_EMBEDDINGS_CACHE,
        config.TRAIN_LABELS_CACHE,
        is_test=False,
        load_cached_data=load_cached_data,
    )

    # Test (Query Images)
    test_emb, test_names = generate_embeddings(
        model,
        test_loader,
        device,
        config.TEST_EMBEDDINGS_CACHE,
        config.TEST_NAMES_CACHE,
        is_test=True,
        load_cached_data=load_cached_data,
    )

    # ---------------------------------------------------------
    # 2. Similarity Computation
    # ---------------------------------------------------------
    print("Computing cosine similarity matrix...")
    # Convert to tensors and move to GPU for fast matrix multiplication
    gal_tensor = torch.from_numpy(gallery_emb).to(device)
    test_tensor = torch.from_numpy(test_emb).to(device)

    # Cosine Similarity = Dot Product (since vectors are L2 normalized)
    # Shape: [Num_Test, Num_Gallery]
    sim_matrix = torch.matmul(test_tensor, gal_tensor.t())

    # ---------------------------------------------------------
    # 3. Prediction & Ranking
    # ---------------------------------------------------------
    print("Generating predictions...")
    submission_data = []

    # Retrieve Top-K candidates
    # We fetch 50 to ensure we can filter out duplicate IDs (same whale, different images)
    # and still have enough unique candidates to fill the top 5 slots.
    top_vals, top_inds = torch.topk(sim_matrix, k=50, dim=1)

    top_vals = top_vals.cpu().numpy()
    top_inds = top_inds.cpu().numpy()

    threshold = config.CONFIDENCE_THRESHOLD

    for i in range(len(test_names)):
        filename = test_names[i]
        scores = top_vals[i]
        indices = top_inds[i]

        # Map indices back to Whale IDs
        candidate_ids = gallery_ids[indices]

        # Filter duplicates while preserving order (highest score first)
        unique_ids = []
        seen = set()
        for cid in candidate_ids:
            if cid not in seen:
                unique_ids.append(cid)
                seen.add(cid)
            if len(unique_ids) >= 5:
                break

        # Logic for Open-Set Recognition
        best_score = scores[0]

        if best_score < threshold:
            # Case: Low Confidence
            # The nearest neighbor is too far. We predict 'new_whale' as the primary label.
            # We fill the remaining slots with the nearest known neighbors.
            final_preds = ["new_whale"] + unique_ids[:4]
        else:
            # Case: High Confidence
            # The nearest neighbor is close enough. We predict it as the primary label.
            # We insert 'new_whale' at the second position to hedge our bet (common strategy for MAP@5),
            # followed by the next best matches.
            final_preds = [unique_ids[0], "new_whale"] + unique_ids[1:4]

        submission_data.append({"Image": filename, "Id": " ".join(final_preds)})

    # ---------------------------------------------------------
    # 4. Save Submission
    # ---------------------------------------------------------
    df_sub = pd.DataFrame(submission_data)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")
    print("Sample predictions:")
    print(df_sub.head())


def run_inference(load_cached_data=True):
    """
    Main entry point for the inference module.
    """
    utils.set_seed(config.SEED)

    # Load Data Loaders
    # We only need gallery and test loaders for inference
    _, _, gallery_loader, test_loader, _, num_classes = get_dataloaders()

    # Initialize Model
    print(f"Initializing model for {num_classes} classes...")
    model = WhaleEfficientNet(num_classes=num_classes)

    # Load Weights
    if os.path.exists(config.MODEL_PATH):
        print(f"Loading model weights from {config.MODEL_PATH}")
        state_dict = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
        model.load_state_dict(state_dict)
    else:
        print(
            "Warning: Model checkpoint not found. Using random initialization (expect poor results)."
        )

    model.to(config.DEVICE)

    # Execute Prediction Pipeline
    predict(model, gallery_loader, test_loader, load_cached_data=load_cached_data)
