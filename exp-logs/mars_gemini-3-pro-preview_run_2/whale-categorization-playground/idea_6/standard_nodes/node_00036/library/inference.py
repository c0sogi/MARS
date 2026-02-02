import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import WhaleDataset, create_id_map
from library.model import WhaleModel
from library.utils import seed_everything


def get_embeddings(dataloader, model, device):
    """
    Extracts embeddings from a dataloader using the provided model.
    Returns normalized embeddings and the associated labels (or filenames).
    """
    model.eval()
    all_embeddings = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch depending on mode (train/val returns labels, test returns filenames)
            if len(batch) == 2:
                images, targets = batch
            else:
                raise ValueError("Unexpected batch structure")

            images = images.to(device)

            # Forward pass
            emb = model(images)

            # Normalize immediately (L2)
            emb = F.normalize(emb, p=2, dim=1)

            all_embeddings.append(emb.cpu().numpy())

            # Handle targets (tensor labels or list of strings)
            if isinstance(targets, torch.Tensor):
                all_targets.extend(targets.numpy())
            else:
                all_targets.extend(targets)

    # Concatenate embeddings
    if len(all_embeddings) > 0:
        all_embeddings = np.concatenate(all_embeddings, axis=0)
    else:
        all_embeddings = np.array([])

    return all_embeddings, np.array(all_targets)


def load_or_compute_embeddings(
    subset_name, dataset, model, device, load_cached_data=True
):
    """
    Handles caching logic for embeddings.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    emb_path = os.path.join(cache_dir, f"{subset_name}_embeddings.npy")
    target_path = os.path.join(cache_dir, f"{subset_name}_targets.npy")

    if load_cached_data and os.path.exists(emb_path) and os.path.exists(target_path):
        print(f"Loading cached embeddings for {subset_name}...")
        embeddings = np.load(emb_path)
        targets = np.load(target_path)
        return embeddings, targets

    print(f"Computing embeddings for {subset_name}...")
    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    embeddings, targets = get_embeddings(dataloader, model, device)

    # Save to cache
    np.save(emb_path, embeddings)
    np.save(target_path, targets)

    return embeddings, targets


def predict(test_embeddings, gallery_embeddings, gallery_labels, id_to_name_map):
    """
    Computes cosine similarity and generates predictions with open-set logic.
    """
    print("Computing similarity matrix...")

    # Convert to torch tensors for GPU acceleration if available, otherwise numpy
    device = torch.device(Config.DEVICE)

    # Move to GPU for faster matrix multiplication
    # Chunking might be necessary for very large datasets, but 2600x6000 fits in A100 memory easily.
    test_tensor = torch.from_numpy(test_embeddings).to(device)
    gallery_tensor = torch.from_numpy(gallery_embeddings).to(device)

    # Cosine Similarity: (N_test, D) @ (N_gallery, D).T -> (N_test, N_gallery)
    # Embeddings are already normalized
    sim_matrix = torch.matmul(test_tensor, gallery_tensor.T)

    predictions = []

    print("Generating predictions...")
    # Iterate over each test image
    for i in range(sim_matrix.size(0)):
        scores = sim_matrix[i]

        # Get top K candidates from the gallery
        # We fetch slightly more than 5 to handle duplicate IDs if we were aggregating,
        # but here we just take the nearest instances.
        # Since gallery might have multiple images of same whale, we want unique IDs.

        # Get top 20 nearest neighbors to ensure we find enough unique IDs
        top_scores, top_indices = torch.topk(scores, k=20)

        top_scores = top_scores.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        best_score = top_scores[0]

        # Map indices to IDs
        found_ids = []
        for idx in top_indices:
            label_idx = gallery_labels[idx]
            whale_id = id_to_name_map[label_idx]
            if whale_id not in found_ids:
                found_ids.append(whale_id)
                if len(found_ids) >= 5:
                    break

        # Apply Open-Set Logic
        # Strategy:
        # If best_score > threshold: [BestID, new_whale, 2ndID, 3rdID, 4thID]
        # If best_score <= threshold: [new_whale, BestID, 2ndID, 3rdID, 4thID]

        final_preds = []

        if best_score > Config.CONFIDENCE_THRESHOLD:
            # Confident match
            final_preds.append(found_ids[0])
            final_preds.append("new_whale")
            final_preds.extend(found_ids[1:])
        else:
            # Low confidence
            final_preds.append("new_whale")
            final_preds.extend(found_ids)

        # Truncate to top 5
        final_preds = final_preds[:5]

        predictions.append(" ".join(final_preds))

    return predictions


def run_inference(load_cached_data=True):
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Inference Device: {device}")

    # 1. Setup Maps
    # We need the exact ID map used during training
    id_map = create_id_map(Config.TRAIN_CSV)
    id_to_name = {v: k for k, v in id_map.items()}
    num_classes = len(id_map)
    print(f"Number of known classes: {num_classes}")

    # 2. Load Model
    print("Loading model...")
    model = WhaleModel(embedding_size=Config.EMBEDDING_SIZE, pretrained=False)

    # Load weights
    checkpoint_path = Config.MODEL_PATH
    if not os.path.exists(checkpoint_path):
        # Fallback to last checkpoint if best doesn't exist (e.g. early stopping weirdness)
        checkpoint_path = os.path.join(Config.WORKING_DIR, "checkpoint_last.pth")

    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(
            "Warning: No checkpoint found! Using random weights (for debugging only)."
        )

    if Config.USE_GRADIENT_CHECKPOINTING:
        model.enable_gradient_checkpointing()

    model.to(device)
    model.eval()

    # 3. Prepare Gallery (Train Data)
    # We use the 'val' mode for deterministic transforms and 'train' subset
    # Filter new_whale is True because we only want known whales in the gallery
    train_dataset = WhaleDataset(
        csv_path=Config.TRAIN_CSV,
        subset_name="train_gallery",
        image_size=Config.IMG_SIZE_FINAL,
        id_map=id_map,
        mode="val",
        filter_new_whale=True,
        load_cached_data=load_cached_data,
    )

    gallery_embeddings, gallery_labels = load_or_compute_embeddings(
        "gallery", train_dataset, model, device, load_cached_data
    )
    print(f"Gallery Embeddings: {gallery_embeddings.shape}")

    # 4. Prepare Queries (Test Data)
    test_dataset = WhaleDataset(
        csv_path=Config.TEST_CSV,
        subset_name="test",
        image_size=Config.IMG_SIZE_FINAL,
        id_map=None,
        mode="test",
        filter_new_whale=False,
        load_cached_data=load_cached_data,
    )

    test_embeddings, test_filenames = load_or_compute_embeddings(
        "test", test_dataset, model, device, load_cached_data
    )
    print(f"Test Embeddings: {test_embeddings.shape}")

    # 5. Generate Predictions
    pred_strings = predict(
        test_embeddings, gallery_embeddings, gallery_labels, id_to_name
    )

    # 6. Save Submission
    submission_df = pd.DataFrame({"Image": test_filenames, "Id": pred_strings})

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    save_path = Config.SUBMISSION_PATH
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")

    # Print sample
    print("\nSample Predictions:")
    print(submission_df.head())
