import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import ChemicalDataset
from library.model import StoichiometryEncoder


def extract_embeddings(model, loader, device):
    """
    Runs inference on a dataloader to extract normalized embeddings.

    Args:
        model (nn.Module): The trained encoder model.
        loader (DataLoader): DataLoader providing images.
        device (torch.device): Device to run inference on.

    Returns:
        torch.Tensor: Tensor of shape (N, embedding_dim) containing normalized embeddings.
    """
    model.eval()
    embeddings = []

    with torch.no_grad():
        for batch in loader:
            # Loader returns (images, targets) or (images, ids) depending on mode.
            # The image tensor is always at index 0.
            images = batch[0].to(device)

            # Forward pass returns (embedding, atom_preds). We only need the embedding.
            emb, _ = model(images)

            # Normalize embeddings for Cosine Similarity (L2 norm)
            emb = F.normalize(emb, p=2, dim=1)

            embeddings.append(emb.cpu())

    # Concatenate all batches into a single tensor
    if len(embeddings) > 0:
        return torch.cat(embeddings, dim=0)
    else:
        return torch.tensor([])


def build_index(model, train_loader, device, load_cached_data=True):
    """
    Generates or loads the training set embeddings (the retrieval index).
    Implements strict caching logic using .npy format.

    Args:
        model (nn.Module): The trained encoder model.
        train_loader (DataLoader): DataLoader for the training set.
        device (torch.device): Device to run inference on.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        torch.Tensor: Tensor of training embeddings.
    """
    cache_path = Config.TRAIN_EMBEDDINGS_PATH

    # Ensure directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached training embeddings from {cache_path}...")
        try:
            embeddings_np = np.load(cache_path)
            return torch.from_numpy(embeddings_np)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print("Computing training embeddings (Index construction)...")
    embeddings = extract_embeddings(model, train_loader, device)

    # 3. Save to cache
    print(f"Saving training embeddings to {cache_path}...")
    try:
        np.save(cache_path, embeddings.numpy())
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

    return embeddings


def query_index(test_embeddings, train_index, device, batch_size=1024):
    """
    Finds the nearest neighbor in train_index for each item in test_embeddings.
    Uses Cosine Similarity.

    Args:
        test_embeddings (torch.Tensor): Query vectors (N_test, dim).
        train_index (torch.Tensor): Index vectors (N_train, dim).
        device (torch.device): Device for computation.
        batch_size (int): Batch size for matrix multiplication to avoid OOM.

    Returns:
        torch.Tensor: Indices of the nearest neighbors in train_index.
    """
    num_test = test_embeddings.size(0)
    nearest_indices = []

    # Move train index to GPU for fast retrieval
    # Assuming A100 memory (40GB) is sufficient for ~1.5M x 256 floats (~1.5GB)
    train_index = train_index.to(device)

    # Process queries in batches
    for i in range(0, num_test, batch_size):
        end = min(i + batch_size, num_test)
        batch_test = test_embeddings[i:end].to(device)

        # Compute Cosine Similarity: (B, D) @ (D, N_train) -> (B, N_train)
        # Since vectors are normalized, dot product equals cosine similarity.
        sim_matrix = torch.matmul(batch_test, train_index.t())

        # Find index of maximum similarity
        _, indices = torch.max(sim_matrix, dim=1)

        nearest_indices.append(indices.cpu())

    # Cleanup GPU memory
    del train_index
    torch.cuda.empty_cache()

    return torch.cat(nearest_indices, dim=0)


def run_retrieval_inference(
    model_path=Config.MODEL_PATH,
    batch_size=Config.VAL_BATCH_SIZE,
    device=Config.DEVICE,
    load_cached_index=True,
    debug=False,
):
    """
    Main driver for the retrieval inference pipeline.
    Generates the submission file.

    Args:
        model_path (str): Path to the trained model weights.
        batch_size (int): Batch size for inference dataloaders.
        device (torch.device): Computation device.
        load_cached_index (bool): Whether to use cached training embeddings.
        debug (bool): If True, runs on a small subset of data.
    """
    print("Starting Retrieval Inference Pipeline...")

    # 1. Load Metadata
    print("Loading metadata...")
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    if debug:
        print(
            f"Debug mode enabled. Using {Config.DEBUG_SAMPLE_SIZE} samples for train and test."
        )
        df_train = df_train.iloc[: Config.DEBUG_SAMPLE_SIZE]
        df_test = df_test.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # 2. Load Model
    print(f"Initializing model and loading weights from {model_path}...")
    model = StoichiometryEncoder(
        backbone_name=Config.BACKBONE,
        pretrained=False,  # We load custom weights, no need to download ImageNet weights
        embedding_dim=Config.EMBEDDING_DIM,
        num_atoms=Config.NUM_ATOMS,
    )

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("WARNING: Model weights not found. Using random initialization.")

    model = model.to(device)

    # 3. Build Training Index
    # Use 'val' mode transforms for deterministic, unaugmented inference
    print("Preparing training data for indexing...")
    train_dataset = ChemicalDataset(df_train, mode="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    train_embeddings = build_index(
        model, train_loader, device, load_cached_data=load_cached_index
    )
    print(f"Training index shape: {train_embeddings.shape}")

    # 4. Extract Test Embeddings
    print("Preparing test data for query...")
    test_dataset = ChemicalDataset(df_test, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print("Computing test embeddings...")
    test_embeddings = extract_embeddings(model, test_loader, device)
    print(f"Test embeddings shape: {test_embeddings.shape}")

    # 5. Query Index
    print("Querying index for nearest neighbors...")
    nearest_indices = query_index(test_embeddings, train_embeddings, device)

    # 6. Map Indices to Labels
    print("Mapping retrieved indices to InChI labels...")
    indices_np = nearest_indices.numpy()

    # Retrieve InChI strings from the training dataframe using the indices
    # df_train was not shuffled in the loader, so indices align directly
    predicted_inchis = df_train.iloc[indices_np]["InChI"].values

    # 7. Generate Submission
    print("Generating submission file...")
    submission = pd.DataFrame(
        {"image_id": df_test["image_id"].values, "InChI": predicted_inchis}
    )

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("First 5 predictions:")
    print(submission.head())
