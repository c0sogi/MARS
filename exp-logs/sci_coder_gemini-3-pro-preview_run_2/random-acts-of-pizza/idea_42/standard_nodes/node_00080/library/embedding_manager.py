import os
import numpy as np
import torch
import gc
from sentence_transformers import SentenceTransformer
from library import config
from library import utils

# Initialize Logger
logger = utils.setup_logger(
    "embedding_manager", os.path.join(config.WORKING_DIR, "embedding_manager.log")
)


def _compute_embeddings(texts, model_name, batch_size=64):
    """
    Computes embeddings for a list of texts using a SentenceTransformer model.

    Args:
        texts (list or pd.Series): The text data to encode.
        model_name (str): The HuggingFace model identifier.
        batch_size (int): Batch size for inference.

    Returns:
        np.ndarray: The computed embeddings.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading model {model_name} on {device}...")

    model = SentenceTransformer(model_name, device=device)
    model.eval()

    # Ensure deterministic behavior
    utils.set_seed(config.SEED)

    logger.info(f"Encoding {len(texts)} samples...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,  # We handle normalization in the pipeline/logic if needed
    )

    # Cleanup
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return embeddings


def get_embeddings(train_df, val_df, test_df, load_cached_data=True):
    """
    Retrieves embeddings for Train, Validation, and Test sets.
    Checks for cached .npy files first. If missing or forced, computes them.

    Args:
        train_df (pd.DataFrame): Training data.
        val_df (pd.DataFrame): Validation data.
        test_df (pd.DataFrame): Test data.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: A dictionary containing numpy arrays for all views and splits.
              Keys match those in config.CACHE_FILES (e.g., 'train_title_emb').
    """
    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Define the expected keys based on config.CACHE_FILES
    required_keys = [
        "train_title_emb",
        "train_body_emb",
        "train_global_emb",
        "val_title_emb",
        "val_body_emb",
        "val_global_emb",
        "test_title_emb",
        "test_body_emb",
        "test_global_emb",
    ]

    # Check if all cache files exist
    all_cached = True
    for key in required_keys:
        if key not in config.CACHE_FILES:
            continue  # Should not happen if config is correct
        if not os.path.exists(config.CACHE_FILES[key]):
            all_cached = False
            break

    if load_cached_data and all_cached:
        logger.info("All embedding files found in cache. Loading...")
        embeddings = {}
        for key in required_keys:
            path = config.CACHE_FILES[key]
            embeddings[key] = np.load(path)
        return embeddings

    logger.info("Cache miss or forced re-computation. Generating embeddings...")

    embeddings = {}

    # =========================================================================
    # 1. High-Resolution Anchors (MiniLM) - Title and Body
    # =========================================================================
    # We process Title and Body sequentially with the same model to save loading time

    # Prepare text lists
    train_titles = train_df[config.TEXT_COL_TITLE].fillna("").astype(str).tolist()
    val_titles = val_df[config.TEXT_COL_TITLE].fillna("").astype(str).tolist()
    test_titles = test_df[config.TEXT_COL_TITLE].fillna("").astype(str).tolist()

    train_bodies = train_df[config.TEXT_COL_BODY].fillna("").astype(str).tolist()
    val_bodies = val_df[config.TEXT_COL_BODY].fillna("").astype(str).tolist()
    test_bodies = test_df[config.TEXT_COL_BODY].fillna("").astype(str).tolist()

    # Load MiniLM once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Anchor Model: {config.MINILM_MODEL_NAME}")
    minilm = SentenceTransformer(config.MINILM_MODEL_NAME, device=device)
    minilm.eval()

    # Helper for batch encoding with current loaded model
    def encode_with_model(model, texts):
        return model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

    logger.info("Encoding Titles (Train/Val/Test)...")
    embeddings["train_title_emb"] = encode_with_model(minilm, train_titles)
    embeddings["val_title_emb"] = encode_with_model(minilm, val_titles)
    embeddings["test_title_emb"] = encode_with_model(minilm, test_titles)

    logger.info("Encoding Bodies (Train/Val/Test)...")
    embeddings["train_body_emb"] = encode_with_model(minilm, train_bodies)
    embeddings["val_body_emb"] = encode_with_model(minilm, val_bodies)
    embeddings["test_body_emb"] = encode_with_model(minilm, test_bodies)

    # Free MiniLM memory
    del minilm
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # =========================================================================
    # 2. Deep Context Auxiliary (MPNet) - Global (Title + Body)
    # =========================================================================

    # 'text_concat' is created by data_loader.extract_text_fields
    train_global = train_df["text_concat"].fillna("").astype(str).tolist()
    val_global = val_df["text_concat"].fillna("").astype(str).tolist()
    test_global = test_df["text_concat"].fillna("").astype(str).tolist()

    logger.info(f"Loading Global Context Model: {config.MPNET_MODEL_NAME}")
    mpnet = SentenceTransformer(config.MPNET_MODEL_NAME, device=device)
    mpnet.eval()

    logger.info("Encoding Global Context (Train/Val/Test)...")
    embeddings["train_global_emb"] = encode_with_model(mpnet, train_global)
    embeddings["val_global_emb"] = encode_with_model(mpnet, val_global)
    embeddings["test_global_emb"] = encode_with_model(mpnet, test_global)

    # Free MPNet memory
    del mpnet
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # =========================================================================
    # 3. Save to Cache
    # =========================================================================
    logger.info("Saving embeddings to cache...")
    for key, arr in embeddings.items():
        if key in config.CACHE_FILES:
            path = config.CACHE_FILES[key]
            np.save(path, arr)

    logger.info("Embedding generation complete.")
    return embeddings
