import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("FeatureExtractor")


def generate_sbert_embeddings(
    df_train, df_val, df_test, load_cached_data: bool = Config.LOAD_CACHED_DATA
):
    """
    Generates or loads SBERT embeddings for the text data.

    Args:
        df_train (pd.DataFrame): Training data containing 'text_combined'.
        df_val (pd.DataFrame): Validation data containing 'text_combined'.
        df_test (pd.DataFrame): Test data containing 'text_combined'.
        load_cached_data (bool): Whether to load from disk if available.

    Returns:
        tuple: (train_embeddings, val_embeddings, test_embeddings) as numpy arrays.
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_emb_path = os.path.join(cache_dir, "train_embeddings.npy")
    val_emb_path = os.path.join(cache_dir, "val_embeddings.npy")
    test_emb_path = os.path.join(cache_dir, "test_embeddings.npy")

    # 1. Try Loading from Cache
    if load_cached_data:
        if (
            os.path.exists(train_emb_path)
            and os.path.exists(val_emb_path)
            and os.path.exists(test_emb_path)
        ):
            logger.info("Loading SBERT embeddings from cache...")
            try:
                train_embeddings = np.load(train_emb_path)
                val_embeddings = np.load(val_emb_path)
                test_embeddings = np.load(test_emb_path)

                # Verify shapes match current dataframes (crucial for Debug mode vs Full mode)
                if (
                    len(train_embeddings) == len(df_train)
                    and len(val_embeddings) == len(df_val)
                    and len(test_embeddings) == len(df_test)
                ):
                    return train_embeddings, val_embeddings, test_embeddings
                else:
                    logger.warning(
                        f"Cached embeddings shape mismatch (Train: {len(train_embeddings)} vs {len(df_train)}). "
                        "Re-computing..."
                    )
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}. Re-computing...")
        else:
            logger.info("Embedding cache files not found. Computing from scratch...")
    else:
        logger.info("Cache loading disabled. Computing embeddings from scratch...")

    # 2. Compute Embeddings
    logger.info(f"Initializing SBERT model: {Config.SBERT_MODEL_NAME}")

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model = SentenceTransformer(Config.SBERT_MODEL_NAME, device=device)

    # Helper function to encode and normalize
    def process_text(df, name):
        logger.info(f"Encoding {name} set ({len(df)} samples)...")
        texts = df["text_combined"].fillna("").astype(str).tolist()

        # Encode
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # We normalize explicitly below
        )

        # Apply L2 Normalization (Project to Hypersphere)
        # Note: SBERT can do this internally, but we do it explicitly to match the strategy description
        embeddings = normalize(embeddings, norm="l2", axis=1)

        return embeddings

    train_embeddings = process_text(df_train, "Train")
    val_embeddings = process_text(df_val, "Validation")
    test_embeddings = process_text(df_test, "Test")

    # 3. Save to Cache
    logger.info(f"Saving embeddings to {cache_dir}...")
    try:
        np.save(train_emb_path, train_embeddings)
        np.save(val_emb_path, val_embeddings)
        np.save(test_emb_path, test_embeddings)
    except Exception as e:
        logger.error(f"Failed to save embedding cache: {e}")

    return train_embeddings, val_embeddings, test_embeddings
