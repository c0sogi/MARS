import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from library.config import Config
from library.utils import setup_logger

# Initialize logger
logger = setup_logger(name="embeddings")


def get_text_embeddings(df_train, df_val, df_test, load_cached_data=True):
    """
    Generates or loads SBERT embeddings for the 'full_text' column of the provided DataFrames.

    The process includes:
    1. checking for cached .npy files.
    2. If not found or load_cached_data is False:
       - Loading the SBERT model defined in Config.
       - Encoding the text.
       - Applying L2 normalization.
       - Saving the results to disk.

    Args:
        df_train (pd.DataFrame): Training data containing 'full_text'.
        df_val (pd.DataFrame): Validation data containing 'full_text'.
        df_test (pd.DataFrame): Test data containing 'full_text'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_embeddings, val_embeddings, test_embeddings) as numpy arrays.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache_path = os.path.join(cache_dir, "train_embeddings.npy")
    val_cache_path = os.path.join(cache_dir, "val_embeddings.npy")
    test_cache_path = os.path.join(cache_dir, "test_embeddings.npy")

    # 1. Check Cache
    if load_cached_data:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
            and os.path.exists(test_cache_path)
        ):
            logger.info("Loading cached text embeddings from disk...")
            try:
                train_embeddings = np.load(train_cache_path)
                val_embeddings = np.load(val_cache_path)
                test_embeddings = np.load(test_cache_path)
                logger.info("Successfully loaded cached embeddings.")
                return train_embeddings, val_embeddings, test_embeddings
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Re-computing embeddings.")
        else:
            logger.info("Cached embeddings not found. Computing from scratch...")
    else:
        logger.info("Ignoring cache. Computing embeddings from scratch...")

    # 2. Load Model
    logger.info(f"Loading SBERT model: {Config.SBERT_MODEL_NAME}...")
    model = SentenceTransformer(Config.SBERT_MODEL_NAME)

    # Ensure model is in eval mode (though sentence-transformers usually handles this)
    model.eval()

    # 3. Helper function to encode and normalize
    def process_split(df, split_name):
        logger.info(f"Encoding {split_name} split ({len(df)} samples)...")
        texts = df["full_text"].tolist()

        # Encode
        # batch_size=32 is standard; show_progress_bar=False to reduce clutter
        embeddings = model.encode(
            texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )

        # Apply L2 Normalization (Project to hypersphere)
        # SBERT models often output normalized vectors, but explicit L2 ensures it.
        embeddings = normalize(embeddings, norm="l2", axis=1)

        return embeddings

    # 4. Process all splits
    train_embeddings = process_split(df_train, "Train")
    val_embeddings = process_split(df_val, "Validation")
    test_embeddings = process_split(df_test, "Test")

    # 5. Save to Cache
    logger.info("Saving computed embeddings to cache...")
    np.save(train_cache_path, train_embeddings)
    np.save(val_cache_path, val_embeddings)
    np.save(test_cache_path, test_embeddings)

    logger.info("Embedding generation complete.")

    return train_embeddings, val_embeddings, test_embeddings
