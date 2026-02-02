import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import setup_logger, save_array, load_array
from library.data_loader import load_and_process_data

logger = setup_logger("embedding_manager")


def get_feature_schema():
    """
    Defines the column indices for the concatenated feature matrix.

    Returns:
        dict: Mapping of feature view names to (start_index, end_index) tuples.
    """
    # Dimensions based on models defined in Config
    # MiniLM-L6-v2 outputs 384 dimensions
    # MPNet-base-v2 outputs 768 dimensions
    dim_anchor = 384
    dim_aux = 768
    dim_meta = len(Config.NUMERICAL_FEATURES)

    # Calculate offsets
    start_title = 0
    end_title = start_title + dim_anchor

    start_body = end_title
    end_body = start_body + dim_anchor

    start_global = end_body
    end_global = start_global + dim_aux

    start_meta = end_global
    end_meta = start_meta + dim_meta

    schema = {
        "title": (start_title, end_title),
        "body": (start_body, end_body),
        "global": (start_global, end_global),
        "meta": (start_meta, end_meta),
        "total_dims": end_meta,
    }
    return schema


def compute_embeddings(df, model_anchor, model_aux, device):
    """
    Computes the multi-view embeddings for a given DataFrame.

    Args:
        df (pd.DataFrame): The dataframe containing text and metadata.
        model_anchor (SentenceTransformer): Model for Title and Body views.
        model_aux (SentenceTransformer): Model for Global Context view.
        device (str): Device to run inference on.

    Returns:
        np.ndarray: Concatenated feature matrix.
    """
    # 1. Prepare Text Inputs
    titles = df[Config.TEXT_COL_TITLE].tolist()
    bodies = df[Config.TEXT_COL_BODY].tolist()
    # Global context is concatenation of Title and Body
    global_texts = [t + " " + b for t, b in zip(titles, bodies)]

    # 2. Encode Views
    # View 1: Title (Anchor Model)
    logger.info("Encoding Title View...")
    emb_title = model_anchor.encode(
        titles,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        device=device,
    )

    # View 2: Body (Anchor Model)
    logger.info("Encoding Body View...")
    emb_body = model_anchor.encode(
        bodies,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        device=device,
    )

    # View 3: Global Context (Aux Model)
    logger.info("Encoding Global Context View...")
    emb_global = model_aux.encode(
        global_texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        device=device,
    )

    # View 4: Metadata
    logger.info("Extracting Metadata...")
    meta_features = df[Config.NUMERICAL_FEATURES].values.astype(np.float32)

    # 3. Concatenate
    # [Title (384) | Body (384) | Global (768) | Meta (N)]
    X = np.hstack([emb_title, emb_body, emb_global, meta_features])

    return X


def generate_embeddings(load_cached_data=True):
    """
    Main function to generate or load embeddings for Train, Val, and Test sets.

    Args:
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, feature_schema)
    """
    # Define paths
    path_train_emb = Config.CACHE_TRAIN_EMBEDDINGS
    path_val_emb = Config.CACHE_VAL_EMBEDDINGS
    path_test_emb = Config.CACHE_TEST_EMBEDDINGS

    # Get Schema
    schema = get_feature_schema()

    # Check Cache
    cache_exists = (
        os.path.exists(path_train_emb)
        and os.path.exists(path_val_emb)
        and os.path.exists(path_test_emb)
    )

    # Load DataFrames to get targets (y) and for processing if cache missing
    # We always load dataframes because we need 'y' which is stored in parquet,
    # and load_and_process_data handles its own caching efficiently.
    df_train, df_val, df_test = load_and_process_data(load_cached_data=load_cached_data)

    # Extract Targets
    y_train = df_train["requester_received_pizza"].values.astype(int)
    y_val = df_val["requester_received_pizza"].values.astype(int)

    if load_cached_data and cache_exists:
        logger.info("Loading embeddings from cache...")
        try:
            X_train = load_array(path_train_emb)
            X_val = load_array(path_val_emb)
            X_test = load_array(path_test_emb)

            # Verify dimensions
            if X_train.shape[1] != schema["total_dims"]:
                logger.warning(
                    f"Cached embedding dimension {X_train.shape[1]} mismatch expected {schema['total_dims']}. Recomputing."
                )
                raise ValueError("Dimension mismatch")

            return X_train, y_train, X_val, y_val, X_test, schema
        except Exception as e:
            logger.warning(f"Failed to load cached embeddings: {e}. Recomputing...")

    # Compute from scratch
    logger.info("Computing embeddings from scratch...")

    # Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load Models
    logger.info(f"Loading Anchor Model: {Config.MODEL_ANCHOR}")
    model_anchor = SentenceTransformer(Config.MODEL_ANCHOR, device=device)

    logger.info(f"Loading Aux Model: {Config.MODEL_AUX}")
    model_aux = SentenceTransformer(Config.MODEL_AUX, device=device)

    # Compute
    logger.info("Processing Training Set...")
    X_train = compute_embeddings(df_train, model_anchor, model_aux, device)

    logger.info("Processing Validation Set...")
    X_val = compute_embeddings(df_val, model_anchor, model_aux, device)

    logger.info("Processing Test Set...")
    X_test = compute_embeddings(df_test, model_anchor, model_aux, device)

    # Save to Cache
    logger.info(f"Saving embeddings to {Config.WORKING_DIR}...")
    save_array(X_train, path_train_emb)
    save_array(X_val, path_val_emb)
    save_array(X_test, path_test_emb)

    # Cleanup models to free GPU memory
    del model_anchor
    del model_aux
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return X_train, y_train, X_val, y_val, X_test, schema
