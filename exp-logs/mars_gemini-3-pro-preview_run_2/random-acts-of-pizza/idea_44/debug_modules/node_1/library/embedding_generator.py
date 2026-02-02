import os
import gc
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import setup_logger, save_to_cache, load_from_cache

# Initialize Logger
logger = setup_logger(
    "embedding_generator", os.path.join(Config.WORKING_DIR, "embedding_generator.log")
)


def generate_embeddings(df_train, df_val, df_test, load_cached_data: bool = True):
    """
    Generates or loads embeddings for Train, Validation, and Test sets.
    Computes 4 sets of embeddings per split:
      1. Anchor Title (MiniLM)
      2. Anchor Body (MiniLM)
      3. Aux Title (MPNet)
      4. Aux Body (MPNet)

    Args:
        df_train (pd.DataFrame): Training data.
        df_val (pd.DataFrame): Validation data.
        df_test (pd.DataFrame): Test data.
        load_cached_data (bool): If True, attempts to load from disk first.

    Returns:
        dict: A nested dictionary containing embeddings for each split.
              Structure:
              {
                  "train": {"anchor_title": np.array, "anchor_body": ..., ...},
                  "val": ...,
                  "test": ...
              }
    """
    # Define splits and their dataframes for iteration
    splits = {
        "train": df_train,
        "val": df_val,
        "test": df_test,
    }

    # Define the embedding types to generate
    # Key: Internal name, Value: (Model Name from Config, Text Column from Config)
    embedding_tasks = [
        ("anchor_title", Config.ANCHOR_MODEL_NAME, Config.TEXT_COL_TITLE),
        ("anchor_body", Config.ANCHOR_MODEL_NAME, Config.TEXT_COL_BODY),
        ("aux_title", Config.AUX_MODEL_NAME, Config.TEXT_COL_TITLE),
        ("aux_body", Config.AUX_MODEL_NAME, Config.TEXT_COL_BODY),
    ]

    # Construct cache paths
    # Structure: cache_paths[split][task_name] = path
    cache_paths = {}
    for split_name in splits.keys():
        cache_paths[split_name] = {}
        for task_name, _, _ in embedding_tasks:
            filename = f"{split_name}_{task_name}.npy"
            path = os.path.join(Config.WORKING_DIR, filename)
            cache_paths[split_name][task_name] = path

    # -------------------------------------------------------------------------
    # 1. Attempt to Load from Cache
    # -------------------------------------------------------------------------
    if load_cached_data:
        logger.info("Checking cache for existing embeddings...")
        all_cached = True
        loaded_data = {s: {} for s in splits}

        for split_name in splits:
            for task_name in cache_paths[split_name]:
                path = cache_paths[split_name][task_name]
                data = load_from_cache(path)
                if data is None:
                    logger.info(f"Cache miss: {path}")
                    all_cached = False
                    break
                loaded_data[split_name][task_name] = data
            if not all_cached:
                break

        if all_cached:
            logger.info("All embeddings successfully loaded from cache.")
            return loaded_data

    # -------------------------------------------------------------------------
    # 2. Compute Embeddings (Cache Miss or Force Recompute)
    # -------------------------------------------------------------------------
    logger.info("Generating embeddings from scratch...")

    # Initialize result structure
    results = {s: {} for s in splits}

    # Group tasks by model to minimize model loading/unloading
    # model_name -> list of (task_name, col_name)
    tasks_by_model = {}
    for task_name, model_name, col_name in embedding_tasks:
        if model_name not in tasks_by_model:
            tasks_by_model[model_name] = []
        tasks_by_model[model_name].append((task_name, col_name))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Inference device: {device}")

    for model_name, tasks in tasks_by_model.items():
        logger.info(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name, device=device)

        for task_name, col_name in tasks:
            logger.info(f"Encoding {task_name} using {model_name}...")

            for split_name, df in splits.items():
                texts = df[col_name].tolist()

                # Encode
                # show_progress_bar=False to keep logs clean as per instructions
                embeddings = model.encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,  # Normalization happens in the pipeline/trainer if needed
                )

                # Store in memory
                results[split_name][task_name] = embeddings

                # Save to cache
                save_path = cache_paths[split_name][task_name]
                save_to_cache(embeddings, save_path)
                logger.info(f"Saved {split_name} {task_name} to {save_path}")

        # Cleanup model to free GPU memory for the next one
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Embedding generation complete.")
    return results
