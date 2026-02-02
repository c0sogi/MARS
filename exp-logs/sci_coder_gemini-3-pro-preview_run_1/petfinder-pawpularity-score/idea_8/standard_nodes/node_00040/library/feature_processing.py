import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from library.config import Config
from library.utils import save_array, load_array, seed_everything
from library.feature_extraction import get_cache_paths
from library.dataset import load_dataset


class FeatureProcessor:
    """
    Handles feature preprocessing for Level-0 experts.
    Implements two pipelines:
    1. Linear/Kernel: Concat(Embeddings, Meta) -> StandardScaler
    2. Tree: PCA(Embeddings) -> Concat(PCA_Embeddings, Meta)
    """

    def __init__(self, expert_group):
        """
        Args:
            expert_group (str): 'linear' for Ridge/SVR, 'tree' for ExtraTrees.
        """
        self.expert_group = expert_group
        self.scaler = None
        self.pca = None

    def fit(self, embeddings, meta):
        """
        Fits the necessary transformers (Scaler or PCA) on the training data.
        """
        if self.expert_group == "linear":
            # For Linear/Kernel: Scale everything
            # Construct full vector to fit scaler
            full_vector = self._concat(embeddings, meta)
            self.scaler = StandardScaler()
            self.scaler.fit(full_vector)

        elif self.expert_group == "tree":
            # For Tree: PCA on embeddings only
            self.pca = PCA(n_components=Config.PCA_COMPONENTS, random_state=Config.SEED)
            self.pca.fit(embeddings)

        return self

    def transform(self, embeddings, meta):
        """
        Applies the transformations.
        """
        if self.expert_group == "linear":
            full_vector = self._concat(embeddings, meta)
            if self.scaler is None:
                raise RuntimeError("Processor not fitted. Call fit() first.")
            return self.scaler.transform(full_vector)

        elif self.expert_group == "tree":
            if self.pca is None:
                raise RuntimeError("Processor not fitted. Call fit() first.")

            # Apply PCA to embeddings
            embeddings_pca = self.pca.transform(embeddings)

            # Concatenate with meta
            return self._concat(embeddings_pca, meta)

        else:
            raise ValueError(f"Unknown expert group: {self.expert_group}")

    def _concat(self, embeddings, meta):
        """
        Helper to concatenate components.
        """
        components = [embeddings]
        if meta is not None:
            components.append(meta)
        return np.concatenate(components, axis=1)


def _load_raw_data(model_name, split):
    """
    Loads raw features, ids, meta, and targets (if avail) for a specific model and split.
    """
    paths = get_cache_paths(model_name, split)

    # Load primary features
    try:
        embeddings = load_array(paths["features"])
        meta = load_array(paths["meta"])
        ids = load_array(paths["ids"])
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not load features for {model_name} ({split}). Run feature extraction first."
        )

    targets = None
    if os.path.exists(paths["targets"]):
        targets = load_array(paths["targets"])

    return embeddings, meta, targets, ids


def get_expert_data(model_name, expert_group, load_cached_data=True):
    """
    Main function to get processed data for Level-0 experts.

    Args:
        model_name (str): Config.MODEL_CLIP, Config.MODEL_DINO, or Config.MODEL_CONVNEXT
        expert_group (str): 'linear' (for Ridge/SVR) or 'tree' (for ExtraTrees)
        load_cached_data (bool): Whether to use cached processed files.

    Returns:
        dict: {
            'X_train': np.ndarray, 'y_train': np.ndarray,
            'X_val': np.ndarray, 'y_val': np.ndarray,
            'X_test': np.ndarray, 'ids_test': np.ndarray
        }
    """
    seed_everything(Config.SEED)

    # Define cache paths for processed data
    cache_dir = Config.WORKING_DIR
    # Create a unique prefix based on model and expert type
    prefix = f"processed_{model_name.replace('/', '_')}_{expert_group}"

    cache_files = {
        "X_train": os.path.join(cache_dir, f"{prefix}_X_train.npy"),
        "y_train": os.path.join(cache_dir, f"{prefix}_y_train.npy"),
        "X_val": os.path.join(cache_dir, f"{prefix}_X_val.npy"),
        "y_val": os.path.join(cache_dir, f"{prefix}_y_val.npy"),
        "X_test": os.path.join(cache_dir, f"{prefix}_X_test.npy"),
        "ids_test": os.path.join(cache_dir, f"{prefix}_ids_test.npy"),
    }

    # 1. Try Loading Cache
    if load_cached_data:
        # Check if all required files exist
        all_exist = all(os.path.exists(p) for p in cache_files.values())
        if all_exist:
            try:
                data = {k: load_array(v) for k, v in cache_files.items()}

                # Cite debug_lesson_5: Validate Cached Artifacts Against Current Runtime Configuration
                df_check = load_dataset("train")
                if len(data["y_train"]) != len(df_check):
                    raise ValueError(
                        f"Cache mismatch: Expected {len(df_check)} samples, found {len(data['y_train'])}."
                    )

                print(f"Loading processed data for {model_name} ({expert_group})...")
                return data
            except Exception as e:
                print(f"Cache invalid or load failed: {e}. Recomputing...")

    print(f"Processing data for {model_name} ({expert_group})...")

    # 2. Load Raw Data
    # Train
    emb_train, meta_train, y_train, _ = _load_raw_data(model_name, "train")
    # Val
    emb_val, meta_val, y_val, _ = _load_raw_data(model_name, "val")
    # Test
    emb_test, meta_test, _, ids_test = _load_raw_data(model_name, "test")

    # 3. Fit Processor
    processor = FeatureProcessor(expert_group)
    processor.fit(emb_train, meta_train)

    # 4. Transform
    X_train = processor.transform(emb_train, meta_train)
    X_val = processor.transform(emb_val, meta_val)
    X_test = processor.transform(emb_test, meta_test)

    # 5. Save to Cache
    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "ids_test": ids_test,
    }

    for k, v in data.items():
        if v is not None:
            save_array(cache_files[k], v)

    return data
