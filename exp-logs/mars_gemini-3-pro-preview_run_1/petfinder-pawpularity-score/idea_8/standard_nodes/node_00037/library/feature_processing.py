import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from library.config import Config
from library.utils import save_array, load_array, seed_everything
from library.feature_extraction import get_cache_paths


class FeatureProcessor:
    """
    Handles feature preprocessing for Level-0 experts.
    Implements two pipelines:
    1. Linear/Kernel: Concat(Embeddings, Scores, Meta) -> StandardScaler
    2. Tree/Neighbor: PCA(Embeddings) -> Concat(PCA_Embeddings, Scores, Meta)
    """

    def __init__(self, expert_group):
        """
        Args:
            expert_group (str): 'linear' for Ridge/SVR, 'tree' for ExtraTrees/KNN.
        """
        self.expert_group = expert_group
        self.scaler = None
        self.pca = None
        self.n_prompts = len(Config.AESTHETIC_PROMPTS)

    def fit(self, embeddings, scores, meta):
        """
        Fits the necessary transformers (Scaler or PCA) on the training data.
        """
        if self.expert_group == "linear":
            # For Linear/Kernel: Scale everything
            # Construct full vector to fit scaler
            full_vector = self._concat(embeddings, scores, meta)
            self.scaler = StandardScaler()
            self.scaler.fit(full_vector)

        elif self.expert_group == "tree":
            # For Tree/Neighbor: PCA on embeddings only
            self.pca = PCA(n_components=Config.PCA_COMPONENTS, random_state=Config.SEED)
            self.pca.fit(embeddings)

        return self

    def transform(self, embeddings, scores, meta):
        """
        Applies the transformations.
        """
        if self.expert_group == "linear":
            full_vector = self._concat(embeddings, scores, meta)
            if self.scaler is None:
                raise RuntimeError("Processor not fitted. Call fit() first.")
            return self.scaler.transform(full_vector)

        elif self.expert_group == "tree":
            if self.pca is None:
                raise RuntimeError("Processor not fitted. Call fit() first.")

            # Apply PCA to embeddings
            embeddings_pca = self.pca.transform(embeddings)

            # Concatenate with scores and meta
            return self._concat(embeddings_pca, scores, meta)

        else:
            raise ValueError(f"Unknown expert group: {self.expert_group}")

    def _concat(self, embeddings, scores, meta):
        """
        Helper to concatenate components. Handles missing scores if any.
        """
        components = [embeddings]
        if scores is not None and scores.shape[1] > 0:
            components.append(scores)
        if meta is not None:
            components.append(meta)
        return np.concatenate(components, axis=1)


def _load_raw_data(model_name, split):
    """
    Loads raw features, ids, meta, and targets (if avail) for a specific model and split.
    Also attempts to load CLIP scores if the model is not CLIP (Zero-Shot Injection).
    """
    paths = get_cache_paths(model_name, split)

    # Load primary features
    try:
        features = load_array(paths["features"])
        meta = load_array(paths["meta"])
        ids = load_array(paths["ids"])
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not load features for {model_name} ({split}). Run feature extraction first."
        )

    targets = None
    if os.path.exists(paths["targets"]):
        targets = load_array(paths["targets"])

    # Separate embeddings and scores
    n_prompts = len(Config.AESTHETIC_PROMPTS)

    if model_name == Config.MODEL_CLIP:
        # CLIP features contain scores at the end (concatenated in extract_and_save_features)
        embeddings = features[:, :-n_prompts]
        scores = features[:, -n_prompts:]
    else:
        # DINO/ConvNeXt features are just embeddings
        embeddings = features
        scores = None

        # Try to inject CLIP scores (Zero-Shot Aesthetic Injection) for other backbones
        # We rely on the deterministic ordering of the dataset
        clip_paths = get_cache_paths(Config.MODEL_CLIP, split)
        if os.path.exists(clip_paths["features"]):
            try:
                clip_features = load_array(clip_paths["features"])
                # Verify length matches to ensure alignment
                if len(clip_features) == len(features):
                    scores = clip_features[:, -n_prompts:]
            except Exception:
                # If CLIP features are corrupt or unloadable, proceed without scores
                pass

    return embeddings, scores, meta, targets, ids


def get_expert_data(model_name, expert_group, load_cached_data=True):
    """
    Main function to get processed data for Level-0 experts.

    Args:
        model_name (str): Config.MODEL_CLIP, Config.MODEL_DINO, or Config.MODEL_CONVNEXT
        expert_group (str): 'linear' (for Ridge/SVR) or 'tree' (for ExtraTrees/KNN)
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
            print(f"Loading processed data for {model_name} ({expert_group})...")
            return {k: load_array(v) for k, v in cache_files.items()}

    print(f"Processing data for {model_name} ({expert_group})...")

    # 2. Load Raw Data
    # Train
    emb_train, scores_train, meta_train, y_train, _ = _load_raw_data(
        model_name, "train"
    )
    # Val
    emb_val, scores_val, meta_val, y_val, _ = _load_raw_data(model_name, "val")
    # Test
    emb_test, scores_test, meta_test, _, ids_test = _load_raw_data(model_name, "test")

    # 3. Fit Processor
    processor = FeatureProcessor(expert_group)
    processor.fit(emb_train, scores_train, meta_train)

    # 4. Transform
    X_train = processor.transform(emb_train, scores_train, meta_train)
    X_val = processor.transform(emb_val, scores_val, meta_val)
    X_test = processor.transform(emb_test, scores_test, meta_test)

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
