import os
import numpy as np
import scipy.sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import set_seed


class RFLearner:
    """
    Random Forest Learner for the Stream A of the Hybrid Ensemble.
    Handles feature assembly, training, and inference.
    """

    def __init__(self, cache_dir=Config.WORKING_DIR):
        """
        Initialize the learner.

        Args:
            cache_dir (str): Directory to store cached assembled feature matrices.
        """
        self.cache_dir = cache_dir
        self.params = Config.RF_PARAMS
        self.model = None

    def _assemble_features(self, components, split_name, load_cached_data=True):
        """
        Assembles various feature components into a single sparse matrix.
        Implements caching to avoid re-stacking on subsequent runs.

        Args:
            components (dict): Dictionary containing feature arrays:
                - 'tfidf': Sparse matrix
                - 'metadata': Dense array
                - 'top_k': Dense array
                - 'prototypes': Dense array
                - 'sentiment': Dense array
            split_name (str): 'train', 'val', or 'test' for cache naming.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            scipy.sparse.csr_matrix: The assembled feature matrix.
        """
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, f"rf_assembled_{split_name}.npz")

        # Determine expected shape from input components
        # We use metadata as the reference for row count
        metadata_raw = components.get("metadata")
        expected_rows = metadata_raw.shape[0] if metadata_raw is not None else None

        # 1. Try Loading from Cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                cached_matrix = scipy.sparse.load_npz(cache_path)
                # Cite debug_lesson_1: Validate cache dimensions to prevent stale data usage
                if expected_rows is None or cached_matrix.shape[0] == expected_rows:
                    return cached_matrix
                else:
                    print(
                        f"Warning: Cached features for '{split_name}' have {cached_matrix.shape[0]} rows, "
                        f"but expected {expected_rows}. Re-assembling."
                    )
            except Exception:
                pass  # Fallback to processing if load fails

        # 2. Process (Assemble) Features
        # Extract components
        tfidf = components.get("tfidf")

        # Convert dense components to sparse CSR for efficient stacking
        # We use get() to allow flexibility, though all are expected
        metadata = scipy.sparse.csr_matrix(metadata_raw)
        top_k = scipy.sparse.csr_matrix(components.get("top_k"))
        prototypes = scipy.sparse.csr_matrix(components.get("prototypes"))
        sentiment = scipy.sparse.csr_matrix(components.get("sentiment"))

        # Stack horizontally
        # Order: TF-IDF | Metadata | Top-K | Prototypes | Sentiment
        assembled_matrix = scipy.sparse.hstack(
            [tfidf, metadata, top_k, prototypes, sentiment], format="csr"
        )

        # 3. Save to Cache
        try:
            scipy.sparse.save_npz(cache_path, assembled_matrix)
        except Exception:
            pass

        return assembled_matrix

    def train(
        self, train_components, y_train, val_components, y_val, load_cached_data=True
    ):
        """
        Trains the Random Forest model.

        Args:
            train_components (dict): Feature components for training.
            y_train (array-like): Training labels.
            val_components (dict): Feature components for validation.
            y_val (array-like): Validation labels.
            load_cached_data (bool): Whether to use cached assembled features.

        Returns:
            RandomForestClassifier: The trained model.
        """
        # Set seed for reproducibility
        set_seed(Config.RANDOM_SEED)

        # Assemble features
        X_train = self._assemble_features(train_components, "train", load_cached_data)
        X_val = self._assemble_features(val_components, "val", load_cached_data)

        # Initialize model
        self.model = RandomForestClassifier(**self.params)

        # Fit model
        self.model.fit(X_train, y_train)

        # Evaluate
        # Predict probability for the positive class (index 1)
        val_preds = self.model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_preds)

        # Print metric with full precision
        print(f"Random Forest Validation AUC: {auc}")

        return self.model

    def predict(self, test_components, load_cached_data=True):
        """
        Generates predictions for the test set.

        Args:
            test_components (dict): Feature components for testing.
            load_cached_data (bool): Whether to use cached assembled features.

        Returns:
            np.ndarray: Predicted probabilities for the positive class.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Assemble features
        X_test = self._assemble_features(test_components, "test", load_cached_data)

        # Predict
        probs = self.model.predict_proba(X_test)[:, 1]

        return probs
