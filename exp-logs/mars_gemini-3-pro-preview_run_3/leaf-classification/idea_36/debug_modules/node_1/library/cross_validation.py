import os
import numpy as np
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import save_pickle, seed_everything
from library.densification import ManifoldDensifier
from library.modeling import create_hybrid_pipeline


class EnsembleTrainer:
    """
    Orchestrates the Stratified K-Fold training process with Manifold Densification.
    """

    def __init__(self, config: Config):
        """
        Initialize the trainer.

        Args:
            config (Config): Configuration object.
        """
        self.config = config
        self.models_dir = os.path.join(self.config.WORKING_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.classes_path = os.path.join(self.models_dir, "classes.pkl")
        self.densifier = ManifoldDensifier(config)
        seed_everything(self.config.SEED)

    def _slice_dict(self, data_dict, indices):
        """
        Slices all numpy arrays in the dictionary using the provided indices.

        Args:
            data_dict (dict): Dictionary of numpy arrays.
            indices (np.ndarray): Indices to select.

        Returns:
            dict: Sliced dictionary.
        """
        subset = {}
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                subset[key] = value[indices]
        return subset

    def train_loop(self, data_dict):
        """
        Executes the Stratified K-Fold training loop.

        Splits the original (12-view) data, densifies it per fold, trains the pipeline,
        and evaluates performance.

        Args:
            data_dict (dict): Dictionary containing 'ids', 'dino', 'conv', 'tab', 'labels'.
                              Visual features are expected to be (N, 12, D).

        Returns:
            float: Average Log Loss across all folds.
        """
        logging.info("Starting Ensemble Training Loop...")

        ids = data_dict["ids"]
        labels_raw = data_dict["labels"]

        # Encode Labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(labels_raw)

        # Save classes for inference
        save_pickle(le.classes_, self.classes_path)
        logging.info(f"Saved class encoding to {self.classes_path}")

        # Create a copy of data_dict with encoded labels for processing
        data_encoded = data_dict.copy()
        data_encoded["labels"] = y_encoded

        # Initialize Stratified K-Fold
        skf = StratifiedKFold(
            n_splits=self.config.N_FOLDS, shuffle=True, random_state=self.config.SEED
        )

        scores = []

        # Iterate over folds
        for fold, (train_idx, val_idx) in enumerate(skf.split(ids, y_encoded)):
            logging.info(f"Processing Fold {fold}...")

            # 1. Slice Data into Train/Val subsets (still 12-view format)
            train_subset = self._slice_dict(data_encoded, train_idx)
            val_subset = self._slice_dict(data_encoded, val_idx)

            # 2. Densify Subsets (Expand to 6 centroids per image)
            # We use specific cache names for each fold to avoid collisions
            train_dense = self.densifier.densify_dataset(
                train_subset, dataset_name=f"fold_{fold}_train", load_cached_data=True
            )
            val_dense = self.densifier.densify_dataset(
                val_subset, dataset_name=f"fold_{fold}_val", load_cached_data=True
            )

            # 3. Prepare Feature Matrices
            # Concatenate: DINO + Conv + Tabular
            X_train = np.hstack(
                [train_dense["dino"], train_dense["conv"], train_dense["tab"]]
            )
            y_train = train_dense["labels"]

            X_val = np.hstack([val_dense["dino"], val_dense["conv"], val_dense["tab"]])

            # 4. Create Pipeline
            # Determine input dimensions from the data
            dino_dim = train_dense["dino"].shape[1]
            conv_dim = train_dense["conv"].shape[1]
            tab_dim = train_dense["tab"].shape[1]

            pipeline = create_hybrid_pipeline(
                dino_dim, conv_dim, tab_dim, pca_variance=self.config.PCA_VARIANCE
            )

            # 5. Train
            pipeline.fit(X_train, y_train)

            # 6. Evaluate
            # Predict on dense validation set
            probs_dense = pipeline.predict_proba(X_val)

            # Aggregate predictions: Average over the 6 centroids for each original image
            # The densified data is ordered: 6 rows per original image
            n_val_samples = len(val_subset["labels"])
            probs_agg = probs_dense.reshape(n_val_samples, 6, -1).mean(axis=1)

            # Calculate Log Loss
            # Compare against the original encoded labels for this fold
            y_val_orig = val_subset["labels"]
            fold_score = log_loss(y_val_orig, probs_agg)

            scores.append(fold_score)
            logging.info(f"Fold {fold} Log Loss: {fold_score}")

            # 7. Save Model
            model_path = os.path.join(self.models_dir, f"pipeline_fold_{fold}.pkl")
            save_pickle(pipeline, model_path)

        avg_score = np.mean(scores)
        logging.info(f"Average CV Log Loss: {avg_score}")

        return avg_score
