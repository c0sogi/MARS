import os
import numpy as np
import joblib
from library.utils import save_submission, WORKING_DIR
from library.feature_extractor import FeatureExtractor
from library.densification import Densifier


class Predictor:
    """
    Manages the inference process using the ensemble of trained models.
    Implements Full-Manifold Test-Time Aggregation.
    """

    def __init__(self, cache_subdir="idea_42"):
        self.cache_subdir = cache_subdir
        self.models_dir = os.path.join(WORKING_DIR, cache_subdir, "models")
        if not os.path.exists(self.models_dir):
            raise FileNotFoundError(
                f"Models directory not found at {self.models_dir}. Train models first."
            )

    def generate_submission(self, load_cached_data=True, limit=None):
        """
        Generates predictions for the test set and saves the submission file.

        Args:
            load_cached_data (bool): Whether to use cached extracted features.
            limit (int): Optional limit on the number of test samples (for debugging).
        """
        print("Starting Inference Process...")

        # 1. Extract Features for Test Set
        extractor = FeatureExtractor()
        print("Loading test features...")
        dino_feats, conv_feats, tab_feats, ids = extractor.extract_and_save_features(
            "test", load_cached_data=load_cached_data, limit=limit
        )

        # 2. Densify Features (Canonical 3x Centroids)
        densifier = Densifier(cache_subdir=self.cache_subdir)
        print("Densifying test features (Canonical 3x Centroids)...")
        dino_canon, conv_canon, tab_canon, ids_canon = densifier.densify_inference_data(
            dino_feats,
            conv_feats,
            tab_feats,
            ids,
            split_name="test",
            load_cached_data=load_cached_data,
        )

        # 3. Construct Feature Matrix
        # Stack: [DINO (1024) | ConvNeXt (1536) | Tabular (192)]
        X_test_expanded = np.hstack([dino_canon, conv_canon, tab_canon])

        # 4. Load Class Names
        classes_path = os.path.join(self.models_dir, "classes.pkl")
        if not os.path.exists(classes_path):
            raise FileNotFoundError(f"Classes file not found at {classes_path}")
        classes = joblib.load(classes_path)
        n_classes = len(classes)
        n_samples = len(ids)

        # 5. Ensemble Prediction
        # Initialize accumulator for probabilities
        avg_probs = np.zeros((n_samples, n_classes), dtype=np.float64)

        # We expect 10 folds
        n_folds = 10
        successful_folds = 0

        print(f"Aggregating predictions across {n_folds} folds...")

        for fold_idx in range(n_folds):
            model_path = os.path.join(self.models_dir, f"pipeline_fold_{fold_idx}.pkl")

            if not os.path.exists(model_path):
                print(
                    f"Warning: Model for fold {fold_idx} not found at {model_path}. Skipping."
                )
                continue

            # Load Model
            pipeline = joblib.load(model_path)

            # Predict on expanded set (3 views per image)
            # Shape: (N * 3, n_classes)
            probs_expanded = pipeline.predict_proba(X_test_expanded)

            # Align predictions to global label space (Cite debug_lesson_2)
            if probs_expanded.shape[1] != n_classes:
                full_probs = np.zeros((probs_expanded.shape[0], n_classes))
                full_probs[:, pipeline.classes_] = probs_expanded
                probs_expanded = full_probs

            # Reshape to (N, 3, n_classes) to separate views per image
            probs_reshaped = probs_expanded.reshape(n_samples, 3, n_classes)

            # Average across the 3 views (Intra-model aggregation)
            # Shape: (N, n_classes)
            probs_fold = np.mean(probs_reshaped, axis=1)

            # Accumulate
            avg_probs += probs_fold
            successful_folds += 1

        if successful_folds == 0:
            raise RuntimeError(
                "No models were successfully loaded. Cannot generate submission."
            )

        # Average across folds (Inter-model aggregation)
        avg_probs /= successful_folds
        print(f"Inference complete. Aggregated {successful_folds} models.")

        # 6. Save Submission
        save_submission(ids, classes, avg_probs, filename="submission.csv")
