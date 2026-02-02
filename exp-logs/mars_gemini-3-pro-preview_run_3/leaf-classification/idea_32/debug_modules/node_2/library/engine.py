import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from library.config import Config
from library.utils import seed_everything, clip_probabilities
from library.data_processing import DataProcessor
from library.model_pipeline import LeafClassifier


class Engine:
    """
    Orchestrates the training and inference workflow for the Leaf Species Identification task.
    Implements Stratified K-Fold Cross-Validation with Manifold Densification and
    Full-Manifold Test-Time Aggregation.
    """

    def __init__(self):
        self.processor = DataProcessor()
        self.models = []
        self.class_names = None

    def run(self):
        """
        Executes the full pipeline: Training -> Inference -> Submission.
        """
        seed_everything(Config.SEED)
        self.train_folds()
        self.generate_submission()

    def train_folds(self):
        """
        Performs 10-fold Stratified Cross-Validation.
        Trains the LeafClassifier on densified data and evaluates using Log Loss.
        """
        print("Loading training data...")
        # Load densified data: IDs and labels are repeated 3 times per image
        # d_img: (N*3, D_img), d_tab: (N*3, D_tab)
        d_img, d_tab, d_ids, d_labels = self.processor.get_train_data(load_cached=True)

        # Concatenate visual and tabular features
        # Structure: [DINO | ConvNeXt | Tabular]
        X = np.concatenate([d_img, d_tab], axis=1)
        y = d_labels
        ids = d_ids

        # Get unique IDs for stratification to prevent leakage.
        # We must ensure all 3 centroids of an image end up in the same fold.
        # np.unique returns sorted unique elements.
        unique_ids, unique_indices = np.unique(ids, return_index=True)
        unique_labels = y[unique_indices]

        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        fold_scores = []
        self.models = []  # Reset models list

        print(f"Starting {Config.N_FOLDS}-Fold Cross-Validation...")

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(unique_ids, unique_labels)
        ):
            # Identify IDs for this fold
            fold_train_ids = unique_ids[train_idx]
            fold_val_ids = unique_ids[val_idx]

            # Create boolean masks for the densified dataset
            # np.isin preserves the relative order of the input array for the True values.
            # Since input is ordered as blocks of 3 [id1, id1, id1, id2...],
            # the masked arrays will also be ordered as blocks of 3.
            train_mask = np.isin(ids, fold_train_ids)
            val_mask = np.isin(ids, fold_val_ids)

            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]

            # Initialize and Train Classifier
            clf = LeafClassifier()
            clf.fit(X_train, y_train)
            self.models.append(clf)

            # Capture class names from the first model (assuming consistency)
            if self.class_names is None:
                # Access the classes_ attribute from the underlying sklearn pipeline
                self.class_names = clf.pipeline.classes_

            # Validation Inference: Full-Manifold Aggregation
            # 1. Predict on all validation centroids (Shape: N_val_samples * 3, N_classes)
            val_probs_all = clf.predict_proba(X_val)

            # 2. Aggregate per image (average over 3 centroids)
            # We rely on the preserved block structure of X_val
            n_val_images = len(fold_val_ids)

            # Reshape to (N_images, 3, N_classes) and average along axis 1
            val_probs_reshaped = val_probs_all.reshape(n_val_images, 3, -1)
            val_probs_avg = np.mean(val_probs_reshaped, axis=1)

            # Get true labels (1 per image)
            # y_val contains 3 repeats per image. We take every 3rd element.
            y_val_unique = y_val[::3]

            # Calculate Metric
            score = log_loss(y_val_unique, val_probs_avg, labels=self.class_names)
            fold_scores.append(score)

            print(f"Fold {fold + 1} Log Loss: {score:.15f}")

        print(f"Average Log Loss: {np.mean(fold_scores):.15f}")

    def generate_submission(self):
        """
        Generates predictions for the test set using the ensemble of trained models.
        Saves the result to submission.csv.
        """
        print("Loading test data...")
        # Load densified test data
        d_img, d_tab, d_ids, _ = self.processor.get_test_data(load_cached=True)
        X_test = np.concatenate([d_img, d_tab], axis=1)

        # Test IDs are repeated 3 times. Get unique list for submission to match rows.
        # Data is ordered [id1, id1, id1, id2, id2, id2...]
        test_ids_unique = d_ids[::3]
        n_test_images = len(test_ids_unique)

        print("Running inference on test set...")
        # Initialize accumulator for ensemble probabilities
        ensemble_probs = np.zeros((n_test_images, len(self.class_names)))

        for i, clf in enumerate(self.models):
            # Predict on all centroids (Shape: N_test * 3, N_classes)
            probs_all = clf.predict_proba(X_test)

            # Aggregate centroids: (N, 3, C) -> (N, C)
            probs_reshaped = probs_all.reshape(n_test_images, 3, -1)
            probs_avg = np.mean(probs_reshaped, axis=1)

            # Add to ensemble
            ensemble_probs += probs_avg

        # Average across models
        ensemble_probs /= len(self.models)

        # Clip probabilities to avoid log loss extremes
        ensemble_probs = clip_probabilities(ensemble_probs)

        # Create Submission DataFrame
        df_sub = pd.DataFrame(ensemble_probs, columns=self.class_names)
        df_sub.insert(0, "id", test_ids_unique)

        # Save
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Done.")
