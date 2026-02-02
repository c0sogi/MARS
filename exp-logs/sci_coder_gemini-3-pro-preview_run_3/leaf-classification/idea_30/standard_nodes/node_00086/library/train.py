import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from library.config import Config
from library.data_processing import DatasetManager
from library.pipeline import ModelFactory


class Trainer:
    """
    Orchestrates the training and submission process for the Leaf Species Classification task.
    Implements Stratified K-Fold Cross-Validation with Manifold Densification and
    Test-Time Aggregation (TTA) over orthogonal centroids.
    """

    def __init__(self):
        self.dataset_manager = DatasetManager()
        self.model_factory = ModelFactory()
        # Directory to store trained models for the ensemble
        self.models_dir = os.path.join(Config.WORKING_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)

    def run(self):
        """
        Executes the full training pipeline:
        1. Loads and caches features.
        2. Performs Stratified K-Fold CV.
        3. Trains models on densified data (3x samples).
        4. Evaluates on validation data (aggregated centroids).
        5. Generates submission file using the full ensemble.
        """
        # 1. Load Data
        print("Loading data...")
        # load_cached_data=True ensures we use pre-computed features if available
        data = self.dataset_manager.load_data(load_cached_data=True)

        # 2. Setup Cross Validation
        skf = self.dataset_manager.get_stratified_kfold()

        # Extract training identifiers for splitting (N samples)
        train_ids_unique = data["train"]["ids"]
        train_labels = data["train"]["labels"]

        # Get feature indices for the pipeline (DINO vs Conv vs Tabular)
        feature_indices = self.dataset_manager.get_feature_indices()

        fold_scores = []
        classes = None

        print(f"Starting {Config.N_FOLDS}-Fold Stratified Cross-Validation...")

        # 3. Training Loop
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(train_ids_unique, train_labels)
        ):
            # --- Data Preparation ---
            # Prepare Densified Training Data (3x samples per image)
            # We flatten the 3 centroids into the batch dimension
            X_train, y_train, _ = self.dataset_manager.prepare_training_set(
                data["train"], indices=train_idx
            )

            # Prepare Densified Validation Data (3x samples per image)
            # We need X_val for prediction.
            X_val, _, _ = self.dataset_manager.prepare_training_set(
                data["train"], indices=val_idx
            )

            # Ground truth for validation (1 label per image)
            y_val_true = train_labels[val_idx]

            # --- Model Training ---
            pipeline = self.model_factory.create_pipeline(feature_indices)
            pipeline.fit(X_train, y_train)

            # Capture classes from the first fold (LDA sorts them automatically)
            if classes is None:
                classes = pipeline.named_steps["classifier"].classes_
                # Save classes for consistency in submission
                with open(os.path.join(self.models_dir, "classes.pkl"), "wb") as f:
                    pickle.dump(classes, f)

            # --- Evaluation ---
            # Predict on all 3 centroids for each validation image
            # Shape: (3 * N_val, n_classes)
            probs_expanded = pipeline.predict_proba(X_val)

            # Aggregation: Reshape to (N_val, 3, n_classes) and mean pooling
            # This strictly adheres to the "Train on the Aggregation" principle
            n_val = len(val_idx)
            n_classes = len(classes)
            probs_reshaped = probs_expanded.reshape(n_val, 3, n_classes)
            probs_mean = np.mean(probs_reshaped, axis=1)

            # Calculate Metric
            # Note: sklearn log_loss handles the clipping internally
            score = log_loss(y_val_true, probs_mean, labels=classes)
            print(f"Fold {fold} Log Loss: {score}")
            fold_scores.append(score)

            # --- Save Model ---
            model_path = os.path.join(self.models_dir, f"pipeline_fold_{fold}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(pipeline, f)

        # 4. Final Results
        avg_score = np.mean(fold_scores)
        print(f"Average Log Loss: {avg_score}")

        # 5. Submission
        self.generate_submission(data["test"], classes)

    def generate_submission(self, test_data_dict, classes):
        """
        Generates the submission file by averaging predictions from all K-Fold models.
        Applies Test-Time Aggregation (TTA) over the 3 orthogonal centroids.
        """
        print("Generating submission...")

        # Prepare Densified Test Data (3x samples per image)
        # Note: prepare_training_set returns (X, ids) when labels are missing
        X_test, test_ids_expanded = self.dataset_manager.prepare_training_set(
            test_data_dict
        )

        # Extract unique IDs (every 3rd element) to match the aggregated predictions
        # test_ids_expanded is [id1, id1, id1, id2, id2, id2...]
        test_ids = test_ids_expanded[::3]

        n_test = len(test_ids)
        n_classes = len(classes)

        # Accumulator for ensemble predictions
        ensemble_probs = np.zeros((n_test, n_classes))

        # Iterate over all saved models
        for fold in range(Config.N_FOLDS):
            model_path = os.path.join(self.models_dir, f"pipeline_fold_{fold}.pkl")
            with open(model_path, "rb") as f:
                pipeline = pickle.load(f)

            # Predict on densified test set
            probs_expanded = pipeline.predict_proba(X_test)

            # Reshape and Average Centroids (TTA)
            # We average the predictions of the 3 views for each model
            probs_reshaped = probs_expanded.reshape(n_test, 3, n_classes)
            probs_mean = np.mean(probs_reshaped, axis=1)

            ensemble_probs += probs_mean

        # Average across folds
        ensemble_probs /= Config.N_FOLDS

        # Create Submission DataFrame
        df_sub = pd.DataFrame(ensemble_probs, columns=classes)
        df_sub.insert(0, "id", test_ids)

        # Save to CSV
        Config.make_dirs()  # Ensure submission dir exists
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
