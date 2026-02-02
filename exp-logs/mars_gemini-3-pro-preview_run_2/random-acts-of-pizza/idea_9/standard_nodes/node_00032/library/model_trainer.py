import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score
from library.utils import set_seed
from library.preprocessor import DataPreprocessor


class EnsembleTrainer:
    """
    Manages the training and tuning of a CV-Bagged Linear Ensemble.
    Cite Lesson 27: Prefer Fold Averaging (CV-Bagging) over Single-Model Retraining.
    """

    def __init__(self, random_state=42, n_splits=5):
        self.random_state = random_state
        self.n_splits = n_splits
        self.models = []
        self.best_c = None
        self.best_cv_score = -1.0

    def get_base_estimator(self, c_value):
        """Creates the base Logistic Regression estimator."""
        return LogisticRegression(
            C=c_value,
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=self.random_state,
        )

    def get_ensemble(self, base_estimator):
        """
        Wraps the base estimator in a BaggingClassifier.
        Cite Lesson 22: Reduce bagging estimators to 10 and constrain complexity.
        """
        return BaggingClassifier(
            estimator=base_estimator,
            n_estimators=10,  # Reduced from 20
            max_samples=0.8,
            max_features=1.0,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def tune_and_train(self, X_train, y_train, X_val=None, y_val=None, c_grid=None):
        """
        1. Performs CV Grid Search on X_train to find optimal C.
        2. Trains an ensemble of models (one per fold) using the best C.
        """
        if c_grid is None:
            # Cite Lesson 22: Constrain hyperparameter search space (remove 100.0)
            c_grid = [1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0]

        print(f"Starting CV grid search over C values: {c_grid}")

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        for c in c_grid:
            base_est = self.get_base_estimator(c)
            model = self.get_ensemble(base_est)

            # Compute cross-validation score (AUC)
            scores = cross_val_score(
                model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1
            )
            mean_auc = np.mean(scores)

            print(f"  C={c}: Mean CV AUC = {mean_auc:.4f}")

            if mean_auc > self.best_cv_score:
                self.best_cv_score = mean_auc
                self.best_c = c

        print(
            f"Grid search complete. Best C: {self.best_c} with CV AUC: {self.best_cv_score:.4f}"
        )

        # Train Final CV-Ensemble
        print("Training final CV-Bagging ensemble...")
        self.models = []

        # Iterate over folds and train a model on each training split
        for fold_idx, (train_idx, _) in enumerate(skf.split(X_train, y_train)):
            X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]

            base_est = self.get_base_estimator(self.best_c)
            model = self.get_ensemble(base_est)
            model.fit(X_fold_train, y_fold_train)
            self.models.append(model)

        print(f"Trained {len(self.models)} models on {self.n_splits} folds.")

    def predict(self, X):
        """Generates probabilities by averaging predictions from all fold models."""
        if not self.models:
            raise RuntimeError("Models have not been trained yet.")

        # Average predictions
        preds = np.zeros(X.shape[0])
        for model in self.models:
            preds += model.predict_proba(X)[:, 1]

        return preds / len(self.models)

    def save_model(self, filepath):
        """Saves the list of trained models."""
        if self.models:
            joblib.dump(self.models, filepath)
            print(f"Ensemble saved to {filepath}")


def run_training_pipeline(load_cached_data=True, debug_sample_size=None):
    """
    Orchestrates the full training pipeline:
    1. Load/Assemble Data (Text + KNN + Meta)
    2. Tune and Train Model
    3. Generate Submission
    """
    # 1. Setup
    set_seed(42)
    working_dir = "./working/idea_9"
    os.makedirs(working_dir, exist_ok=True)
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    # 2. Data Loading & Preprocessing
    print("Initializing Data Preprocessor...")
    preprocessor = DataPreprocessor(k_neighbors=50)

    # This handles feature extraction, KNN generation, and concatenation
    X_train, y_train, X_val, y_val, X_test, test_ids = (
        preprocessor.process_and_load_data(
            load_cached_data=load_cached_data, debug_sample_size=debug_sample_size
        )
    )

    print(f"Data Loaded. Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # 3. Model Training & Tuning
    trainer = EnsembleTrainer(random_state=42)
    trainer.tune_and_train(X_train, y_train, X_val, y_val)

    # Save the best model
    model_path = os.path.join(working_dir, "ensemble_model.joblib")
    trainer.save_model(model_path)

    # 4. Submission Generation
    print("Generating predictions for test set...")
    y_test_pred = trainer.predict(X_test)

    submission_df = pd.DataFrame(
        {"request_id": test_ids, "requester_received_pizza": y_test_pred}
    )

    submission_path = os.path.join(submission_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    # Print final validation metric for reference
    print(f"Final Validation AUC: {trainer.best_auc}")
