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
    Manages the training and tuning of a Neighborhood-Augmented Linear Ensemble.
    The model consists of a BaggingClassifier wrapping a LogisticRegression base estimator.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_model = None
        self.best_c = None
        self.best_auc = -1.0

    def get_base_estimator(self, c_value):
        """Creates the base Logistic Regression estimator with specific regularization."""
        return LogisticRegression(
            C=c_value,
            class_weight="balanced",
            solver="liblinear",  # Good for smaller datasets and strong regularization
            max_iter=1000,
            random_state=self.random_state,
        )

    def get_ensemble(self, base_estimator, n_estimators=20):
        """Wraps the base estimator in a BaggingClassifier."""
        return BaggingClassifier(
            estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=0.8,
            max_features=1.0,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def tune_and_train(self, X_train, y_train, X_val, y_val, c_grid=None):
        """
        Performs a grid search for the optimal C parameter using the validation set.
        Trains the ensemble on X_train and evaluates on X_val.
        """
        if c_grid is None:
            # High regularization regime as per design
            c_grid = [1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0]

        print(f"Starting grid search over C values: {c_grid}")

        for c in c_grid:
            # 1. Configure Model
            base_est = self.get_base_estimator(c)
            model = self.get_ensemble(base_est)

            # 2. Train
            model.fit(X_train, y_train)

            # 3. Evaluate
            # Predict probabilities for the positive class
            y_val_pred = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_val_pred)

            print(f"  C={c}: Validation AUC = {auc}")

            # 4. Track Best
            if auc > self.best_auc:
                self.best_auc = auc
                self.best_c = c
                self.best_model = model

        print(f"Grid search complete. Best C: {self.best_c} with AUC: {self.best_auc}")
        return self.best_model

    def predict(self, X):
        """Generates probabilities for the positive class."""
        if self.best_model is None:
            raise RuntimeError("Model has not been trained yet.")
        return self.best_model.predict_proba(X)[:, 1]

    def save_model(self, filepath):
        """Saves the trained model to disk."""
        if self.best_model is not None:
            joblib.dump(self.best_model, filepath)
            print(f"Model saved to {filepath}")


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
