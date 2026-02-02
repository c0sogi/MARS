import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import log_loss

from library.config import Config
from library.data_loader import LeafDataLoader
from library.preprocessing import FeatureScaler


class LeafEnsemble:
    """
    Ensemble of Logistic Regression and LDA.
    Combines discriminative and generative linear models to improve robustness on small data.
    """

    def __init__(self, params=None):
        """
        Initialize the model with hyperparameters.

        Args:
            params (dict, optional): Hyperparameters for LogisticRegressionCV.
                                     Defaults to Config.MODEL_PARAMS.
        """
        self.params = params if params is not None else Config.MODEL_PARAMS

        # Estimator 1: Logistic Regression CV (Cite Lesson 00004)
        # Dynamic regularization tuning proved effective.
        self.lr = LogisticRegressionCV(**self.params)

        # Estimator 2: LDA with shrinkage (Cite Lesson 00001 extension)
        # LDA is effective for small sample sizes. Shrinkage handles high dimensionality.
        self.lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

        # Estimator 3: KNN (Instance-based)
        # Captures local manifold structure that global linear models might miss.
        # Manhattan distance is often preferred in high-dimensional spaces.
        self.knn = KNeighborsClassifier(
            n_neighbors=5, weights="distance", metric="manhattan", n_jobs=-1
        )

        # Soft Voting Ensemble (Cite Lesson 00006)
        # Combining Discriminative (LR), Generative (LDA), and Instance-based (KNN) models.
        self.model = VotingClassifier(
            estimators=[("lr", self.lr), ("lda", self.lda), ("knn", self.knn)],
            voting="soft",
            n_jobs=None,
        )
        self.classes_ = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the ensemble model and evaluates on validation set.
        """
        print("Training Ensemble (LR + LDA)...")
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_

        # Report best C found for LR component
        try:
            lr_fitted = self.model.named_estimators_["lr"]
            if hasattr(lr_fitted, "C_"):
                print(f"LR Best C (mean): {np.mean(lr_fitted.C_)}")
        except Exception as e:
            print(f"Could not retrieve LR stats: {e}")

        if X_val is not None and y_val is not None:
            print("Evaluating on validation set...")
            y_pred_proba = self.model.predict_proba(X_val)

            # Calculate Log Loss
            val_loss = log_loss(y_val, y_pred_proba, labels=self.classes_)
            print(f"Validation Multi-class Log Loss: {val_loss}")

    def predict(self, X):
        """
        Predict class probabilities for input features.

        Args:
            X (np.ndarray): Features.

        Returns:
            np.ndarray: Predicted probabilities of shape (n_samples, n_classes).
        """
        return self.model.predict_proba(X)

    def save(self, filepath):
        """
        Saves the trained model to disk.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """
        Loads a trained model from disk.
        """
        self.model = joblib.load(filepath)
        self.classes_ = self.model.classes_
        print(f"Model loaded from {filepath}")


def generate_submission(model, X_test, test_ids, encoder, output_path):
    """
    Generates the submission CSV file.

    Args:
        model (LogisticBaseline): Trained model instance.
        X_test (np.ndarray): Test features.
        test_ids (np.ndarray): Test sample IDs.
        encoder (LabelEncoder): Fitted LabelEncoder to retrieve class names.
        output_path (str): Path to save the submission CSV.
    """
    print("Generating submission...")

    # Get probabilities
    probs = model.predict(X_test)

    # Clip probabilities to avoid log loss extremes (as per task description, though scoring does it too)
    # max(min(p, 1-10^-15), 10^-15)
    probs = np.clip(probs, 1e-15, 1 - 1e-15)

    # Create DataFrame
    # Columns must be the species names
    species_names = encoder.classes_

    # Ensure the order of columns matches the model's classes
    # The model was trained on encoded integers 0..N-1 which correspond to encoder.classes_ indices
    # So model.predict_proba returns columns in order 0, 1, 2... which matches species_names order.

    submission_df = pd.DataFrame(probs, columns=species_names)
    submission_df.insert(0, "id", test_ids)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run(load_cached_data=True):
    """
    Orchestrates the full training and submission pipeline.
    """
    # 1. Load Data
    loader = LeafDataLoader()
    data = loader.load_data(load_cached_data=load_cached_data)

    X_train, y_train, train_ids = data["train"]
    X_val, y_val, val_ids = data["val"]
    X_test, test_ids = data["test"]
    encoder = data["encoder"]

    # 2. Scale Features
    scaler = FeatureScaler()
    X_train_scaled, X_val_scaled, X_test_scaled = scaler.scale_features(
        X_train, X_val, X_test, load_cached_data=load_cached_data
    )

    # 3. Initialize and Train Model
    model = LogisticBaseline(params=Config.MODEL_PARAMS)
    model.train(X_train_scaled, y_train, X_val_scaled, y_val)

    # 4. Save Model (Optional, for persistence)
    model_path = os.path.join(Config.WORKING_DIR, "logistic_model.joblib")
    model.save(model_path)

    # 5. Generate Submission
    generate_submission(model, X_test_scaled, test_ids, encoder, Config.SUBMISSION_PATH)
