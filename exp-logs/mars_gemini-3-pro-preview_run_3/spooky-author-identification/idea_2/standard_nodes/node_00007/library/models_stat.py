import os
import numpy as np
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from library.config import Config
from library.utils import calculate_log_loss, set_seed
from library.data_loader import get_tfidf_vectors


class StatisticalEnsemble(BaseEstimator, ClassifierMixin):
    """
    A weighted ensemble of Logistic Regression and Multinomial Naive Bayes
    designed for the Authorship Attribution task.
    """

    def __init__(self, lr_C=1.0, nb_alpha=0.01, seed=Config.SEED):
        self.lr_C = lr_C
        self.nb_alpha = nb_alpha
        self.seed = seed

        # Base models
        self.lr_model = None
        self.nb_model = None

        # Ensemble weights (LR weight, NB weight)
        self.lr_weight = 0.5
        self.nb_weight = 0.5

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the constituent models and optimizes the ensemble weights
        if validation data is provided.

        Args:
            X_train: Training features (sparse matrix).
            y_train: Training labels (indices).
            X_val: Validation features (sparse matrix), optional.
            y_val: Validation labels (indices), optional.
        """
        # Set seed for reproducibility
        set_seed(self.seed)

        print("Initializing statistical models...")
        # Initialize Logistic Regression
        # 'saga' solver is efficient for large sparse datasets and supports multinomial loss
        self.lr_model = LogisticRegression(
            C=self.lr_C,
            solver="saga",
            multi_class="multinomial",
            max_iter=1000,
            random_state=self.seed,
            n_jobs=-1,
            verbose=0,
        )

        # Initialize Multinomial Naive Bayes
        self.nb_model = MultinomialNB(alpha=self.nb_alpha)

        # Train Naive Bayes
        print("Training Multinomial Naive Bayes...")
        self.nb_model.fit(X_train, y_train)

        # Train Logistic Regression
        print("Training Logistic Regression...")
        self.lr_model.fit(X_train, y_train)

        # Evaluation and Weight Optimization
        if X_val is not None and y_val is not None:
            print("Validating and optimizing ensemble weights...")

            # Get predictions from base models on validation set
            p_val_lr = self.lr_model.predict_proba(X_val)
            p_val_nb = self.nb_model.predict_proba(X_val)

            # Calculate individual losses
            loss_lr = calculate_log_loss(y_val, p_val_lr)
            loss_nb = calculate_log_loss(y_val, p_val_nb)

            print(f"Validation Log Loss - Logistic Regression: {loss_lr}")
            print(f"Validation Log Loss - Naive Bayes: {loss_nb}")

            # Grid search for optimal weight
            # w is the weight for Logistic Regression
            best_loss = float("inf")
            best_w = 0.5

            # Search space: 0.0 to 1.0 with step 0.01
            search_space = np.linspace(0, 1, 101)

            for w in search_space:
                p_blend = w * p_val_lr + (1 - w) * p_val_nb
                loss = calculate_log_loss(y_val, p_blend)

                if loss < best_loss:
                    best_loss = loss
                    best_w = w

            self.lr_weight = best_w
            self.nb_weight = 1.0 - best_w

            print(f"Optimal Weights Found: LR={self.lr_weight}, NB={self.nb_weight}")
            print(f"Best Ensemble Validation Log Loss: {best_loss}")

        else:
            print("No validation data provided. Using default equal weights.")
            self.lr_weight = 0.5
            self.nb_weight = 0.5

        return self

    def predict_proba(self, X):
        """
        Returns the weighted probability estimates.
        """
        if self.lr_model is None or self.nb_model is None:
            raise RuntimeError("Models must be trained before prediction.")

        p_lr = self.lr_model.predict_proba(X)
        p_nb = self.nb_model.predict_proba(X)

        return self.lr_weight * p_lr + self.nb_weight * p_nb

    def predict(self, X):
        """
        Predict class labels for X.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def save(self, path=Config.STATISTICAL_MODEL_SAVE_PATH):
        """
        Saves the trained ensemble object using joblib.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Model saved to {path}")

    @staticmethod
    def load(path=Config.STATISTICAL_MODEL_SAVE_PATH):
        """
        Loads a trained ensemble object.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
        return joblib.load(path)


def train_statistical_branch(train_df, val_df, test_df, load_cached_vectors=True):
    """
    Orchestrates the training of the statistical branch.

    1. Loads/Computes TF-IDF vectors.
    2. Trains the StatisticalEnsemble.
    3. Saves the model.

    Args:
        train_df, val_df, test_df: Pandas DataFrames.
        load_cached_vectors (bool): Whether to use cached TF-IDF matrices.

    Returns:
        model: The trained StatisticalEnsemble.
        X_test: The test feature matrix (useful for inference later).
    """
    # 1. Get Features
    print("Preparing features for statistical branch...")
    X_train, X_val, X_test = get_tfidf_vectors(
        train_df, val_df, test_df, load_cached_data=load_cached_vectors
    )

    # 2. Prepare Labels
    # Map string labels 'EAP', 'HPL', 'MWS' to integers 0, 1, 2
    y_train = train_df["author"].map(Config.LABEL_MAP).values
    y_val = val_df["author"].map(Config.LABEL_MAP).values

    # 3. Train Model
    model = StatisticalEnsemble(seed=Config.SEED)
    model.fit(X_train, y_train, X_val, y_val)

    # 4. Save Model
    model.save()

    return model, X_test
