import numpy as np
from library.model_definitions import get_logistic_regression, get_lda
from library.feature_pipeline import process_data
from library.utils import save_submission


class HybridEnsemble:
    """
    A Linear Hybrid Ensemble combining:
    1. Logistic Regression (Discriminative Linear) - Uses Scaled Features
    2. Linear Discriminant Analysis (Generative Linear) - Uses Scaled Features
    """

    def __init__(self, random_state=42, n_jobs=-1):
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Initialize models using factory functions
        self.lr = get_logistic_regression(random_state=random_state, n_jobs=n_jobs)
        self.lda = get_lda()

    def fit(self, X_linear, y):
        """
        Trains the component models on their respective feature views.

        Args:
            X_linear: Globally scaled features for Linear Models (LR, LDA).
            y: Target labels.
        """
        # 1. Train Logistic Regression
        print("Training Logistic Regression (Discriminative Linear)...")
        self.lr.fit(X_linear, y)
        lr_acc = self.lr.score(X_linear, y)
        print(f"LR Training Accuracy: {lr_acc}")

        # 2. Train LDA
        print("Training LDA (Generative Linear)...")
        self.lda.fit(X_linear, y)
        lda_acc = self.lda.score(X_linear, y)
        print(f"LDA Training Accuracy: {lda_acc}")

        return self

    def predict_proba(self, X_linear):
        """
        Predicts probabilities using Soft Voting.

        Args:
            X_linear: Globally scaled features.

        Returns:
            Averaged probability matrix.
        """
        print("Predicting with Logistic Regression...")
        probs_lr = self.lr.predict_proba(X_linear)

        print("Predicting with LDA...")
        probs_lda = self.lda.predict_proba(X_linear)

        print("Calculating Ensemble Predictions (Soft Vote)...")
        # Average the probabilities
        probs_ensemble = (probs_lr + probs_lda) / 2.0

        return probs_ensemble


def run_pipeline(
    metadata_dir="./metadata",
    cache_dir="./working/idea_3",
    submission_path="./submission/submission.csv",
    random_state=42,
):
    """
    Orchestrates the full pipeline: Data Loading -> Training -> Inference -> Submission.
    """
    print("Initializing Pipeline...")

    # 1. Load and Process Data
    # This handles scaling and caching
    (
        X_train_scaled,
        y_train,
        X_test_scaled,
        test_ids,
        classes,
    ) = process_data(
        metadata_dir=metadata_dir,
        cache_dir=cache_dir,
        load_cached_data=True,
        random_state=random_state,
    )

    # 2. Initialize Ensemble
    ensemble = HybridEnsemble(random_state=random_state, n_jobs=-1)

    # 3. Train Ensemble
    # We use the full training set (Train + Val) for maximum sample efficiency
    ensemble.fit(X_train_scaled, y_train)

    # 4. Generate Predictions
    probs = ensemble.predict_proba(X_test_scaled)

    # 5. Save Submission
    save_submission(test_ids, classes, probs, submission_path)

    print("Pipeline completed successfully.")
