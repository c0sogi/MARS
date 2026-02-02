import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from library.config import (
    RANDOM_SEED,
    N_FOLDS,
    SCORING_METRIC,
    LR_CS,
    LDA_SOLVER,
    LDA_SHRINKAGE,
)


def get_logistic_cv():
    """
    Creates and configures a LogisticRegressionCV estimator.

    Returns:
        LogisticRegressionCV: Configured estimator with L2 regularization and multinomial loss.
    """
    return LogisticRegressionCV(
        Cs=LR_CS,
        cv=N_FOLDS,
        scoring=SCORING_METRIC,
        solver="lbfgs",
        multi_class="multinomial",
        penalty="l2",
        random_state=RANDOM_SEED,
        max_iter=5000,
        n_jobs=-1,
    )


def get_lda():
    """
    Creates and configures a LinearDiscriminantAnalysis estimator.

    Returns:
        LinearDiscriminantAnalysis: Configured estimator with Ledoit-Wolf shrinkage.
    """
    return LinearDiscriminantAnalysis(solver=LDA_SOLVER, shrinkage=LDA_SHRINKAGE)


def train_and_evaluate(model, X, y, model_name="Model"):
    """
    Trains the model on the provided data and evaluates it.

    For LogisticRegressionCV, it fits the model and extracts the internal CV score
    to avoid redundant computation.
    For LDA, it runs cross_val_score first to get the metric, then fits on the full data.

    Args:
        model: The estimator instance.
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        model_name (str): Name for logging.

    Returns:
        tuple: (fitted_model, best_cv_score)
    """
    best_score = None

    # Logic for LogisticRegressionCV (utilize internal CV results)
    if isinstance(model, LogisticRegressionCV):
        print(f"Training {model_name} (LogisticRegressionCV)...")
        model.fit(X, y)

        # Extract internal CV scores
        # scores_ is a dict {class_label: array of shape (n_folds, n_cs)}
        # For multi_class='multinomial', the scores are global (same for all keys).
        # We take the values from the first key found.
        first_key = next(iter(model.scores_))
        scores_matrix = model.scores_[first_key]  # Shape (n_folds, n_cs)

        # Calculate mean score across folds for each C
        mean_scores_per_c = np.mean(scores_matrix, axis=0)

        # The model selects the best C based on the max score (since metric is neg_log_loss)
        best_score = np.max(mean_scores_per_c)

    # Logic for LDA (explicit CV required)
    else:
        print(f"Evaluating {model_name} (LDA) with CV...")
        # Compute CV score explicitly
        cv_scores = cross_val_score(
            model, X, y, cv=N_FOLDS, scoring=SCORING_METRIC, n_jobs=-1
        )
        best_score = np.mean(cv_scores)

        print(f"Training {model_name} (LDA) on full data...")
        model.fit(X, y)

    # Print score with full precision
    print(f"{model_name} {SCORING_METRIC}: {best_score}")

    return model, best_score
