import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import Config


def get_discriminative_solver():
    """
    Constructs the Discriminative Linear Solver (Logistic Regression) with
    internal Cross-Validation and a dense hyperparameter grid.

    Returns:
        sklearn.linear_model.LogisticRegressionCV: The configured estimator.
    """
    # Initialize LogisticRegressionCV with settings from Config
    # We use 'multinomial' explicitly to ensure proper multi-class handling with Log Loss
    clf = LogisticRegressionCV(
        Cs=Config.LR_CS,
        cv=Config.CV_FOLDS,
        scoring=Config.LR_SCORING,
        solver=Config.LR_SOLVER,
        penalty=Config.LR_PENALTY,
        max_iter=Config.LR_MAX_ITER,
        random_state=Config.RANDOM_SEED,
        n_jobs=-1,  # Use all available cores
        multi_class="multinomial",
    )
    return clf


def get_generative_solver():
    """
    Constructs the Generative Linear Solver (LDA) with Ledoit-Wolf shrinkage.

    Returns:
        sklearn.discriminant_analysis.LinearDiscriminantAnalysis: The configured estimator.
    """
    clf = LinearDiscriminantAnalysis(
        solver=Config.LDA_SOLVER, shrinkage=Config.LDA_SHRINKAGE
    )
    return clf


def train_predict(model, X_train, y_train, X_test, model_name="Model"):
    """
    Fits the model on training data and returns probability predictions for the test set.
    Also prints internal CV metrics if available.

    Args:
        model: The sklearn estimator to train.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test features.
        model_name (str): Name of the model for logging purposes.

    Returns:
        np.ndarray: Predicted probabilities for the test set (N_test, N_classes).
    """
    print(
        f"Training {model_name} on {X_train.shape[0]} samples with {X_train.shape[1]} features..."
    )

    # Fit the model
    model.fit(X_train, y_train)

    # Report metrics if available (specifically for LogisticRegressionCV)
    if isinstance(model, LogisticRegressionCV):
        try:
            # scores_ is a dict {class: scores}
            # For 'multinomial', all classes share the same scores grid usually stored under the first class key
            # or sometimes under all keys depending on sklearn version/setup.
            # We aggregate to find the best mean score.

            # Get the scores for one class (in multinomial, the path is usually symmetric)
            first_class = list(model.scores_.keys())[0]
            scores_grid = model.scores_[first_class]  # Shape: (n_folds, n_Cs)

            # Calculate mean score across folds for each C
            mean_scores = np.mean(scores_grid, axis=0)

            # The scoring is 'neg_log_loss', so higher is better (closer to 0)
            best_score = np.max(mean_scores)
            best_c_idx = np.argmax(mean_scores)
            best_c = model.Cs_[best_c_idx]

            # Convert back to positive log loss for reporting
            log_loss_val = -best_score

            print(f"{model_name} - Best Internal CV Log Loss: {log_loss_val}")
            print(f"{model_name} - Best C: {best_c}")

        except Exception as e:
            print(f"Could not extract internal CV scores for {model_name}: {e}")
    else:
        print(f"{model_name} trained successfully (No internal CV scores available).")

    # Predict probabilities
    print(f"Generating predictions for {model_name}...")
    y_pred_proba = model.predict_proba(X_test)

    return y_pred_proba
