import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from library.config import (
    LR_SOLVER,
    LR_PENALTY,
    LR_MAX_ITER,
    LR_C_GRID,
    CV_FOLDS,
    N_BAGGING_ESTIMATORS,
    BAGGING_MAX_FEATURES,
    BAGGING_BOOTSTRAP,
    BAGGING_N_JOBS,
    LDA_SOLVER,
    LDA_SHRINKAGE,
    RANDOM_SEED,
)


def tune_logistic_regression(X_train, y_train):
    """
    Tunes the regularization strength (C) for Logistic Regression using Cross-Validation.

    This function utilizes LogisticRegressionCV with a constrained grid to find the
    hyperparameter that minimizes log loss. It enforces a multinomial fit to ensure
    a single global C is selected.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.

    Returns:
        float: The optimal C value found.
    """
    print(f"Tuning Logistic Regression with Grid: {LR_C_GRID} and CV={CV_FOLDS}...")

    # Initialize LogisticRegressionCV
    # We use 'neg_log_loss' scoring to align directly with the competition metric.
    # multi_class='multinomial' ensures we optimize the cross-entropy loss directly.
    clf = LogisticRegressionCV(
        Cs=LR_C_GRID,
        cv=CV_FOLDS,
        solver=LR_SOLVER,
        penalty=LR_PENALTY,
        max_iter=LR_MAX_ITER,
        multi_class="multinomial",
        scoring="neg_log_loss",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=0,
    )

    clf.fit(X_train, y_train)

    # Extract optimal C
    # In multinomial mode, C_ is an array of shape (1,) containing the best C
    optimal_c = clf.C_[0]

    # Extract best score for logging purposes
    # scores_ is a dict {class_label: scores_array}
    # For multinomial with a global metric, values are identical across keys.
    first_class = list(clf.scores_.keys())[0]
    scores_array = clf.scores_[first_class]  # shape (n_folds, n_Cs)
    mean_scores = scores_array.mean(axis=0)
    best_score = mean_scores.max()

    print(f"  Best C: {optimal_c}")
    # best_score is negative log loss, so we negate it to print the actual log loss
    print(f"  Best CV Log Loss: {-best_score:.10f}")

    return optimal_c


def get_hybrid_ensemble_components(optimal_c):
    """
    Constructs the LDA and Bagged Logistic Regression models.

    Args:
        optimal_c (float): The optimal regularization strength for Logistic Regression
                           identified during the tuning phase.

    Returns:
        tuple: (lda_model, bagging_model)
            - lda_model: The LinearDiscriminantAnalysis instance (Generative Branch).
            - bagging_model: The BaggingClassifier instance wrapping LogisticRegression (Discriminative Branch).
    """
    print(f"Initializing Hybrid Ensemble components with C={optimal_c}...")

    # 1. Generative Branch: Linear Discriminant Analysis (Anchor)
    # Uses Ledoit-Wolf shrinkage to handle high-dimensional, low-sample data effectively.
    lda_model = LinearDiscriminantAnalysis(solver=LDA_SOLVER, shrinkage=LDA_SHRINKAGE)

    # 2. Discriminative Branch: Bagged Logistic Regression (Innovation)
    # Base estimator: Logistic Regression with the tuned C
    base_lr = LogisticRegression(
        C=optimal_c,
        solver=LR_SOLVER,
        penalty=LR_PENALTY,
        max_iter=LR_MAX_ITER,
        multi_class="multinomial",
        n_jobs=1,  # Base estimator runs on 1 core; Bagging handles parallelism
        random_state=RANDOM_SEED,
    )

    # Bagging Wrapper
    # Ensembles multiple linear models trained on random feature subsets (max_features=0.6)
    # to reduce variance and mitigate the curse of dimensionality.
    bagging_model = BaggingClassifier(
        estimator=base_lr,
        n_estimators=N_BAGGING_ESTIMATORS,
        max_features=BAGGING_MAX_FEATURES,
        bootstrap=BAGGING_BOOTSTRAP,
        n_jobs=BAGGING_N_JOBS,
        random_state=RANDOM_SEED,
    )

    return lda_model, bagging_model
