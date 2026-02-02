import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed
from library.features import get_feature_pipeline


def get_model(C=1.0, class_weight=None):
    """
    Constructs the Bagged Logistic Regression model architecture.

    This function creates a BaggingClassifier ensemble with a LogisticRegression
    base estimator. This combination provides the high bias/low variance properties
    needed for the noisy text/metadata inputs while stabilizing predictions via bagging.

    Args:
        C (float): Inverse of regularization strength for the LogisticRegression base learner.
                   Smaller values specify stronger regularization.
        class_weight (dict or str): Weights associated with classes in the form {class_label: weight}.
                                    If 'balanced', uses the values of y to automatically adjust weights.

    Returns:
        sklearn.ensemble.BaggingClassifier: The initialized ensemble model.
    """
    set_seed(Config.SEED)

    # Initialize the linear base learner
    # 'liblinear' is chosen for its efficiency on small-to-medium datasets and support for L2 penalty
    base_estimator = LogisticRegression(
        C=C,
        class_weight=class_weight,
        solver="liblinear",
        penalty="l2",
        max_iter=1000,
        random_state=Config.SEED,
    )

    # Initialize the Bagging ensemble
    # This wraps the linear model to reduce variance and improve stability
    model = BaggingClassifier(
        estimator=base_estimator,
        n_estimators=Config.N_BAGGING_ESTIMATORS,
        n_jobs=-1,  # Utilize all available vCPUs for parallel fitting of base estimators
        random_state=Config.SEED,
    )

    return model


def tune_hyperparameters(X, y):
    """
    Performs a Grid Search to optimize the hyperparameters of the Logistic Regression base learner.

    This function constructs a full pipeline including the feature engineering steps
    (SBERT, TF-IDF + Chi2, Metadata Scaling) to ensure that supervised feature selection
    occurs within the cross-validation folds, preventing data leakage.

    Args:
        X (pd.DataFrame): The raw training features.
        y (pd.Series or np.array): The training target labels.

    Returns:
        dict: A dictionary containing the best hyperparameters ('C' and 'class_weight').
    """
    set_seed(Config.SEED)

    print("Constructing tuning pipeline...")

    # 1. Retrieve the feature engineering pipeline
    # This transformer handles heterogeneous data (Text, Lists, Metadata)
    preprocessor = get_feature_pipeline()

    # 2. Initialize the model with default placeholders
    # The actual parameters will be injected by GridSearchCV
    model = get_model()

    # 3. Create the end-to-end pipeline
    # Pipeline: Raw Data -> Feature Engineering -> Bagged Classifier
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])

    # 4. Define the hyperparameter grid
    # We target the 'estimator' (LogisticRegression) inside the 'classifier' (BaggingClassifier)
    # Note: scikit-learn >= 1.2 uses 'estimator' param for BaggingClassifier
    param_grid = {
        "classifier__estimator__C": Config.C_GRID,
        "classifier__estimator__class_weight": Config.CLASS_WEIGHT_GRID,
    }

    # 5. Configure Cross-Validation
    # StratifiedKFold ensures class distribution is preserved in each fold
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=Config.SEED)

    print(
        f"Starting Grid Search over {len(Config.C_GRID) * len(Config.CLASS_WEIGHT_GRID)} parameter combinations..."
    )

    # 6. Execute Grid Search
    # n_jobs=1 is used here to avoid conflicts with GPU usage in SBERTTransformer
    # Parallelism is handled inside BaggingClassifier (n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,
        verbose=1,
    )

    grid_search.fit(X, y)

    print("Grid Search Complete.")
    print(f"Best ROC AUC Score: {grid_search.best_score_}")
    print(f"Best Parameters: {grid_search.best_params_}")

    # Extract the clean parameter dictionary for the base estimator
    best_params = {
        "C": grid_search.best_params_["classifier__estimator__C"],
        "class_weight": grid_search.best_params_["classifier__estimator__class_weight"],
    }

    return best_params
