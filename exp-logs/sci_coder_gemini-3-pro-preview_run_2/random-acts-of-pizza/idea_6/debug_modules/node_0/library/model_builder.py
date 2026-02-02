import copy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from library.config import LR_BASE_PARAMS, BAGGING_PARAMS


def create_bagged_linear_model(C=1.0):
    """
    Constructs a Bagging Ensemble of Logistic Regression classifiers.

    The architecture consists of:
    1. Base Learner: Logistic Regression with L2 regularization and balanced class weights.
       This provides a robust linear decision boundary in the high-dimensional space.
    2. Ensemble: BaggingClassifier with bootstrap sampling and feature subsampling.
       This reduces variance and prevents overfitting to specific embedding artifacts.

    Args:
        C (float): Inverse of regularization strength for the base Logistic Regression.
                   Smaller values specify stronger regularization. Defaults to 1.0.

    Returns:
        BaggingClassifier: The configured ensemble model ready for training.
    """
    # Deep copy configuration dictionaries to ensure isolation between model instances
    lr_params = copy.deepcopy(LR_BASE_PARAMS)
    bagging_params = copy.deepcopy(BAGGING_PARAMS)

    # Set the specific regularization strength for this instance
    lr_params["C"] = C

    # Initialize the base estimator
    # LR_BASE_PARAMS includes: penalty='l2', class_weight='balanced', solver='liblinear'
    base_estimator = LogisticRegression(**lr_params)

    # Initialize the Bagging Classifier
    # BAGGING_PARAMS includes: n_estimators=100, max_samples=0.8, max_features=0.8
    # We use 'estimator' as the parameter name for the base model (standard in sklearn >= 1.2)
    model = BaggingClassifier(estimator=base_estimator, **bagging_params)

    return model
