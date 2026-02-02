import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from library.config import Config
from library.utils import set_seed


def build_bagged_logistic_regression(
    C: float = 1.0,
    penalty: str = "l2",
    solver: str = "liblinear",
    class_weight: str = None,
    bagging_params: dict = None,
    random_state: int = Config.RANDOM_SEED,
) -> BaggingClassifier:
    """
    Constructs a Bagged Logistic Regression model.
    Used for both the Text Expert (View A) and Metadata Expert (View B).

    Args:
        C (float): Inverse of regularization strength; smaller values specify stronger regularization.
        penalty (str): Norm used in the penalization ('l1', 'l2', etc.).
        solver (str): Algorithm to use in the optimization problem.
        class_weight (str or dict): Weights associated with classes ('balanced' or None).
        bagging_params (dict): Dictionary containing parameters for BaggingClassifier.
                               Defaults to Config.BAGGING_PARAMS if None.
        random_state (int): Seed used by the random number generator.

    Returns:
        BaggingClassifier: The configured ensemble model.
    """
    # Ensure reproducibility globally before model creation
    set_seed(random_state)

    # Use default bagging parameters if not provided
    if bagging_params is None:
        bagging_params = Config.BAGGING_PARAMS.copy()
    else:
        # Create a copy to avoid modifying the input dictionary
        bagging_params = bagging_params.copy()

    # Create the base estimator
    # Note: We set random_state on the base estimator as well, though Bagging
    # controls the data sampling. Some solvers (e.g., liblinear) use RNG.
    base_estimator = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        class_weight=class_weight,
        random_state=random_state,
        max_iter=1000,  # Increased max_iter to ensure convergence
    )

    # Extract bagging specific parameters to pass as kwargs
    # We remove 'random_state' from params if it exists to avoid conflict with the explicit arg
    if "random_state" in bagging_params:
        del bagging_params["random_state"]

    # Instantiate BaggingClassifier
    # We use 'estimator' which is the standard parameter name in recent sklearn versions.
    # If using an older version where 'estimator' is not available, 'base_estimator' would be used,
    # but 'estimator' is preferred for forward compatibility.
    model = BaggingClassifier(
        estimator=base_estimator, random_state=random_state, **bagging_params
    )

    return model


def build_meta_learner(
    C: float = 1.0,
    penalty: str = "l2",
    solver: str = "lbfgs",
    random_state: int = Config.RANDOM_SEED,
) -> LogisticRegression:
    """
    Constructs the Meta-Learner for the stacking layer.
    This model learns to weight the probabilities from the base experts.

    Args:
        C (float): Inverse of regularization strength.
        penalty (str): Norm used in the penalization.
        solver (str): Algorithm to use in the optimization problem.
        random_state (int): Seed used by the random number generator.

    Returns:
        LogisticRegression: The configured meta-learner.
    """
    # Ensure reproducibility
    set_seed(random_state)

    # Instantiate the Logistic Regression model
    # We enforce non-negative weights implicitly via the nature of the problem
    # (probabilities are positive), but standard LR allows negative weights.
    # In a strict stacking setup, one might use NonNegative Least Squares,
    # but a standard LR with L2 regularization is robust enough here.
    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        random_state=random_state,
        max_iter=1000,  # Increased max_iter to ensure convergence
    )

    return model
