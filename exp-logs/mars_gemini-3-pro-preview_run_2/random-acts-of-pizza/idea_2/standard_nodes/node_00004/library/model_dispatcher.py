from sklearn.linear_model import LogisticRegression
from library.config import SEED


def get_logistic_regression(
    C: float = 1.0, solver: str = "lbfgs", max_iter: int = 1000
) -> LogisticRegression:
    """
    Initializes a Logistic Regression model with 'balanced' class weights and L2 regularization.

    This function encapsulates the model definition, ensuring that the specific
    architectural choices (L2 penalty, class weighting, random seed) are consistently
    applied while allowing hyperparameters like 'C' to be tuned externally.

    Args:
        C (float): Inverse of regularization strength; must be a positive float.
                   Smaller values specify stronger regularization. Defaults to 1.0.
        solver (str): Algorithm to use in the optimization problem.
                      Common options: 'lbfgs', 'liblinear'. Defaults to 'lbfgs'.
        max_iter (int): Maximum number of iterations taken for the solvers to converge.
                        Defaults to 1000.

    Returns:
        LogisticRegression: An initialized sklearn LogisticRegression model instance.
    """
    # Initialize the model
    # class_weight='balanced' automatically adjusts weights inversely proportional to class frequencies
    # random_state=SEED ensures reproducibility across runs
    # n_jobs=-1 allows parallel processing if the solver supports it (e.g. for One-vs-Rest, though binary LBFGS is usually sequential)
    model = LogisticRegression(
        C=C,
        class_weight="balanced",
        solver=solver,
        penalty="l2",
        max_iter=max_iter,
        random_state=SEED,
        n_jobs=-1,
    )

    return model
