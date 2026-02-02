import library.config as config
import library.utils as utils
from sklearn.linear_model import LogisticRegression


def build_model():
    """
    Constructs the Logistic Regression model based on configuration.

    Returns:
        sklearn.linear_model.LogisticRegression: The initialized model.
    """
    # Instantiate Logistic Regression with parameters from config
    model = LogisticRegression(
        C=config.C_PARAM,
        solver=config.SOLVER,
        multi_class=config.MULTI_CLASS,
        max_iter=config.MAX_ITER,
        tol=config.TOL,
        random_state=config.RANDOM_STATE,
        verbose=0,  # Ensure silent execution
        n_jobs=-1,  # Utilize available vCPUs
    )
    return model


def train_model(model, X_train, y_train, X_val=None, y_val=None):
    """
    Trains the Logistic Regression model and evaluates on validation set if provided.

    Args:
        model: The initialized LogisticRegression model.
        X_train: Training features (sparse matrix).
        y_train: Training labels (encoded).
        X_val: Validation features (sparse matrix, optional).
        y_val: Validation labels (encoded, optional).

    Returns:
        model: The trained model.
        metrics: Dictionary containing evaluation metrics (if validation data provided).
    """
    print("Fitting model...")
    # Fit the model to the training data
    # The 'tol' parameter in build_model acts as the stopping criterion (Early Stopping)
    model.fit(X_train, y_train)

    metrics = {}

    # Evaluate on validation data if provided
    if X_val is not None and y_val is not None:
        print("Predicting on validation set...")
        # Predict probabilities for the validation set
        y_pred_proba = model.predict_proba(X_val)

        # Calculate Log Loss using the utility function
        # This function handles the specific clipping and rescaling rules
        loss = utils.calculate_log_loss(y_val, y_pred_proba)

        # Print the metric with full precision
        print(f"Validation Log Loss: {loss}")
        metrics["log_loss"] = loss

    return model, metrics
