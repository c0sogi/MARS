import library.config as config
import library.utils as utils
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier


def build_model():
    """
    Constructs a Voting Classifier (LR + MNB) based on configuration.

    Returns:
        sklearn.ensemble.VotingClassifier: The initialized model ensemble.
    """
    # 1. Logistic Regression
    lr_model = LogisticRegression(
        C=config.C_PARAM,
        solver=config.SOLVER,
        multi_class=config.MULTI_CLASS,
        max_iter=config.MAX_ITER,
        tol=config.TOL,
        random_state=config.RANDOM_STATE,
        verbose=0,
        n_jobs=-1,
    )

    # 2. Multinomial Naive Bayes
    # Alpha=0.1 is often better for text than default 1.0
    nb_model = MultinomialNB(alpha=0.1)

    # 3. Voting Classifier
    # Soft voting averages the probabilities
    # We weight LR higher (3) vs NB (1) as LR is generally better calibrated
    model = VotingClassifier(
        estimators=[("lr", lr_model), ("nb", nb_model)],
        voting="soft",
        weights=[3, 1],
        n_jobs=-1,
    )

    return model


def train_model(model, X_train, y_train, X_val=None, y_val=None):
    """
    Trains the model and evaluates on validation set if provided.

    Args:
        model: The initialized model (VotingClassifier).
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
