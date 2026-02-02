import os
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import log_loss
from library import config
from library import preprocessing

# Set fixed random seed for reproducibility
np.random.seed(config.SEED)


class EmpiricalPriorLDA:
    """
    Stabilized Linear Discriminant Analysis with Empirical Priors.

    Cite solution_lesson_node_00033: Enforcing uniform priors degraded performance
    compared to using empirical priors on stratified splits. We revert to using
    observed class frequencies.

    Cite solution_lesson_node_00029: The 'lsqr' solver with auto shrinkage
    provided the best historical performance.
    """

    def __init__(self):
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Fits the LDA model with empirical priors.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.

        Returns:
            self
        """
        # Identify unique classes from the target vector
        self.classes_ = np.unique(y)

        # Initialize LDA with specific solver and default (empirical) priors
        # solver='lsqr': Least squares solution, supports shrinkage.
        # shrinkage='auto': Applies Ledoit-Wolf shrinkage to the covariance matrix.
        self.model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X (np.ndarray): Features.

        Returns:
            np.ndarray: Probability matrix of shape (n_samples, n_classes).
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before calling predict_proba.")
        return self.model.predict_proba(X)


def clip_probabilities(probas):
    """
    Clips probabilities to avoid log loss extremes as per task description.
    Formula: max(min(p, 1-10^-15), 10^-15)

    Args:
        probas (np.ndarray): Input probabilities.

    Returns:
        np.ndarray: Clipped probabilities.
    """
    return np.clip(probas, config.EPSILON, 1.0 - config.EPSILON)


def train_and_evaluate(load_cached_data=True):
    """
    Trains the UniformPriorLDA model on the training set and evaluates on the validation set.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.

    Returns:
        tuple: (model, pipeline, val_loss)
    """
    print("Initializing Preprocessing Pipeline...")
    # 1. Get Fitted Pipeline (fits on train data)
    pipeline = preprocessing.get_fitted_pipeline(load_cached_data=load_cached_data)

    print("Loading and Transforming Data...")
    # 2. Get Transformed Data
    # We use the 'train' split from metadata for training
    X_train, y_train, _ = preprocessing.get_transformed_data(
        "train", pipeline, load_cached_data=load_cached_data
    )
    # We use the 'val' split from metadata for evaluation
    X_val, y_val, _ = preprocessing.get_transformed_data(
        "val", pipeline, load_cached_data=load_cached_data
    )

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Validation Data Shape: {X_val.shape}")

    # 3. Initialize and Fit Model
    print("Training EmpiricalPriorLDA Model...")
    model = EmpiricalPriorLDA()
    model.fit(X_train, y_train)

    # 4. Evaluate
    print("Evaluating on Validation Set...")
    val_probas = model.predict_proba(X_val)

    # Clip probabilities before scoring to match submission logic and metric definition
    val_probas_clipped = clip_probabilities(val_probas)

    # Calculate Log Loss
    val_loss = log_loss(y_val, val_probas_clipped, labels=model.classes_)

    # Print full precision as requested
    print("Validation Log Loss:", val_loss)

    return model, pipeline, val_loss


def generate_submission(model, pipeline, load_cached_data=True):
    """
    Generates the submission file for the test set.

    Args:
        model (EmpiricalPriorLDA): Trained model.
        pipeline (GaussianPipeline): Fitted preprocessing pipeline.
        load_cached_data (bool): Whether to use cached preprocessed data.
    """
    print("Generating Submission...")

    # 1. Load Test Data
    X_test, _, ids_test = preprocessing.get_transformed_data(
        "test", pipeline, load_cached_data=load_cached_data
    )

    # 2. Predict
    probas = model.predict_proba(X_test)

    # 3. Clip Probabilities
    probas = clip_probabilities(probas)

    # 4. Create DataFrame
    # Columns must be the class names in the order the model learned them.
    # sklearn's LDA sorts classes_ automatically (usually alphanumeric for strings).
    df_sub = pd.DataFrame(probas, columns=model.classes_)

    # Insert ID column at the beginning
    df_sub.insert(0, config.ID_COL, ids_test)

    # 5. Save
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(config.SUBMISSION_FILE_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_FILE_PATH}")
