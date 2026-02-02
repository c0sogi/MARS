import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import check_random_state
from library import config, utils, preprocessing


class StratifiedBaggedLDA(BaseEstimator, ClassifierMixin):
    """
    A Bagging ensemble of Linear Discriminant Analysis models using Stratified Bootstrap.

    This ensures that every bootstrap sample contains at least one instance of every class,
    which is critical for datasets with many classes and few samples per class.
    """

    def __init__(
        self,
        n_estimators=config.N_ESTIMATORS,
        solver=config.LDA_SOLVER,
        shrinkage=config.LDA_SHRINKAGE,
        random_state=config.SEED,
    ):
        self.n_estimators = n_estimators
        self.solver = solver
        self.shrinkage = shrinkage
        self.random_state = random_state
        self.estimators_ = []
        self.classes_ = None
        self.n_classes_ = 0

    def fit(self, X, y):
        """
        Fit the ensemble on training data using stratified bootstrap resampling.
        """
        # Ensure reproducibility
        rng = check_random_state(self.random_state)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.estimators_ = []

        # Define the base estimator
        base_estimator = LinearDiscriminantAnalysis(
            solver=self.solver, shrinkage=self.shrinkage
        )

        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        for i in range(self.n_estimators):
            # Stratified Bootstrap Resampling
            resampled_indices = []

            for cls in self.classes_:
                # Get indices for the current class
                cls_indices = indices[y == cls]

                # Sample with replacement (size = number of samples in this class)
                # This maintains the class balance ratio in the bootstrap sample
                # while introducing variance needed for bagging.
                if len(cls_indices) > 0:
                    sampled = rng.choice(
                        cls_indices, size=len(cls_indices), replace=True
                    )
                    resampled_indices.append(sampled)

            # Combine all indices
            if resampled_indices:
                resampled_indices = np.concatenate(resampled_indices)

                # Shuffle indices to mix classes (optional but good practice)
                rng.shuffle(resampled_indices)

                # Create bootstrap dataset
                X_boot = X[resampled_indices]
                y_boot = y[resampled_indices]

                # Clone and fit the base estimator
                estimator = clone(base_estimator)
                estimator.fit(X_boot, y_boot)
                self.estimators_.append(estimator)

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for X by averaging predictions from all base estimators.
        """
        if not self.estimators_:
            raise RuntimeError("The model has not been fitted yet.")

        # Initialize probability accumulator
        # Shape: (n_samples, n_classes)
        proba_sum = np.zeros((X.shape[0], self.n_classes_))

        # Sum probabilities from all estimators
        for estimator in self.estimators_:
            # LDA predict_proba returns columns in sorted order of classes
            proba_sum += estimator.predict_proba(X)

        # Average
        avg_proba = proba_sum / len(self.estimators_)

        return avg_proba

    def predict(self, X):
        """
        Predict class labels for X.
        """
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


def train_model(load_cached_data=True):
    """
    Loads preprocessed data, trains the StratifiedBaggedLDA model,
    and evaluates it on the validation set.
    """
    # 1. Load Preprocessed Data
    # The preprocessing pipeline (PowerTransformer + StandardScaler) is handled in library/preprocessing.py
    print("Loading preprocessed data for training...")
    X_train, y_train, X_val, y_val, _, _, _ = preprocessing.get_preprocessed_data(
        load_cached_data=load_cached_data
    )

    # 2. Initialize Model
    print(f"Initializing StratifiedBaggedLDA with {config.N_ESTIMATORS} estimators...")
    model = StratifiedBaggedLDA(
        n_estimators=config.N_ESTIMATORS,
        solver=config.LDA_SOLVER,
        shrinkage=config.LDA_SHRINKAGE,
        random_state=config.SEED,
    )

    # 3. Fit Model
    print("Fitting model...")
    model.fit(X_train, y_train)

    # 4. Evaluate
    print("Evaluating on validation set...")
    y_pred_val = model.predict_proba(X_val)

    # Calculate Log Loss
    loss = utils.calculate_log_loss(y_val, y_pred_val)
    print(f"Validation Log Loss: {loss}")

    return model


def predict_and_submit(model, load_cached_data=True):
    """
    Generates predictions for the test set and saves the submission file.
    """
    # 1. Load Test Data
    print("Loading preprocessed test data...")
    _, _, _, _, X_test, test_ids, classes = preprocessing.get_preprocessed_data(
        load_cached_data=load_cached_data
    )

    # 2. Generate Predictions
    print("Predicting probabilities for test set...")
    y_pred_test = model.predict_proba(X_test)

    # 3. Save Submission
    utils.save_submission(
        test_ids, y_pred_test, classes, output_path=config.SUBMISSION_CSV
    )
