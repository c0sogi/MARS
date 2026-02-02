import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import log_loss, accuracy_score

from library.config import Config
from library.utils import seed_everything, save_submission
from library.preprocessing import FeaturePipeline
from library.data_loader import _get_class_mapping


class SingleLDA:
    """
    A wrapper for a single Linear Discriminant Analysis (LDA) model.
    Optimized for high-dimensional, linearly separable feature spaces where
    ensembling/subsampling introduces bias (Cite solution_lesson_node_00034).
    """

    def __init__(
        self,
        solver=Config.LDA_SOLVER,
        shrinkage=Config.LDA_SHRINKAGE,
    ):
        self.solver = solver
        self.shrinkage = shrinkage
        self.clf = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Fits the LDA model on the full feature set.
        """
        seed_everything(Config.SEED)
        self.classes_ = np.unique(y)

        print(f"Training Single LDA with full feature set ({X.shape[1]} features)...")
        self.clf = LinearDiscriminantAnalysis(
            solver=self.solver, shrinkage=self.shrinkage
        )
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X):
        if self.clf is None:
            raise RuntimeError("Model not fitted yet.")
        return self.clf.predict_proba(X)

    def predict(self, X):
        if self.clf is None:
            raise RuntimeError("Model not fitted yet.")
        return self.clf.predict(X)


def train_evaluate_predict(load_cached_data=True):
    """
    Main execution function to run the training pipeline.

    1. Loads processed data (Train, Val, Test) via FeaturePipeline.
    2. Trains the RandomSubspaceLDA ensemble.
    3. Evaluates performance on the Validation set.
    4. Generates predictions for the Test set.
    5. Saves the submission file.

    Args:
        load_cached_data (bool): Whether to attempt loading features from cache.
    """
    # Set global seed
    seed_everything(Config.SEED)

    print("========================================")
    print("Starting Random Subspace LDA Pipeline")
    print("========================================")

    # 1. Initialize Pipeline and Load Data
    pipeline = FeaturePipeline()

    print("\n[1/5] Loading Training Data...")
    X_train, y_train, _ = pipeline.get_processed_data(
        "train", load_cached_data=load_cached_data
    )

    print("\n[2/5] Loading Validation Data...")
    X_val, y_val, _ = pipeline.get_processed_data(
        "val", load_cached_data=load_cached_data
    )

    # 2. Train Model
    print("\n[3/5] Training Ensemble Model...")
    model = RandomSubspaceLDA(
        n_estimators=Config.N_ESTIMATORS,
        subspace_fraction=Config.SUBSPACE_FRACTION,
        solver=Config.LDA_SOLVER,
        shrinkage=Config.LDA_SHRINKAGE,
        random_state=Config.SEED,
    )

    model.fit(X_train, y_train)

    # 3. Evaluate
    print("\n[4/5] Evaluating on Validation Set...")
    val_probs = model.predict_proba(X_val)
    val_preds = model.predict(X_val)

    # Calculate metrics
    # y_val contains class indices (0..98).
    # model.classes_ contains the sorted unique class indices found in training.
    val_loss = log_loss(y_val, val_probs, labels=model.classes_)
    val_acc = accuracy_score(y_val, val_preds)

    print(f"Validation Log Loss: {val_loss}")
    print(f"Validation Accuracy: {val_acc}")

    # 4. Predict on Test and Save
    print("\n[5/5] Generating Test Predictions...")
    X_test, _, ids_test = pipeline.get_processed_data(
        "test", load_cached_data=load_cached_data
    )

    test_probs = model.predict_proba(X_test)

    # Retrieve class names for column headers
    # _get_class_mapping returns (dict, list_of_names)
    _, class_names = _get_class_mapping(load_cached_data=load_cached_data)

    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    save_submission(ids_test, test_probs, class_names, filename=Config.SUBMISSION_PATH)

    print("Pipeline completed successfully.")
