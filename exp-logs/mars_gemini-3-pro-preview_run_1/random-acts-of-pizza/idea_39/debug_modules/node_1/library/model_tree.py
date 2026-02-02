import numpy as np
from sklearn.ensemble import RandomForestClassifier
from library.config import (
    RF_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF,
    RF_CLASS_WEIGHT,
    RF_N_JOBS,
    SEED,
)
from library.utils import set_seed, compute_auc
from library.features import FeaturePipeline


class DualViewRandomForest:
    """
    Stream A: Dual-View Consistency Random Forest.

    This class implements the tree-based component of the hybrid ensemble.
    It uses a Random Forest Classifier trained on a rich feature set including:
    - TF-IDF vectors (Title + Body)
    - Numerical Metadata (Account age, karma, etc.)
    - Top-K Community Binary Flags
    - Dual-View Global Consistency Scalars (Topic & Narrative alignment)
    """

    def __init__(self):
        """
        Initialize the Random Forest model with hyperparameters from config.
        """
        set_seed(SEED)
        self.model = RandomForestClassifier(
            n_estimators=RF_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            class_weight=RF_CLASS_WEIGHT,
            n_jobs=RF_N_JOBS,
            random_state=SEED,
            verbose=0,  # Silent execution
        )

    def fit(self, X, y):
        """
        Trains the Random Forest model on the provided data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.
        """
        self.model.fit(X, y)

    def predict_proba(self, X):
        """
        Generates probability predictions for the positive class (received pizza).

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Probabilities for class 1.
        """
        # predict_proba returns shape (n_samples, n_classes).
        # We assume binary classification and take the second column (index 1).
        return self.model.predict_proba(X)[:, 1]

    def run(self, load_cached_data=True):
        """
        Orchestrates the full pipeline for the Random Forest stream:
        1. Loads pre-processed features using FeaturePipeline.
        2. Trains the model on the training set.
        3. Evaluates performance on the validation set.
        4. Generates predictions for the test set.

        Args:
            load_cached_data (bool): Whether to load features from the cache if available.

        Returns:
            tuple: (test_ids, test_preds, val_auc)
                - test_ids (np.ndarray): Request IDs for the test set.
                - test_preds (np.ndarray): Predicted probabilities for the test set.
                - val_auc (float): AUC score on the validation set.
        """
        # 1. Load Data
        print("Initializing Feature Pipeline for Random Forest...")
        pipeline = FeaturePipeline()

        # The pipeline returns a tuple (rf_data, mlp_data). We only need rf_data.
        rf_data, _ = pipeline.run(load_cached_data=load_cached_data)

        X_train = rf_data["X_train"]
        y_train = rf_data["y_train"]
        X_val = rf_data["X_val"]
        y_val = rf_data["y_val"]
        X_test = rf_data["X_test"]
        test_ids = rf_data["test_ids"]

        print(
            f"RF Data Loaded. Shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}"
        )

        # 2. Train
        print("Training Random Forest...")
        self.fit(X_train, y_train)

        # 3. Validate
        print("Evaluating on Validation Set...")
        val_preds = self.predict_proba(X_val)
        val_auc = compute_auc(y_val, val_preds)

        # Print full precision as requested
        print(f"Random Forest Validation AUC: {val_auc}")

        # 4. Test Inference
        print("Generating Test Predictions...")
        test_preds = self.predict_proba(X_test)

        return test_ids, test_preds, val_auc
