import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library.config import Config


class StreamARF:
    """
    Stream A: Dual-Lexical Augmented Random Forest.

    This model combines:
    1. Full-Spectrum Metadata (rf_meta)
    2. Dual-Lexical TF-IDF Features (rf_tfidf)
    3. Discrete Semantic Topics (rf_topics)
    """

    def __init__(self):
        """
        Initializes the Random Forest Classifier with hyperparameters from Config.
        """
        self.model = RandomForestClassifier(
            n_estimators=Config.RF_N_ESTIMATORS,
            class_weight=Config.RF_CLASS_WEIGHT,
            max_depth=Config.RF_MAX_DEPTH,
            min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
            n_jobs=Config.RF_N_JOBS,
            random_state=Config.RANDOM_SEED,
            verbose=0,
        )

    def _prepare_features(self, data_dict):
        """
        Concatenates the specific feature sets required for the Random Forest.

        Args:
            data_dict (dict or NpzFile): Dictionary containing feature arrays.

        Returns:
            np.ndarray: Concatenated feature matrix X.
        """
        # Feature engineering script produces dense arrays for all these components
        # rf_meta: (N, M_meta)
        # rf_tfidf: (N, M_tfidf_title + M_tfidf_body)
        # rf_topics: (N, K_topics)

        X = np.hstack(
            [data_dict["rf_meta"], data_dict["rf_tfidf"], data_dict["rf_topics"]]
        )
        return X

    def fit(self, train_data):
        """
        Trains the Random Forest model.

        Args:
            train_data (dict): Dictionary containing training features and labels ('y').
        """
        X_train = self._prepare_features(train_data)
        y_train = train_data["y"]

        print(
            f"Stream A (RF): Training on feature matrix with shape {X_train.shape}..."
        )
        self.model.fit(X_train, y_train)
        print("Stream A (RF): Training complete.")

    def predict_proba(self, data):
        """
        Generates probability predictions for the positive class.

        Args:
            data (dict): Dictionary containing features.

        Returns:
            np.ndarray: Array of probabilities for class 1.
        """
        X = self._prepare_features(data)
        # predict_proba returns [prob_0, prob_1], we want prob_1
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, val_data):
        """
        Evaluates the model on validation data using ROC AUC.

        Args:
            val_data (dict): Dictionary containing validation features and labels.

        Returns:
            float: ROC AUC score.
        """
        y_val = val_data["y"]
        preds = self.predict_proba(val_data)
        auc = roc_auc_score(y_val, preds)
        # Printing full precision as requested
        print(f"Stream A (RF) Validation ROC AUC: {auc}")
        return auc
