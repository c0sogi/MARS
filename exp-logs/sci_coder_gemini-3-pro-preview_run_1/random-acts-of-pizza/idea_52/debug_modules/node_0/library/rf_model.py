import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library.config import (
    RF_N_ESTIMATORS,
    RF_MIN_SAMPLES_LEAF,
    RF_CLASS_WEIGHT,
    RF_N_JOBS,
    RF_MAX_DEPTH,
    RANDOM_STATE,
    WORKING_DIR,
)


class InteractionRandomForest:
    def __init__(self):
        """
        Initializes the Interaction-Enhanced Random Forest model.
        Hyperparameters are sourced from library.config.
        """
        self.model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            class_weight=RF_CLASS_WEIGHT,
            n_jobs=RF_N_JOBS,
            max_depth=RF_MAX_DEPTH,
            random_state=RANDOM_STATE,
            verbose=0,  # Silent execution
        )

    def _assemble_features(self, feature_dict):
        """
        Concatenates the specific feature groups required for the Random Forest.

        The ensemble strategy requires:
        1. TF-IDF (Dense representation of Title + Body)
        2. Raw Metadata (Account age, upvotes, etc.)
        3. Top-K Community Flags (Binary indicators)
        4. Consistency Scalars (Cosine similarity between text and history)
        5. Explicit Interactions (Consistency * Credibility metrics)

        Args:
            feature_dict (dict): Output from FeatureEngineer.transform/generate_features.

        Returns:
            np.ndarray: Horizontally stacked feature matrix.
        """
        # Extract components using keys defined in feature_engineer.py
        tfidf = feature_dict.get("rf_tfidf")
        metadata = feature_dict.get("rf_metadata")
        top_k = feature_dict.get("rf_top_k")
        consistency = feature_dict.get("consistency")
        interactions = feature_dict.get("rf_interactions")

        feature_list = []

        # Add features if they exist in the dictionary
        if tfidf is not None:
            feature_list.append(tfidf)

        if metadata is not None:
            feature_list.append(metadata)

        if top_k is not None:
            feature_list.append(top_k)

        if consistency is not None:
            feature_list.append(consistency)

        if interactions is not None:
            feature_list.append(interactions)

        if not feature_list:
            raise ValueError("No valid features provided for Random Forest assembly.")

        # Concatenate horizontally
        # All inputs are expected to be numpy arrays with shape (N, D)
        try:
            X = np.hstack(feature_list)
        except ValueError as e:
            shapes = [f.shape for f in feature_list]
            raise ValueError(
                f"Feature dimension mismatch during assembly. Shapes: {shapes}. Error: {e}"
            )

        return X

    def train(
        self,
        train_features,
        train_labels,
        val_features=None,
        val_labels=None,
        save_path=None,
    ):
        """
        Trains the Random Forest model and optionally evaluates on a validation set.

        Args:
            train_features (dict): Dictionary of training features.
            train_labels (array-like): Training labels (0/1).
            val_features (dict, optional): Dictionary of validation features.
            val_labels (array-like, optional): Validation labels.
            save_path (str, optional): Path to save the trained model (using joblib).
        """
        print("Assembling training features for Random Forest...")
        X_train = self._assemble_features(train_features)
        y_train = np.array(train_labels)

        print(f"Training Random Forest with input shape: {X_train.shape}...")
        self.model.fit(X_train, y_train)

        # Validation Evaluation
        if val_features is not None and val_labels is not None:
            print("Assembling validation features for Random Forest...")
            X_val = self._assemble_features(val_features)
            y_val = np.array(val_labels)

            # Predict probabilities for the positive class (index 1)
            val_probs = self.model.predict_proba(X_val)[:, 1]

            # Calculate AUC
            auc_score = roc_auc_score(y_val, val_probs)

            # Print full precision metric
            print(f"Random Forest Validation AUC: {auc_score}")

        # Save Model
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(self.model, save_path)
            print(f"Random Forest model saved to {save_path}")

    def predict_proba(self, feature_dict):
        """
        Generates probability predictions for the positive class.

        Args:
            feature_dict (dict): Dictionary of features.

        Returns:
            np.ndarray: Probabilities for class 1.
        """
        X = self._assemble_features(feature_dict)
        # Return probabilities for class 1
        probs = self.model.predict_proba(X)[:, 1]
        return probs

    def load(self, load_path):
        """
        Loads a trained model from disk.

        Args:
            load_path (str): Path to the joblib file.
        """
        if os.path.exists(load_path):
            self.model = joblib.load(load_path)
            print(f"Random Forest model loaded from {load_path}")
        else:
            raise FileNotFoundError(f"Model file not found at {load_path}")
