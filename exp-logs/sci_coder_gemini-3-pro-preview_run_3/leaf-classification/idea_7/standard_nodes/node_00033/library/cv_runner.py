import os
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import seed_everything, calculate_log_loss
from library.model_factory import create_hybrid_pipeline


class StratifiedEnsembleTrainer:
    """
    Manages the Stratified K-Fold Cross-Validation training of the hybrid LDA ensemble.
    Executes the training strategy, saves per-fold models, and computes OOF scores.
    """

    def __init__(self):
        """
        Initialize the trainer with configuration settings.
        """
        seed_everything(Config.SEED)
        self.n_folds = Config.N_FOLDS
        self.pipeline_path_template = Config.PIPELINE_PATH

    def cross_validate(self, dino_features, conv_features, tabular_features, labels):
        """
        Performs Stratified K-Fold Cross-Validation.

        1. Concatenates feature streams into a single matrix.
        2. Splits data into K stratified folds.
        3. For each fold:
           - Creates a new hybrid pipeline (PCA + Quantile + LDA).
           - Fits the pipeline on training data.
           - Predicts on validation data.
           - Calculates and reports Log Loss.
           - Saves the fitted pipeline to disk.

        Args:
            dino_features (np.ndarray): Global geometry features (N, D1).
            conv_features (np.ndarray): Local margin features (N, D2).
            tabular_features (np.ndarray): Engineered tabular features (N, 192).
            labels (np.ndarray): Encoded class labels (N,).

        Returns:
            list: A list of validation log loss scores for each fold.
        """
        # 1. Feature Concatenation
        # The model factory expects features in the order: [DINO | CONV | TABULAR]
        print("Concatenating features for training...")
        X = np.hstack([dino_features, conv_features, tabular_features])
        y = labels

        # Determine dimensions to pass to the model factory for correct slicing
        dino_dim = dino_features.shape[1]
        conv_dim = conv_features.shape[1]
        tabular_dim = tabular_features.shape[1]

        # Identify all unique classes to pass to log_loss
        # This prevents errors if a fold's validation set is missing some classes
        all_classes = np.unique(y)

        # 2. Stratified K-Fold Setup
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=Config.SEED
        )

        fold_scores = []

        print(f"Starting {self.n_folds}-Fold Cross-Validation...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            # Split Data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 3. Create Pipeline
            # Instantiate a fresh pipeline for this fold
            pipeline = create_hybrid_pipeline(dino_dim, conv_dim, tabular_dim)

            # 4. Train
            # Fit the pipeline (Transformers + LDA)
            pipeline.fit(X_train, y_train)

            # 5. Validate
            # Predict probabilities
            y_pred_proba = pipeline.predict_proba(X_val)

            # Calculate Metric
            # calculate_log_loss handles clipping and row normalization
            score = calculate_log_loss(y_val, y_pred_proba, labels=all_classes)
            fold_scores.append(score)

            # Print full precision score as requested
            print(f"Fold {fold} Log Loss: {score}")

            # 6. Save Pipeline
            save_path = self.pipeline_path_template.format(fold=fold)
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, "wb") as f:
                pickle.dump(pipeline, f)

        # Aggregate Results
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        print("Cross-Validation Complete.")
        print(f"Mean Log Loss: {mean_score}")
        print(f"Std Log Loss: {std_score}")

        return fold_scores

    def predict_test(self, dino_features, conv_features, tabular_features):
        """
        Generates ensemble predictions for the test set.
        Loads all K saved pipelines, generates probabilities from each,
        and computes the arithmetic mean of the probabilities.

        Args:
            dino_features (np.ndarray): Test global features (N_test, D1).
            conv_features (np.ndarray): Test local features (N_test, D2).
            tabular_features (np.ndarray): Test tabular features (N_test, 192).

        Returns:
            np.ndarray: Averaged probability predictions (N_test, N_classes).
        """
        # Concatenate features
        X_test = np.hstack([dino_features, conv_features, tabular_features])

        all_probs = []

        print(f"Generating predictions using ensemble of {self.n_folds} models...")

        for fold in range(self.n_folds):
            model_path = self.pipeline_path_template.format(fold=fold)

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model for fold {fold} not found at {model_path}. "
                    "Run cross_validate first."
                )

            # Load pipeline
            with open(model_path, "rb") as f:
                pipeline = pickle.load(f)

            # Predict
            probs = pipeline.predict_proba(X_test)
            all_probs.append(probs)

        # Average predictions (Soft Voting)
        # Shape: (N_test, N_classes)
        avg_probs = np.mean(all_probs, axis=0)

        return avg_probs
