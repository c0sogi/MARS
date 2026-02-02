import os
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import log_loss
from library.config import Config
from library.data_processing import get_processed_data, get_stratified_folds
from library.utils import seed_everything


class SelectiveTopologyPipeline:
    """
    Implements the Selective-Topology Orthogonal Manifold-Densified LDA pipeline.

    Architecture:
    1. Visual Stream: PCA (Linear, No Whitening) to preserve linear topology.
    2. Tabular Stream: QuantileTransformer (Normal) to gaussianize arbitrary histograms.
    3. Fusion: Concatenation.
    4. Classifier: LDA with Ledoit-Wolf shrinkage.
    """

    def __init__(self):
        # Independent Subspace Reduction for Visual Stream
        # Retain 99% variance, strictly linear (no whitening)
        self.pca = PCA(n_components=Config.PCA_VARIANCE_THRESHOLD, whiten=False)

        # Gaussianization for Tabular Stream
        self.qt = QuantileTransformer(
            output_distribution="normal", random_state=Config.RANDOM_SEED
        )

        # Classifier
        # solver='lsqr' with shrinkage='auto' utilizes the Ledoit-Wolf lemma for covariance estimation
        self.lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

        self.visual_cols = []
        self.tabular_cols = []
        self.classes_ = None

    def _split_features(self, df):
        """Identifies and separates visual and tabular feature columns."""
        if not self.visual_cols:
            # Infer columns during fit
            self.visual_cols = [c for c in df.columns if c.startswith("feat_")]
            self.tabular_cols = [
                c
                for c in df.columns
                if c.startswith("margin")
                or c.startswith("shape")
                or c.startswith("texture")
            ]

        X_visual = df[self.visual_cols].values
        X_tabular = df[self.tabular_cols].values
        return X_visual, X_tabular

    def fit(self, df, y):
        """
        Fits the pipeline components.
        Args:
            df (pd.DataFrame): Training data containing features.
            y (np.array): Target labels.
        """
        X_visual, X_tabular = self._split_features(df)

        # 1. Fit-Transform Visual Stream (PCA)
        X_visual_pca = self.pca.fit_transform(X_visual)

        # 2. Fit-Transform Tabular Stream (Quantile)
        X_tabular_qt = self.qt.fit_transform(X_tabular)

        # 3. Early Fusion
        X_fused = np.hstack([X_visual_pca, X_tabular_qt])

        # 4. Train Classifier
        self.lda.fit(X_fused, y)
        self.classes_ = self.lda.classes_

        return self

    def predict_proba(self, df):
        """
        Predicts class probabilities.
        Args:
            df (pd.DataFrame): Data containing features.
        Returns:
            np.ndarray: Probability matrix (N_samples, N_classes).
        """
        X_visual, X_tabular = self._split_features(df)

        # 1. Transform Visual Stream
        X_visual_pca = self.pca.transform(X_visual)

        # 2. Transform Tabular Stream
        X_tabular_qt = self.qt.transform(X_tabular)

        # 3. Fusion
        X_fused = np.hstack([X_visual_pca, X_tabular_qt])

        # 4. Predict
        return self.lda.predict_proba(X_fused)

    def save(self, filepath):
        """Saves the pipeline to disk."""
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath):
        """Loads the pipeline from disk."""
        return joblib.load(filepath)


def train_ensemble(sample_limit=None):
    """
    Orchestrates the K-Fold training of the SelectiveTopologyPipeline.

    Args:
        sample_limit (int, optional): Limit dataset size for debugging.
    """
    seed_everything(Config.RANDOM_SEED)

    # 1. Load Data
    # This handles caching of the densified dataset (3 centroids per image)
    df_train = get_processed_data(
        mode="train", load_cached_data=True, sample_limit=sample_limit
    )

    if df_train.empty:
        print("Error: Training data is empty.")
        return

    # 2. Generate Stratified Folds
    # Ensures all 3 centroids of an image stay in the same fold
    folds = get_stratified_folds(
        df_train, n_folds=Config.N_FOLDS, seed=Config.RANDOM_SEED
    )

    oof_preds = []
    oof_targets = []
    fold_scores = []

    print(f"\nStarting {Config.N_FOLDS}-Fold Cross-Validation...")

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n--- Fold {fold_idx} ---")

        # Split Data
        train_df = df_train.iloc[train_idx].reset_index(drop=True)
        val_df = df_train.iloc[val_idx].reset_index(drop=True)

        y_train = train_df["species"].values
        y_val = val_df["species"].values

        # Initialize and Train Pipeline
        model = SelectiveTopologyPipeline()
        model.fit(train_df, y_train)

        # Evaluate
        # We evaluate on the densified validation set (all 3 centroids)
        # This gives a proxy for performance, though final inference aggregates them.
        y_pred_proba = model.predict_proba(val_df)

        # Compute Metric
        score = log_loss(y_val, y_pred_proba, labels=model.classes_)
        print(f"Fold {fold_idx} Log Loss: {score}")

        fold_scores.append(score)

        # Save Model
        model_path = os.path.join(
            Config.CACHE_PATH_MODELS, f"pipeline_fold_{fold_idx}.pkl"
        )
        model.save(model_path)
        print(f"Saved model to {model_path}")

    print("\n--- Training Complete ---")
    print(f"Average Log Loss: {np.mean(fold_scores)}")
    print(f"Std Dev Log Loss: {np.std(fold_scores)}")


def generate_submission(sample_limit=None):
    """
    Generates predictions for the test set using the trained ensemble.
    Performs Full-Manifold Test-Time Aggregation (averaging 3 centroids per image).
    """
    seed_everything(Config.RANDOM_SEED)

    # 1. Load Test Data
    df_test = get_processed_data(
        mode="test", load_cached_data=True, sample_limit=sample_limit
    )

    if df_test.empty:
        print("Error: Test data is empty.")
        return

    # 2. Load Models
    models = []
    for fold_idx in range(Config.N_FOLDS):
        model_path = os.path.join(
            Config.CACHE_PATH_MODELS, f"pipeline_fold_{fold_idx}.pkl"
        )
        if os.path.exists(model_path):
            models.append(SelectiveTopologyPipeline.load(model_path))
        else:
            print(f"Warning: Model for fold {fold_idx} not found at {model_path}")

    if not models:
        print("Error: No trained models found.")
        return

    print(
        f"\nGenerating predictions using {len(models)} models on {len(df_test)} test samples (densified)..."
    )

    # 3. Generate Predictions
    # We collect predictions from all models
    # Shape: (N_Models, N_Test_Samples, N_Classes)
    all_preds = []

    for i, model in enumerate(models):
        preds = model.predict_proba(df_test)
        all_preds.append(preds)

    # Average across models (Ensemble Averaging)
    # Shape: (N_Test_Samples, N_Classes)
    avg_preds_densified = np.mean(all_preds, axis=0)

    # 4. Aggregate Centroids (Test-Time Augmentation)
    # df_test has 3 rows per image (Centroids A, B, C)
    # We need to average these 3 rows for each unique image ID

    # Get unique IDs and their order
    unique_ids = df_test["id"].unique()

    # Prepare final prediction array
    # Assuming classes are consistent across all models (they are fitted on same species set)
    classes = models[0].classes_
    final_preds = []
    final_ids = []

    # Iterate by unique ID to average the corresponding rows
    # Note: df_test is sorted by ID in data_processing, but let's be robust
    df_test["pred_idx"] = range(len(df_test))

    for uid in unique_ids:
        # Get indices for this image (should be 3 rows)
        indices = df_test[df_test["id"] == uid]["pred_idx"].values

        # Get predictions for these rows
        img_preds = avg_preds_densified[indices]

        # Average across the 3 centroids
        img_avg_pred = np.mean(img_preds, axis=0)

        final_preds.append(img_avg_pred)
        final_ids.append(uid)

    final_preds = np.array(final_preds)

    # 5. Post-Processing & Formatting
    # Clip probabilities to avoid log loss extremes
    final_preds = np.clip(final_preds, 1e-15, 1 - 1e-15)

    # Create Submission DataFrame
    submission_df = pd.DataFrame(final_preds, columns=classes)
    submission_df.insert(0, "id", final_ids)

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())
