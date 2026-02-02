import os
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import seed_everything, calculate_log_loss


def build_pipeline(column_indices):
    """
    Constructs the Stratified Selective-Topology scikit-learn pipeline.

    The pipeline applies:
    1. Independent Subspace Reduction (PCA) to DINOv2 and ConvNeXt streams to preserve linear topology.
    2. Non-Linear Gaussianization (QuantileTransformer) to the Tabular stream to match LDA assumptions.
    3. Global Variance Alignment (StandardScaler) to ensure uniform regularization shrinkage.
    4. Linear Discriminant Analysis (LDA) with Ledoit-Wolf shrinkage.

    Args:
        column_indices (dict): Dictionary containing lists of indices for 'dino', 'conv', and 'tabular' features.

    Returns:
        sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    # 1. Define Transformers for each stream

    # Visual Stream 1: DINOv2 (Global Geometry)
    # We use PCA to reduce dimensions while strictly preserving the linear separability of the deep features.
    dino_transformer = PCA(n_components=Config.PCA_VARIANCE, random_state=Config.SEED)

    # Visual Stream 2: ConvNeXt (Local Texture)
    # Similarly, PCA is used here. We avoid non-linear distortions on deep features.
    conv_transformer = PCA(n_components=Config.PCA_VARIANCE, random_state=Config.SEED)

    # Tabular Stream: Handcrafted Features
    # These features have arbitrary distributions (histograms). We force them into a Gaussian distribution
    # to satisfy the normality assumption of the LDA classifier.
    tabular_transformer = QuantileTransformer(
        output_distribution=Config.TABULAR_OUTPUT_DIST, random_state=Config.SEED
    )

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("dino_pca", dino_transformer, column_indices["dino"]),
            ("conv_pca", conv_transformer, column_indices["conv"]),
            ("tab_qt", tabular_transformer, column_indices["tabular"]),
        ],
        n_jobs=None,
        verbose=False,
    )

    # 2. Define Classifier
    # LDA with Ledoit-Wolf shrinkage ('auto') is robust for HDLSS (High Dimension, Low Sample Size) problems.
    # Solver 'lsqr' supports shrinkage.
    classifier = LinearDiscriminantAnalysis(
        solver=Config.LDA_SOLVER, shrinkage=Config.LDA_SHRINKAGE
    )

    # 3. Construct Pipeline
    # Preprocessor -> Global Variance Alignment -> Classifier
    # StandardScaler is crucial here to ensure the Ledoit-Wolf shrinkage penalty is applied uniformly
    # across the PCA components (visual) and Quantile components (tabular).
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )

    return pipeline


class EnsembleTrainer:
    """
    Manages the Stratified K-Fold training and inference of the LDA ensemble.
    Handles the 'Orthogonal Manifold Densification' strategy where predictions
    are aggregated across 3 orthogonal centroids per image.
    """

    def __init__(self, model_dir):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        seed_everything(Config.SEED)

    def train(self, X, y, ids, column_indices):
        """
        Trains the ensemble using Stratified K-Fold on unique image IDs.

        Args:
            X (np.ndarray): Densified feature matrix [3N, D].
            y (np.ndarray): Labels [3N].
            ids (np.ndarray): Image IDs [3N].
            column_indices (dict): Feature column mapping.
        """
        # Identify Unique Images for Splitting
        # The dataset is densified (3 rows per image). We must split based on unique IDs
        # to ensure all 3 centroids of an image stay in the same fold (prevent leakage).
        unique_ids, unique_indices = np.unique(ids, return_index=True)
        unique_labels = y[unique_indices]

        # Initialize Stratified K-Fold
        skf = StratifiedKFold(
            n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
        )

        fold_metrics = []
        print(f"Starting {Config.N_FOLDS}-Fold Stratified Training...")

        for fold, (train_idx_unique, val_idx_unique) in enumerate(
            skf.split(unique_ids, unique_labels)
        ):
            # Map unique indices back to the densified dataset indices
            train_ids_set = set(unique_ids[train_idx_unique])
            val_ids_set = set(unique_ids[val_idx_unique])

            # Create boolean masks
            train_mask = np.isin(ids, list(train_ids_set))
            val_mask = np.isin(ids, list(val_ids_set))

            # Split data
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]

            # Build and Fit Pipeline
            pipeline = build_pipeline(column_indices)
            pipeline.fit(X_train, y_train)

            # Save Model
            model_path = os.path.join(self.model_dir, f"pipeline_fold_{fold}.pkl")
            joblib.dump(pipeline, model_path)

            # Validation with Aggregation
            # Predict on all densified validation samples [3 * N_val]
            val_probs_densified = pipeline.predict_proba(X_val)
            val_ids_densified = ids[val_mask]
            val_y_densified = y[val_mask]

            # Aggregate predictions by ID (Average the 3 centroids)
            u_val_ids = np.unique(val_ids_densified)
            agg_probs = []
            agg_targets = []

            for uid in u_val_ids:
                # Find indices for this image ID
                idx_map = np.where(val_ids_densified == uid)[0]

                # Average probabilities
                mean_prob = np.mean(val_probs_densified[idx_map], axis=0)
                agg_probs.append(mean_prob)

                # Get ground truth label (all rows for this ID have the same label)
                agg_targets.append(val_y_densified[idx_map[0]])

            agg_probs = np.array(agg_probs)
            agg_targets = np.array(agg_targets)

            # Calculate and Print Metric
            loss = calculate_log_loss(agg_targets, agg_probs)
            fold_metrics.append(loss)
            print(f"Fold {fold} Log Loss: {loss:.10f}")

        print(f"Average Log Loss: {np.mean(fold_metrics):.10f}")

        # Save class mapping for inference
        joblib.dump(pipeline.classes_, os.path.join(self.model_dir, "classes.pkl"))

    def predict(self, X, ids, column_indices):
        """
        Generates predictions using the trained ensemble with Full-Manifold Test-Time Aggregation.

        1. Predicts using all K models on all 3 centroids.
        2. Averages predictions across models.
        3. Averages predictions across centroids (IDs).

        Args:
            X (np.ndarray): Densified test features [3N, D].
            ids (np.ndarray): Test IDs [3N].
            column_indices (dict): Feature mapping.

        Returns:
            tuple: (unique_ids, averaged_probabilities, class_names)
        """
        classes_path = os.path.join(self.model_dir, "classes.pkl")
        if not os.path.exists(classes_path):
            raise FileNotFoundError("Classes file not found. Ensure model is trained.")

        classes = joblib.load(classes_path)

        n_samples = X.shape[0]
        n_classes = len(classes)

        # Accumulate predictions from all folds
        ensemble_probs_densified = np.zeros((n_samples, n_classes))

        for fold in range(Config.N_FOLDS):
            model_path = os.path.join(self.model_dir, f"pipeline_fold_{fold}.pkl")
            pipeline = joblib.load(model_path)

            # Predict
            probs = pipeline.predict_proba(X)
            ensemble_probs_densified += probs

        # Average across folds
        ensemble_probs_densified /= Config.N_FOLDS

        # Aggregate by ID (Average across the 3 orthogonal centroids)
        unique_ids = np.unique(ids)
        final_probs = []

        # Iterate through unique IDs to preserve order or map correctly
        # Note: np.unique sorts the IDs.
        for uid in unique_ids:
            idx_map = np.where(ids == uid)[0]
            # Average the ensemble predictions for the 3 views of this image
            mean_prob = np.mean(ensemble_probs_densified[idx_map], axis=0)
            final_probs.append(mean_prob)

        return unique_ids, np.array(final_probs), classes
