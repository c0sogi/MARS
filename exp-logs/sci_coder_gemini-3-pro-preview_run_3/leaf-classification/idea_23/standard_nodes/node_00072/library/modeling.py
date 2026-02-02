import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from library.config import Config
from library.utils import seed_everything, calculate_log_loss, save_submission
from library.data_manager import DataManager


class OrthogonalLDAPipeline(BaseEstimator, ClassifierMixin):
    """
    Implements the Selective Feature Topology strategy.

    - Visual Streams (DINOv2, ConvNeXt): Processed via Independent PCA (Linear Topology Preservation).
    - Tabular Stream: Processed via QuantileTransformer (Non-Linear Gaussianization).
    - Fusion: Early fusion of processed streams.
    - Classifier: Linear Discriminant Analysis with Ledoit-Wolf shrinkage.
    """

    def __init__(self, pca_variance=0.99, random_state=Config.SEED):
        self.pca_variance = pca_variance
        self.random_state = random_state

        # Independent Subspace Reduction for Visual Streams
        # We strictly preserve linear topology for deep features
        self.pca_dino = PCA(
            n_components=pca_variance, svd_solver="full", random_state=random_state
        )
        self.pca_conv = PCA(
            n_components=pca_variance, svd_solver="full", random_state=random_state
        )

        # Non-Linear Gaussianization for Tabular Features
        self.qt_tab = QuantileTransformer(
            output_distribution="normal", random_state=random_state
        )

        # Classifier: LDA with Ledoit-Wolf shrinkage
        # solver='lsqr' supports shrinkage
        self.lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

    def fit(self, X_dino, X_conv, X_tab, y):
        # 1. Transform Visual Streams (Linear)
        X_dino_red = self.pca_dino.fit_transform(X_dino)
        X_conv_red = self.pca_conv.fit_transform(X_conv)

        # 2. Transform Tabular Stream (Non-Linear)
        X_tab_trans = self.qt_tab.fit_transform(X_tab)

        # 3. Early Fusion
        X_final = np.concatenate([X_dino_red, X_conv_red, X_tab_trans], axis=1)

        # 4. Train Classifier
        self.lda.fit(X_final, y)
        return self

    def predict_proba(self, X_dino, X_conv, X_tab):
        # Apply learned transformations
        X_dino_red = self.pca_dino.transform(X_dino)
        X_conv_red = self.pca_conv.transform(X_conv)
        X_tab_trans = self.qt_tab.transform(X_tab)

        # Concatenate
        X_final = np.concatenate([X_dino_red, X_conv_red, X_tab_trans], axis=1)

        # Predict
        return self.lda.predict_proba(X_final)


def train_and_predict():
    """
    Orchestrates the training and inference process using Manifold Densification
    and Stratified K-Fold Cross-Validation.
    """
    seed_everything(Config.SEED)

    # ==========================================
    # 1. Data Loading
    # ==========================================
    dm = DataManager()
    # Load raw 12-view features and tabular data
    (train_img, train_tab, train_lbl, train_ids), (test_img, test_tab, test_ids) = (
        dm.load_raw_data()
    )

    # Map classes to integers
    unique_classes = np.unique(train_lbl)
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    idx_to_class = {i: cls for i, cls in enumerate(unique_classes)}
    n_classes = len(unique_classes)

    y_encoded = np.array([class_to_idx[l] for l in train_lbl])

    # ==========================================
    # 2. Prepare Test Data (Densified)
    # ==========================================
    # We generate 3 orthogonal centroids for each test image.
    # We will accumulate predictions across folds on this densified set.
    test_densified = dm.create_orthogonal_centroids(test_img, test_tab, ids=test_ids)
    test_dino_dens = test_densified["dino"]
    test_conv_dens = test_densified["convnext"]
    test_tab_dens = test_densified["tabular"]
    test_ids_dens = test_densified["ids"]

    # Accumulator for test probabilities (Shape: [3*N_test, n_classes])
    test_probs_sum = np.zeros((len(test_ids_dens), n_classes))

    # ==========================================
    # 3. Stratified K-Fold Training
    # ==========================================
    # We split based on UNIQUE IDs to prevent data leakage between centroids of the same image.
    skf = dm.get_stratified_kfold()
    fold_scores = []

    print(
        f"Starting {Config.N_FOLDS}-Fold Cross-Validation with Manifold Densification..."
    )

    # We pass y_encoded to split to ensure stratification, but we split indices of unique images
    # train_img shape is (N, 12, D), so len(train_img) == number of unique images
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_img, y_encoded)):
        # Split Raw Data (Image-level split)
        X_img_train, X_img_val = train_img[train_idx], train_img[val_idx]
        X_tab_train, X_tab_val = train_tab[train_idx], train_tab[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        ids_val = train_ids[val_idx]

        # Apply Manifold Densification (N -> 3N samples)
        # Train Set: Densify to augment data
        train_dens = dm.create_orthogonal_centroids(
            X_img_train, X_tab_train, labels=y_train
        )

        # Val Set: Densify to use full manifold for evaluation
        val_dens = dm.create_orthogonal_centroids(
            X_img_val, X_tab_val, labels=y_val, ids=ids_val
        )

        # Initialize and Train Model
        model = OrthogonalLDAPipeline(
            pca_variance=Config.PCA_VARIANCE, random_state=Config.SEED
        )
        model.fit(
            train_dens["dino"],
            train_dens["convnext"],
            train_dens["tabular"],
            train_dens["labels"],
        )

        # Validation Inference
        val_probs_dens = model.predict_proba(
            val_dens["dino"], val_dens["convnext"], val_dens["tabular"]
        )

        # Validation Aggregation (Full-Manifold Aggregation)
        # Average predictions across the 3 centroids for each validation image ID
        val_pred_df = pd.DataFrame(val_probs_dens, columns=unique_classes)
        val_pred_df["id"] = val_dens["ids"]

        # Group by ID and compute mean probability vector
        val_agg = val_pred_df.groupby("id").mean()

        # Retrieve true labels (aligned by ID)
        # We use the first label found for each ID (all centroids have same label)
        val_true_labels = []
        # Since groupby sorts by ID, we need to match that order.
        # We can create a mapping from ID to Label from the val data
        id_to_label = dict(
            zip(val_dens["ids"], [idx_to_class[l] for l in val_dens["labels"]])
        )
        val_true_labels = [id_to_label[i] for i in val_agg.index]

        # Calculate Metric
        score = calculate_log_loss(val_true_labels, val_agg.values)
        fold_scores.append(score)
        print(f"Fold {fold+1} Log Loss: {score}")

        # Test Inference (Accumulate)
        test_probs = model.predict_proba(test_dino_dens, test_conv_dens, test_tab_dens)
        test_probs_sum += test_probs

    # ==========================================
    # 4. Final Aggregation and Submission
    # ==========================================
    print(f"\nAverage CV Log Loss: {np.mean(fold_scores)}")

    # Average across folds
    test_probs_avg_folds = test_probs_sum / Config.N_FOLDS

    # Aggregate across Centroids (Full-Manifold Aggregation)
    test_pred_df = pd.DataFrame(test_probs_avg_folds, columns=unique_classes)
    test_pred_df["id"] = test_ids_dens

    # Group by ID and mean
    final_submission_df = test_pred_df.groupby("id").mean().reset_index()

    # Extract final arrays
    ids = final_submission_df["id"].values
    probs = final_submission_df[unique_classes].values

    # Save
    save_submission(ids, probs, list(unique_classes))
