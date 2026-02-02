import os
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import seed_everything, calculate_log_loss, save_submission


class StratifiedSelectiveTopologyModel:
    """
    Implements the Stratified Selective-Topology Orthogonal Manifold-Densified LDA strategy.
    Manages a K-Fold ensemble of pipelines with independent subspace reduction and global variance alignment.
    """

    def __init__(self):
        self.pca_variance = Config.PCA_VARIANCE
        self.n_splits = Config.N_SPLITS
        self.seed = Config.SEED
        self.models_dir = os.path.join(Config.WORKING_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        self.models = []
        self.label_encoder = None
        self.classes_ = None

    def _get_fold_data(self, data, indices):
        """
        Extracts densified data corresponding to the given unique image indices.
        Maps unique indices [0..N-1] to densified indices [idx, idx+N, idx+2N].
        """
        n_unique = len(data["ids"]) // 3

        # Map unique indices to the 3 densified blocks (A, B, C)
        # Block A: 0 to N-1
        # Block B: N to 2N-1
        # Block C: 2N to 3N-1
        idx_A = indices
        idx_B = indices + n_unique
        idx_C = indices + 2 * n_unique

        full_indices = np.concatenate([idx_A, idx_B, idx_C])

        fold_data = {
            "dino": data["dino"][full_indices],
            "conv": data["conv"][full_indices],
            "tabular": data["tabular"][full_indices],
            "ids": data["ids"][full_indices],
        }

        if "y" in data and data["y"] is not None:
            fold_data["y"] = data["y"][full_indices]

        return fold_data

    def fit(self, train_data, val_data=None):
        """
        Trains the K-Fold ensemble on the densified training data.

        Args:
            train_data (dict): Dictionary containing densified 'dino', 'conv', 'tabular', 'ids', 'y'.
            val_data (dict, optional): External validation set (densified).
        """
        seed_everything(self.seed)

        # 1. Setup Label Encoder
        # Use unique labels to fit encoder
        n_train_unique = len(train_data["ids"]) // 3
        unique_train_y = train_data["y"][:n_train_unique]

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(unique_train_y)
        self.classes_ = self.label_encoder.classes_

        # Save classes for inference
        joblib.dump(self.classes_, os.path.join(self.models_dir, "classes.pkl"))

        # 2. Stratified K-Fold Split
        # Split based on unique images
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.seed
        )

        print(
            f"Starting {self.n_splits}-Fold Stratified Training on {n_train_unique} unique samples ({len(train_data['ids'])} densified)..."
        )

        fold_scores = []
        oof_preds = np.zeros((n_train_unique, len(self.classes_)))
        oof_targets = np.zeros(n_train_unique)

        # Dummy X for split (using zeros), y for stratification
        split_gen = skf.split(np.zeros(n_train_unique), unique_train_y)

        for fold, (train_idx, valid_idx) in enumerate(split_gen):
            print(f"\n--- Fold {fold + 1}/{self.n_splits} ---")

            # Prepare Fold Data
            X_train = self._get_fold_data(train_data, train_idx)
            X_valid = self._get_fold_data(train_data, valid_idx)

            y_train_enc = self.label_encoder.transform(X_train["y"])
            y_valid_enc = self.label_encoder.transform(
                X_valid["y"]
            )  # Densified targets

            # --- Pipeline Training ---

            # 1. Independent PCA for Visual Streams
            # Fit on densified training data to capture variance of all rotations
            pca_dino = PCA(n_components=self.pca_variance, random_state=self.seed)
            dino_train_pca = pca_dino.fit_transform(X_train["dino"])
            dino_valid_pca = pca_dino.transform(X_valid["dino"])

            pca_conv = PCA(n_components=self.pca_variance, random_state=self.seed)
            conv_train_pca = pca_conv.fit_transform(X_train["conv"])
            conv_valid_pca = pca_conv.transform(X_valid["conv"])

            # 2. Quantile Transformer for Tabular (Gaussianization)
            qt_tab = QuantileTransformer(
                output_distribution="normal", random_state=self.seed
            )
            tab_train_qt = qt_tab.fit_transform(X_train["tabular"])
            tab_valid_qt = qt_tab.transform(X_valid["tabular"])

            # 3. Concatenation
            X_train_concat = np.hstack([dino_train_pca, conv_train_pca, tab_train_qt])
            X_valid_concat = np.hstack([dino_valid_pca, conv_valid_pca, tab_valid_qt])

            # 4. Global Variance Alignment
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_concat)
            X_valid_scaled = scaler.transform(X_valid_concat)

            # 5. LDA Classifier with Ledoit-Wolf shrinkage
            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            lda.fit(X_train_scaled, y_train_enc)

            # --- Validation & Aggregation ---

            # Predict on densified validation set
            probs_densified = lda.predict_proba(X_valid_scaled)

            # Aggregate Centroids: Average (Block A + Block B + Block C) / 3
            n_val_unique = len(valid_idx)
            p_A = probs_densified[:n_val_unique]
            p_B = probs_densified[n_val_unique : 2 * n_val_unique]
            p_C = probs_densified[2 * n_val_unique :]

            probs_agg = (p_A + p_B + p_C) / 3.0

            # Get unique targets for scoring
            y_valid_unique_enc = y_valid_enc[:n_val_unique]

            # Calculate Score
            score = calculate_log_loss(y_valid_unique_enc, probs_agg)
            fold_scores.append(score)
            print(f"Fold {fold + 1} Log Loss: {score}")

            # Store OOF predictions
            oof_preds[valid_idx] = probs_agg
            oof_targets[valid_idx] = y_valid_unique_enc

            # Save Pipeline
            pipeline = {
                "pca_dino": pca_dino,
                "pca_conv": pca_conv,
                "qt_tab": qt_tab,
                "scaler": scaler,
                "lda": lda,
            }
            self.models.append(pipeline)
            joblib.dump(
                pipeline, os.path.join(self.models_dir, f"pipeline_fold_{fold}.pkl")
            )

        # --- Global Metrics ---
        global_cv_score = calculate_log_loss(oof_targets, oof_preds)
        avg_fold_score = np.mean(fold_scores)
        print("\n=== Training Complete ===")
        print(f"Average Fold Log Loss: {avg_fold_score}")
        print(f"Overall OOF Log Loss:  {global_cv_score}")

        # External Validation
        if val_data is not None:
            print("\nEvaluating on External Validation Set...")
            val_probs, _ = self.predict_proba(val_data)

            # Get unique targets
            n_val_ext_unique = len(val_data["ids"]) // 3
            y_val_ext = val_data["y"][:n_val_ext_unique]
            y_val_ext_enc = self.label_encoder.transform(y_val_ext)

            ext_score = calculate_log_loss(y_val_ext_enc, val_probs)
            print(f"External Validation Log Loss: {ext_score}")

    def predict_proba(self, test_data):
        """
        Generates aggregated probabilities for the test set using the trained ensemble.

        Args:
            test_data (dict): Densified test data.

        Returns:
            tuple: (probabilities [N_unique, N_classes], class_names)
        """
        if not self.models:
            # Try loading models if list is empty
            print("Loading models from disk...")
            try:
                self.classes_ = joblib.load(
                    os.path.join(self.models_dir, "classes.pkl")
                )
                for fold in range(self.n_splits):
                    path = os.path.join(self.models_dir, f"pipeline_fold_{fold}.pkl")
                    if os.path.exists(path):
                        self.models.append(joblib.load(path))
            except FileNotFoundError:
                pass

            if not self.models:
                raise RuntimeError("No trained models found.")

        n_samples_total = len(test_data["ids"])
        n_unique = n_samples_total // 3
        n_classes = len(self.classes_)

        # Accumulator for ensemble predictions
        ensemble_probs = np.zeros((n_unique, n_classes))

        for i, model in enumerate(self.models):
            # Transform Features
            dino_pca = model["pca_dino"].transform(test_data["dino"])
            conv_pca = model["pca_conv"].transform(test_data["conv"])
            tab_qt = model["qt_tab"].transform(test_data["tabular"])

            X_concat = np.hstack([dino_pca, conv_pca, tab_qt])
            X_scaled = model["scaler"].transform(X_concat)

            # Predict (Densified)
            probs_densified = model["lda"].predict_proba(X_scaled)

            # Aggregate Centroids (A, B, C)
            p_A = probs_densified[:n_unique]
            p_B = probs_densified[n_unique : 2 * n_unique]
            p_C = probs_densified[2 * n_unique :]

            probs_agg = (p_A + p_B + p_C) / 3.0

            ensemble_probs += probs_agg

        # Average over ensemble
        ensemble_probs /= len(self.models)

        return ensemble_probs, self.classes_

    def predict_and_save(self, test_data, output_path=Config.SUBMISSION_PATH):
        """
        Generates predictions and saves them to the submission file.
        """
        print("Generating predictions for test set...")
        probs, classes = self.predict_proba(test_data)

        # Extract unique IDs (corresponding to the aggregated probabilities)
        n_unique = len(test_data["ids"]) // 3
        unique_ids = test_data["ids"][:n_unique]

        save_submission(unique_ids, probs, classes, output_path)
