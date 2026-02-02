import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import log_loss

from library.config import Config
from library.utils import setup_logger, save_model, load_model
from library.data_manager import LeafDataManager


class ModelTrainer:
    """
    Handles the construction, training, and evaluation of the
    Selective-Topology Orthogonal Manifold-Densified LDA pipeline.
    """

    def __init__(self):
        self.logger = setup_logger("ModelTrainer")
        self.data_manager = LeafDataManager()

    def _concat_features(self, data_dict):
        """
        Concatenates separate feature arrays into a single X matrix.
        Returns X and the column slice indices for each feature type.
        """
        dino = data_dict["dino"]
        conv = data_dict["conv"]
        tab = data_dict["tab"]

        # Dimensions
        dim_dino = dino.shape[1]
        dim_conv = conv.shape[1]
        dim_tab = tab.shape[1]

        # Concatenate horizontally: [DINO, CONV, TAB]
        X = np.hstack([dino, conv, tab])

        # Define slices
        slices = {
            "dino": slice(0, dim_dino),
            "conv": slice(dim_dino, dim_dino + dim_conv),
            "tab": slice(dim_dino + dim_conv, dim_dino + dim_conv + dim_tab),
        }

        return X, slices

    def create_model_pipeline(self, slices):
        """
        Constructs the Scikit-Learn pipeline with independent subspace reduction
        and selective feature topology.
        """
        # 1. Column Transformer
        # - Visual streams: PCA (Linear topology preservation)
        # - Tabular stream: QuantileTransformer (Gaussianization)
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "pca_dino",
                    PCA(n_components=Config.PCA_VARIANCE, random_state=Config.SEED),
                    slices["dino"],
                ),
                (
                    "pca_conv",
                    PCA(n_components=Config.PCA_VARIANCE, random_state=Config.SEED),
                    slices["conv"],
                ),
                (
                    "quant_tab",
                    QuantileTransformer(
                        output_distribution=Config.TABULAR_OUTPUT_DIST,
                        random_state=Config.SEED,
                    ),
                    slices["tab"],
                ),
            ],
            verbose_feature_names_out=False,
        )

        # 2. Pipeline
        # - Preprocessor
        # - Global Variance Alignment (StandardScaler)
        # - Classifier (LDA with Ledoit-Wolf shrinkage)
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LinearDiscriminantAnalysis(
                        solver=Config.LDA_SOLVER, shrinkage=Config.LDA_SHRINKAGE
                    ),
                ),
            ]
        )

        return pipeline

    def train_ensemble(self):
        """
        Trains the K-Fold ensemble.
        """
        self.logger.info(f"Starting training for {Config.N_FOLDS} folds...")
        oof_preds = []
        oof_targets = []
        scores = []

        # We need a consistent label encoder across all folds
        # Load full training metadata to fit the encoder
        df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
        df_val = pd.read_csv(Config.VAL_METADATA_PATH)
        all_species = pd.concat([df_train["species"], df_val["species"]]).unique()

        le = LabelEncoder()
        le.fit(all_species)

        # Save the encoder for inference
        save_model(le, "label_encoder.pkl")
        self.logger.info(f"LabelEncoder saved. Classes: {len(le.classes_)}")

        for fold_idx in range(Config.N_FOLDS):
            self.logger.info(f"\n--- Fold {fold_idx} ---")

            # 1. Get Data
            train_data, train_labels, val_data, val_labels = (
                self.data_manager.get_fold_data(fold_idx, load_cached_data=True)
            )

            # 2. Prepare X and y
            X_train, slices = self._concat_features(train_data)
            y_train = le.transform(train_labels)

            X_val, _ = self._concat_features(val_data)
            y_val = le.transform(val_labels)

            # 3. Create Pipeline
            model = self.create_model_pipeline(slices)

            # 4. Train
            model.fit(X_train, y_train)

            # 5. Evaluate
            # Predict probabilities
            y_pred_prob = model.predict_proba(X_val)

            # Clip probabilities to avoid log(0)
            y_pred_prob = np.clip(y_pred_prob, 1e-15, 1 - 1e-15)

            score = log_loss(y_val, y_pred_prob)
            scores.append(score)
            self.logger.info(f"Fold {fold_idx} Log Loss: {score}")

            # Store for OOF calculation
            oof_preds.append(y_pred_prob)
            oof_targets.append(y_val)

            # 6. Save Model
            save_model(model, f"pipeline_fold_{fold_idx}.pkl")

        # Calculate Mean Score
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        self.logger.info(f"\nTraining Complete.")
        self.logger.info(f"Mean Log Loss: {mean_score}")
        self.logger.info(f"Std Log Loss:  {std_score}")

    def predict_ensemble(self):
        """
        Generates predictions for the test set using the trained ensemble.
        Aggregates predictions across folds and orthogonal centroids.
        """
        self.logger.info("Starting inference...")

        # 1. Load Data
        test_data, test_ids_densified = self.data_manager.get_test_data(
            load_cached_data=True
        )
        X_test, _ = self._concat_features(test_data)

        # 2. Load Label Encoder
        try:
            le = load_model("label_encoder.pkl")
        except FileNotFoundError:
            self.logger.error("Label encoder not found. Run train_ensemble first.")
            return

        # 3. Predict with each fold
        fold_probs = []

        for fold_idx in range(Config.N_FOLDS):
            model_name = f"pipeline_fold_{fold_idx}.pkl"
            try:
                model = load_model(model_name)
            except FileNotFoundError:
                self.logger.warning(f"Model {model_name} not found. Skipping.")
                continue

            probs = model.predict_proba(X_test)
            fold_probs.append(probs)

        if not fold_probs:
            self.logger.error("No models loaded.")
            return

        # 4. Average across folds
        # Shape: (N_test_densified, N_classes)
        avg_fold_probs = np.mean(fold_probs, axis=0)

        # 5. Aggregate Centroids (Full-Manifold Aggregation)
        # The test_ids_densified contains repeated IDs (3 per image)
        # We group by ID and take the mean

        df_preds = pd.DataFrame(avg_fold_probs, columns=le.classes_)
        df_preds["id"] = test_ids_densified

        # Group by ID and average
        final_preds = df_preds.groupby("id").mean().reset_index()

        # 6. Post-processing
        # Clip probabilities as per metric requirement
        numeric_cols = final_preds.columns.drop("id")
        final_preds[numeric_cols] = final_preds[numeric_cols].clip(1e-15, 1 - 1e-15)

        # 7. Save Submission
        self.logger.info(f"Saving submission to {Config.SUBMISSION_PATH}...")
        final_preds.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info("Submission saved successfully.")

        # Print sample
        self.logger.info("Sample predictions:")
        print(final_preds.head())
