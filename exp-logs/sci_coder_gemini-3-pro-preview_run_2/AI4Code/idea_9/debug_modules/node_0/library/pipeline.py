import os
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, compute_kendall_tau
from library.data_loader import NotebookLoader
from library.vectorizers import SemanticSpace
from library.anchor_features import AnchorEngine
from library.models import Stage1Ridge, Stage2LGBM


class RankingPipeline:
    """
    Orchestrates the Multi-View Stacked Ranking pipeline.
    Manages data loading, feature engineering, model training, and inference.
    """

    def __init__(self):
        seed_everything(Config.RANDOM_SEED)

        # Initialize components
        self.semantic_space = SemanticSpace()
        self.anchor_engine = AnchorEngine(self.semantic_space)
        self.stage1_model = Stage1Ridge()
        self.stage2_model = Stage2LGBM()

        # Placeholders for data
        self.df_train = None
        self.df_val = None
        self.df_test = None

    def load_data(self, debug=Config.DEBUG):
        """
        Loads train, validation, and test datasets using the NotebookLoader.
        """
        print("--- Loading Data ---")
        self.df_train = NotebookLoader.load_dataset(
            Config.TRAIN_METADATA_PATH,
            "train_cells",
            load_cached_data=True,
            debug=debug,
        )
        self.df_val = NotebookLoader.load_dataset(
            Config.VAL_METADATA_PATH, "val_cells", load_cached_data=True, debug=debug
        )
        self.df_test = NotebookLoader.load_dataset(
            Config.TEST_METADATA_PATH, "test_cells", load_cached_data=True, debug=debug
        )
        print(f"Train size: {len(self.df_train)}")
        print(f"Val size: {len(self.df_val)}")
        print(f"Test size: {len(self.df_test)}")

    def fit_vectorizers(self):
        """
        Fits the TF-IDF and SVD models on the training data.
        """
        print("--- Fitting Vectorizers ---")
        self.semantic_space.fit(self.df_train, load_cached_models=True)

    def run_stage1_cv(self):
        """
        Runs Cross-Validation for Stage 1 Ridge Regression on training data.
        Returns Out-Of-Fold (OOF) predictions for markdown cells.
        """
        print("--- Running Stage 1 CV (Ridge) ---")
        # Filter for markdown cells
        md_mask = self.df_train["cell_type"] == "markdown"
        df_md = self.df_train[md_mask].reset_index(drop=True)

        # Transform features
        X = self.semantic_space.transform_tfidf(df_md["source"])
        y = df_md["norm_rank"].values
        groups = df_md["ancestor_id"].values

        # Get OOF predictions
        oof_preds = self.stage1_model.get_oof_predictions(X, y, groups)
        return oof_preds

    def train_stage1_full(self):
        """
        Trains Stage 1 Ridge Regression on the full training dataset.
        Used for generating predictions on Validation and Test sets.
        """
        print("--- Training Stage 1 Full (Ridge) ---")
        md_mask = self.df_train["cell_type"] == "markdown"
        df_md = self.df_train[md_mask]

        X = self.semantic_space.transform_tfidf(df_md["source"])
        y = df_md["norm_rank"].values

        self.stage1_model.fit(X, y)

        # Save model
        model_path = os.path.join(Config.WORKING_DIR, "stage1_ridge.joblib")
        self.stage1_model.save(model_path)

    def _predict_stage1(self, df):
        """
        Helper to predict using the fully trained Stage 1 model.
        """
        md_mask = df["cell_type"] == "markdown"
        df_md = df[md_mask]

        if len(df_md) == 0:
            return np.array([])

        X = self.semantic_space.transform_tfidf(df_md["source"])
        return self.stage1_model.predict(X)

    def build_stage2_dataset(self, df, cache_name, ridge_preds, is_train=False):
        """
        Constructs the dense feature matrix for Stage 2.
        Combines Ridge predictions, Anchor features, and LSA vectors.

        Args:
            df: DataFrame containing all cells (code + markdown).
            cache_name: Name for caching anchor features.
            ridge_preds: Predictions from Stage 1 (OOF for train, Preds for val/test).
            is_train: Boolean indicating if this is training data (has targets).

        Returns:
            X: Feature matrix
            y: Target array (if is_train=True, else None)
            groups: Group array (if is_train=True, else None)
        """
        print(f"--- Building Stage 2 Dataset ({cache_name}) ---")

        # 1. Compute Anchor Features (Lexical & Latent)
        # This uses both code and markdown cells to find relationships
        anchor_df = self.anchor_engine.compute_features(
            df, cache_name, load_cached_data=True
        )

        # Filter df to markdown only, as Stage 2 only predicts ranks for markdown
        md_mask = df["cell_type"] == "markdown"
        df_md = df[md_mask].reset_index(drop=True)

        # Ensure alignment: anchor_df should correspond to markdown cells in df
        # The anchor_engine returns features for markdown cells in the same order
        if len(anchor_df) != len(df_md):
            raise ValueError(
                f"Mismatch in anchor features length: {len(anchor_df)} vs {len(df_md)}"
            )

        # 2. Get LSA Vectors (Context)
        lsa_vectors = self.semantic_space.transform_svd(df_md["source"])

        # 3. Combine Features
        # Features: [Ridge_Pred, Lex_Rank, Lex_Sim, Lat_Rank, Lat_Sim, LSA_0, ..., LSA_127]

        # Reshape ridge_preds to (N, 1)
        ridge_preds_col = ridge_preds.reshape(-1, 1)

        # Extract anchor columns
        anchor_cols = [
            "lexical_anchor_rank",
            "lexical_anchor_sim",
            "latent_anchor_rank",
            "latent_anchor_sim",
        ]
        anchor_feats = anchor_df[anchor_cols].values.astype(np.float32)

        # Concatenate
        X = np.hstack([ridge_preds_col, anchor_feats, lsa_vectors])

        y = None
        groups = None

        if is_train:
            y = df_md["norm_rank"].values
            groups = df_md["ancestor_id"].values

        return X, y, groups

    def train_stage2(self, train_oof_preds):
        """
        Trains the Stage 2 LightGBM model.
        """
        print("--- Training Stage 2 (LightGBM) ---")

        # 1. Prepare Train Data
        X_train, y_train, _ = self.build_stage2_dataset(
            self.df_train, "train", train_oof_preds, is_train=True
        )

        # 2. Prepare Validation Data
        # We need Stage 1 predictions for Val first
        val_ridge_preds = self._predict_stage1(self.df_val)

        X_val, y_val, _ = self.build_stage2_dataset(
            self.df_val, "val", val_ridge_preds, is_train=True
        )

        # 3. Train
        # Define feature names for interpretability
        feat_names = (
            ["ridge_pred"]
            + ["lex_rank", "lex_sim", "lat_rank", "lat_sim"]
            + [f"lsa_{i}" for i in range(Config.SVD_N_COMPONENTS)]
        )

        self.stage2_model.fit(X_train, y_train, X_val, y_val, feature_names=feat_names)

        # Save model
        model_path = os.path.join(Config.WORKING_DIR, "stage2_lgbm.joblib")
        self.stage2_model.save(model_path)

    def execute_training(self):
        """
        Runs the complete training pipeline.
        """
        self.load_data()
        self.fit_vectorizers()

        # Stage 1
        train_oof = self.run_stage1_cv()
        self.train_stage1_full()

        # Stage 2
        self.train_stage2(train_oof)

        print("Training pipeline completed successfully.")

    def predict_submission(self):
        """
        Runs inference on the test set and generates the submission file.
        """
        print("--- Generating Submission ---")

        # Ensure data and models are loaded
        if self.df_test is None:
            self.load_data()

        # Load models if not in memory
        if self.stage1_model.model is None:
            self.stage1_model.load(
                os.path.join(Config.WORKING_DIR, "stage1_ridge.joblib")
            )
        if self.stage2_model.model is None:
            self.stage2_model.load(
                os.path.join(Config.WORKING_DIR, "stage2_lgbm.joblib")
            )

        # 1. Stage 1 Predictions
        test_ridge_preds = self._predict_stage1(self.df_test)

        # 2. Stage 2 Features & Predictions
        X_test, _, _ = self.build_stage2_dataset(
            self.df_test, "test", test_ridge_preds, is_train=False
        )

        test_final_preds = self.stage2_model.predict(X_test)

        # 3. Reconstruct Notebook Orders
        print("Reconstructing notebook orders...")

        # Map predictions back to dataframe
        md_mask = self.df_test["cell_type"] == "markdown"
        self.df_test.loc[md_mask, "pred_rank"] = test_final_preds

        submission_data = []

        # Group by notebook to sort
        # We assume code cells are in correct relative order in the input
        for nb_id, group in tqdm(self.df_test.groupby("notebook_id", observed=True)):
            # Separate code and markdown
            code_cells = group[group["cell_type"] == "code"].copy()
            md_cells = group[group["cell_type"] == "markdown"].copy()

            # Assign ranks to code cells: 0.0 to 1.0 based on position
            n_code = len(code_cells)
            if n_code > 0:
                if n_code == 1:
                    code_cells["pred_rank"] = 0.0
                else:
                    code_cells["pred_rank"] = np.arange(n_code) / (n_code - 1)

            # Concatenate
            full_nb = pd.concat([code_cells, md_cells])

            # Sort by predicted rank
            full_nb = full_nb.sort_values("pred_rank")

            # Extract ID string
            cell_order = " ".join(full_nb["cell_id"].astype(str).tolist())

            submission_data.append({"id": nb_id, "cell_order": cell_order})

        # 4. Save Submission
        submission_df = pd.DataFrame(submission_data)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(submission_df.head())
