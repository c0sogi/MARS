import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from library.config import Config
from library.utils import set_seed
from library.data_processing import NotebookProcessor
from library.feature_extraction import DualViewFeaturePipeline
from library.model_factory import Stage1Ridge, Stage2LGBM


class TrainPipeline:
    """
    Orchestrates the two-stage training process for the Stacked Hybrid Ranking model.
    """

    def __init__(self):
        self.config = Config
        set_seed(self.config.SEED)

        # Initialize components
        self.processor = NotebookProcessor()
        self.feature_pipeline = DualViewFeaturePipeline()
        self.stage1_model = Stage1Ridge()
        self.stage2_model = Stage2LGBM()

    def run(self, load_cached_data=True):
        """
        Executes the full training pipeline.

        Args:
            load_cached_data (bool): If True, attempts to load intermediate artifacts
                                     (processed data, features, models) from disk.
                                     If False, re-computes everything from scratch.
        """
        print("Starting Training Pipeline...")

        # ----------------------------------------------------------------------
        # 1. Data Loading
        # ----------------------------------------------------------------------
        print("--- Step 1: Loading Data ---")
        df_train = self.processor.load_train_data(load_cached_data=load_cached_data)
        df_val = self.processor.load_val_data(load_cached_data=load_cached_data)

        # Filter for Markdown cells (Targets)
        # We assume the index order is preserved for alignment later
        df_train_md = df_train[df_train["cell_type"] == "markdown"].reset_index(
            drop=True
        )
        df_val_md = df_val[df_val["cell_type"] == "markdown"].reset_index(drop=True)

        # Extract Targets and Groups
        y_train = df_train_md["pct_rank"].values
        y_val = df_val_md["pct_rank"].values
        groups_train = df_train_md["ancestor_id"].values

        # ----------------------------------------------------------------------
        # 2. Feature Engineering (Global Vectorizers)
        # ----------------------------------------------------------------------
        print("--- Step 2: Fitting Vectorizers ---")

        # Check if vectorizers exist to avoid re-fitting if caching is enabled
        vectorizers_exist = os.path.exists(
            self.config.CACHE_TFIDF_MODEL
        ) and os.path.exists(self.config.CACHE_SVD_MODEL)

        if load_cached_data and vectorizers_exist:
            print("Loading existing vectorizers from cache...")
            # Models will be loaded implicitly by feature_pipeline when needed
        else:
            # Fit on full training data
            self.feature_pipeline.fit(df_train)

        # ----------------------------------------------------------------------
        # 3. Stage 1: Ridge Regression (The "Signpost" Model)
        # ----------------------------------------------------------------------
        print("--- Step 3: Stage 1 (Ridge) ---")

        # Load TF-IDF model to transform text for Ridge
        self.feature_pipeline._load_models()
        tfidf = self.feature_pipeline.tfidf

        print("Vectorizing text for Stage 1...")
        # Handle potential NaNs in source
        train_source = df_train_md["source"].astype(str).fillna("")
        val_source = df_val_md["source"].astype(str).fillna("")

        X_train_sparse = tfidf.transform(train_source)
        X_val_sparse = tfidf.transform(val_source)

        # Train Ridge on full training set (for final inference usage)
        self.stage1_model.fit(X_train_sparse, y_train)

        # Generate OOF predictions for Train (Feature for Stage 2)
        train_pred_s1 = self.stage1_model.get_oof_predictions(
            X_train_sparse,
            y_train,
            groups=groups_train,
            load_cached_data=load_cached_data,
        )

        # Generate Predictions for Val (Feature for Stage 2)
        val_pred_s1 = self.stage1_model.predict(X_val_sparse)

        mae_s1 = mean_absolute_error(y_val, val_pred_s1)
        print(f"Stage 1 Validation MAE: {mae_s1}")

        # ----------------------------------------------------------------------
        # 4. Stage 2: Feature Extraction (Neighborhoods)
        # ----------------------------------------------------------------------
        print("--- Step 4: Extracting Neighborhood Features ---")

        # Extract Decoupled Dual-View Neighborhood Features
        # These methods return DataFrames aligned with the markdown cells of the input
        df_train_feats = self.feature_pipeline.extract_features(
            df_train, mode="train", load_cached_data=load_cached_data
        )
        df_val_feats = self.feature_pipeline.extract_features(
            df_val, mode="val", load_cached_data=load_cached_data
        )

        # ----------------------------------------------------------------------
        # 5. Stage 2: Dataset Assembly
        # ----------------------------------------------------------------------
        print("--- Step 5: Assembling Stage 2 Dataset ---")

        # Identify feature columns (exclude metadata/targets)
        exclude_cols = ["id", "cell_id", "ancestor_id", "pct_rank"]
        feature_cols = [c for c in df_train_feats.columns if c not in exclude_cols]

        print(f"Stage 2 Features: {feature_cols}")

        # Convert features to numpy
        X_train_s2_base = df_train_feats[feature_cols].values
        X_val_s2_base = df_val_feats[feature_cols].values

        # Stack Stage 1 predictions as a new feature
        X_train_final = np.column_stack([X_train_s2_base, train_pred_s1])
        X_val_final = np.column_stack([X_val_s2_base, val_pred_s1])

        # Extract aligned targets from the feature dataframe (sanity check)
        y_train_s2 = df_train_feats["pct_rank"].values
        y_val_s2 = df_val_feats["pct_rank"].values

        # ----------------------------------------------------------------------
        # 6. Stage 2: LightGBM Training
        # ----------------------------------------------------------------------
        print("--- Step 6: Stage 2 (LightGBM) ---")

        self.stage2_model.fit(X_train_final, y_train_s2, X_val_final, y_val_s2)

        # ----------------------------------------------------------------------
        # 7. Final Evaluation
        # ----------------------------------------------------------------------
        print("--- Step 7: Final Evaluation ---")

        val_preds_final = self.stage2_model.predict(X_val_final)
        final_mae = mean_absolute_error(y_val_s2, val_preds_final)

        print(f"Final Validation MAE: {final_mae}")
        print("Pipeline Completed Successfully.")
