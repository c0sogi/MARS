import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed
from library.data_processing import NotebookProcessor
from library.feature_extraction import DualViewFeaturePipeline
from library.model_factory import Stage1Ridge, Stage2LGBM


class InferencePipeline:
    """
    Manages the inference workflow for the test set.
    Executes the two-stage stacked regression pipeline and generates the submission file.
    """

    def __init__(self):
        self.config = Config
        set_seed(self.config.SEED)

        self.processor = NotebookProcessor()
        self.feature_pipeline = DualViewFeaturePipeline()
        self.stage1_model = Stage1Ridge()
        self.stage2_model = Stage2LGBM()

    def predict_test_set(self, load_cached_data=True):
        """
        Runs the full inference pipeline on the test set.

        Args:
            load_cached_data (bool): If True, attempts to load cached processed data/features.
        """
        print("Starting Inference Pipeline...")

        # ----------------------------------------------------------------------
        # 1. Load and Prepare Test Data
        # ----------------------------------------------------------------------
        print("--- Step 1: Loading Test Data ---")
        df_test = self.processor.load_test_data(load_cached_data=load_cached_data)

        # Ensure deterministic order (Sorted by ID, then original appearance)
        # This is critical for alignment between df_test_md and extract_features output
        # (since extract_features uses groupby(sort=True))
        df_test["original_index"] = df_test.index
        df_test = df_test.sort_values(["id", "original_index"]).reset_index(drop=True)

        # ----------------------------------------------------------------------
        # 2. Assign Anchor Ranks to Code Cells
        # ----------------------------------------------------------------------
        # The feature extraction pipeline needs code cells to have valid [0, 1] ranks
        # to compute neighborhood statistics. NotebookProcessor sets them to -1 by default.
        print("--- Step 2: Assigning Anchor Ranks ---")

        # We'll update the 'pct_rank' column in place using numpy for speed
        new_pct_ranks = df_test["pct_rank"].values.copy()
        cell_types = df_test["cell_type"].values

        # Iterate over groups to calculate equidistant ranks
        # We use pandas groupby indices to avoid slow apply
        for _, indices in df_test.groupby("id").indices.items():
            # Identify code cells in this notebook
            nb_cell_types = cell_types[indices]
            code_mask = nb_cell_types == "code"
            n_code = np.sum(code_mask)

            if n_code > 0:
                if n_code == 1:
                    ranks = np.array([0.0])
                else:
                    ranks = np.linspace(0.0, 1.0, n_code)

                # Map local code mask to global indices
                global_code_indices = indices[code_mask]
                new_pct_ranks[global_code_indices] = ranks

        df_test["pct_rank"] = new_pct_ranks

        # ----------------------------------------------------------------------
        # 3. Stage 1 Inference (Ridge)
        # ----------------------------------------------------------------------
        print("--- Step 3: Stage 1 Prediction ---")

        # Filter for markdown cells (targets)
        df_test_md = df_test[df_test["cell_type"] == "markdown"].reset_index(drop=True)

        if len(df_test_md) == 0:
            print("Warning: No markdown cells in test set.")
            # Handle edge case by creating empty prediction array
            preds_s1 = np.array([])
        else:
            # Load Vectorizers
            self.feature_pipeline._load_models()

            # Vectorize
            test_source = df_test_md["source"].astype(str).fillna("")
            X_test_sparse = self.feature_pipeline.tfidf.transform(test_source)

            # Predict
            preds_s1 = self.stage1_model.predict(X_test_sparse)

        # ----------------------------------------------------------------------
        # 4. Stage 2 Feature Extraction & Inference
        # ----------------------------------------------------------------------
        print("--- Step 4: Stage 2 Feature Extraction & Prediction ---")

        if len(df_test_md) > 0:
            # Extract features using the full dataframe (Code + Markdown)
            # This uses the 'pct_rank' we fixed in Step 2
            df_test_feats = self.feature_pipeline.extract_features(
                df_test, mode="test", load_cached_data=load_cached_data
            )

            # Assemble Feature Matrix
            exclude_cols = ["id", "cell_id", "ancestor_id", "pct_rank"]
            feature_cols = [c for c in df_test_feats.columns if c not in exclude_cols]

            X_test_s2_base = df_test_feats[feature_cols].values

            # Stack Stage 1 predictions
            X_test_final = np.column_stack([X_test_s2_base, preds_s1])

            # Predict with LightGBM
            preds_s2 = self.stage2_model.predict(X_test_final)

            # Store predictions
            df_test_md["pred_rank"] = preds_s2
        else:
            df_test_md["pred_rank"] = []

        # ----------------------------------------------------------------------
        # 5. Post-Processing (Anchor Sorting)
        # ----------------------------------------------------------------------
        print("--- Step 5: Generating Submission ---")

        # Create a map for fast lookup: (id, cell_id) -> predicted_rank
        pred_map = dict(
            zip(zip(df_test_md["id"], df_test_md["cell_id"]), df_test_md["pred_rank"])
        )

        submission_rows = []

        # Process each notebook to generate the final order
        for nb_id, group in df_test.groupby("id"):
            # 1. Code Cells: Use the fixed ranks we assigned earlier
            code_cells = group[group["cell_type"] == "code"].copy()
            # Ensure they use the 'pct_rank' column which now has [0,1] values
            code_cells["final_rank"] = code_cells["pct_rank"]

            # 2. Markdown Cells: Use predicted ranks
            md_cells = group[group["cell_type"] == "markdown"].copy()
            # Lookup predictions
            md_ranks = [pred_map.get((nb_id, cid), 0.5) for cid in md_cells["cell_id"]]
            md_cells["final_rank"] = md_ranks

            # 3. Combine and Sort
            combined = pd.concat([code_cells, md_cells])
            combined = combined.sort_values("final_rank")

            # 4. Extract Order
            cell_order = " ".join(combined["cell_id"].astype(str).tolist())
            submission_rows.append({"id": nb_id, "cell_order": cell_order})

        # ----------------------------------------------------------------------
        # 6. Save Submission
        # ----------------------------------------------------------------------
        df_submission = pd.DataFrame(submission_rows)
        os.makedirs(self.config.SUBMISSION_DIR, exist_ok=True)
        df_submission.to_csv(self.config.SUBMISSION_PATH, index=False)

        print(f"Submission saved to {self.config.SUBMISSION_PATH}")
