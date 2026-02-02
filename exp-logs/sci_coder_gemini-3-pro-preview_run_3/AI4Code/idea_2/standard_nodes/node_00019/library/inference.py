import os
import lightgbm as lgb
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import set_seed
from library.feature_extractor import EmbeddingGenerator, RegressionFeatureGenerator


class InferenceEngine:
    """
    Manages the inference process for the LightGBM model.
    """

    def __init__(self, debug_limit: int = None):
        self.debug_limit = debug_limit
        set_seed(Config.SEED)

    def _ensure_features(self):
        """
        Ensures that the test set features (Parquet file) exist.
        """
        # 1. Embeddings
        if not os.path.exists(Config.TEST_CACHE_PATH):
            print(
                f"Test embeddings not found at {Config.TEST_CACHE_PATH}. Generating..."
            )
            gen = EmbeddingGenerator()
            gen.process_split("test", debug_limit=self.debug_limit)

        # 2. Regression Features
        if not os.path.exists(Config.TEST_TABULAR_PATH):
            print(
                f"Test tabular features not found at {Config.TEST_TABULAR_PATH}. Generating..."
            )
            proc = RegressionFeatureGenerator()
            proc.process_split("test")

    def run_inference(self):
        """
        Runs the model on the test dataset.
        """
        # 1. Ensure input features exist
        self._ensure_features()

        # 2. Load Features
        print("Loading test features...")
        df_test = pd.read_parquet(Config.TEST_TABULAR_PATH)

        feature_cols = [
            "n_code",
            "sim_max",
            "sim_mean",
            "sim_std",
            "best_match_loc",
            "center_of_mass",
        ]
        X_test = df_test[feature_cols]

        # 3. Load Model
        print("Loading model...")
        if not os.path.exists(Config.MODEL_SAVE_PATH):
            raise FileNotFoundError(f"Model not found at {Config.MODEL_SAVE_PATH}")

        model = lgb.Booster(model_file=Config.MODEL_SAVE_PATH)

        # 4. Predict
        print(f"Predicting on {len(X_test)} samples...")
        preds = model.predict(X_test)

        # Map back to (nb_id, cell_id)
        predictions = {}
        for idx, row in df_test.iterrows():
            predictions[(row["notebook_id"], row["cell_id"])] = float(preds[idx])

        return predictions

    def generate_submission(self):
        # 1. Get predictions
        md_predictions = self.run_inference()

        # 2. Load structural data to reconstruct
        # We use the embeddings file for structure as it has cell types
        print("Loading test structure...")
        df = pd.read_parquet(Config.TEST_CACHE_PATH)

        if self.debug_limit:
            target_nb_ids = df["notebook_id"].unique()[: self.debug_limit]
            df = df[df["notebook_id"].isin(target_nb_ids)]

        # 3. Reconstruct Order
        print("Reconstructing cell orders...")
        df["orig_idx"] = np.arange(len(df))
        grouped = df.groupby("notebook_id")
        submission_rows = []

        for nb_id, group in grouped:
            code_cells = group[group["cell_type"] == "code"].sort_values("orig_idx")
            md_cells = group[group["cell_type"] == "markdown"]

            ranked_cells = []

            # Code anchors
            code_ids = code_cells["cell_id"].tolist()
            n_code = len(code_ids)
            for i, cid in enumerate(code_ids):
                ranked_cells.append((cid, float(i)))

            # Markdown
            for _, row in md_cells.iterrows():
                cid = row["cell_id"]
                pred_ratio = md_predictions.get((nb_id, cid), 0.0)
                rank_score = pred_ratio * n_code
                ranked_cells.append((cid, rank_score))

            ranked_cells.sort(key=lambda x: x[1])
            final_order = [x[0] for x in ranked_cells]
            submission_rows.append({"id": nb_id, "cell_order": " ".join(final_order)})

        sub_df = pd.DataFrame(submission_rows)
        sub_df = sub_df[["id", "cell_order"]]
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Submission generated successfully.")
