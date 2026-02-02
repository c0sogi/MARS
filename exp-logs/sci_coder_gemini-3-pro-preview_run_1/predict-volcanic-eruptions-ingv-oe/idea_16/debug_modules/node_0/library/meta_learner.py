import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from library.config import Config
from library.utils import seed_everything, get_score


class StackingEnsemble:
    """
    Implements the Meta-Learner for the Trend-Augmented Spectral Stacking Strategy.
    Combines predictions from the Tabular Branch (LightGBM) and Vision Branch (EfficientNet)
    using Ridge Regression.
    """

    def __init__(self):
        self.config = Config
        seed_everything(self.config.SEED)

        # Paths for input OOF and Test predictions
        self.tabular_oof_path = os.path.join(self.config.WORKING_DIR, "tabular_oof.csv")
        self.vision_oof_path = os.path.join(self.config.WORKING_DIR, "vision_oof.csv")

        self.tabular_test_path = os.path.join(
            self.config.WORKING_DIR, "tabular_test.csv"
        )
        self.vision_test_path = os.path.join(self.config.WORKING_DIR, "vision_test.csv")

        self.model = Ridge(alpha=self.config.META_ALPHA, random_state=self.config.SEED)

    def _load_ground_truth(self):
        """
        Loads and concatenates train and val metadata to get the full ground truth.
        Returns a DataFrame with segment_id and time_to_eruption.
        """
        if not os.path.exists(self.config.TRAIN_METADATA_PATH):
            raise FileNotFoundError(
                f"Train metadata not found at {self.config.TRAIN_METADATA_PATH}"
            )
        if not os.path.exists(self.config.VAL_METADATA_PATH):
            raise FileNotFoundError(
                f"Val metadata not found at {self.config.VAL_METADATA_PATH}"
            )

        df_train = pd.read_csv(self.config.TRAIN_METADATA_PATH)
        df_val = pd.read_csv(self.config.VAL_METADATA_PATH)

        # Concatenate
        df_full = pd.concat([df_train, df_val], ignore_index=True)

        # Keep only necessary columns
        return df_full[[self.config.SEGMENT_ID_COL, self.config.TARGET_COL]]

    def train_meta_model(self):
        """
        Loads OOF predictions, aligns them with ground truth, and trains the Ridge regressor.
        Prints the coefficients and the ensemble MAE.
        """
        print("Loading OOF predictions and ground truth...")

        if not os.path.exists(self.tabular_oof_path):
            raise FileNotFoundError(
                f"Tabular OOF file not found at {self.tabular_oof_path}"
            )
        if not os.path.exists(self.vision_oof_path):
            raise FileNotFoundError(
                f"Vision OOF file not found at {self.vision_oof_path}"
            )

        # Load OOFs
        df_tab = pd.read_csv(self.tabular_oof_path)
        df_vis = pd.read_csv(self.vision_oof_path)

        # Rename columns to avoid collision
        df_tab = df_tab.rename(columns={self.config.TARGET_COL: "pred_tabular"})
        df_vis = df_vis.rename(columns={self.config.TARGET_COL: "pred_vision"})

        # Load Ground Truth
        df_gt = self._load_ground_truth()

        # Merge all dataframes on segment_id
        # We use inner join to ensure we only train on segments present in all files
        df_merge = df_gt.merge(df_tab, on=self.config.SEGMENT_ID_COL, how="inner")
        df_merge = df_merge.merge(df_vis, on=self.config.SEGMENT_ID_COL, how="inner")

        print(f"Aligned {len(df_merge)} samples for meta-training.")

        # Prepare X and y
        X = df_merge[["pred_tabular", "pred_vision"]].values
        y = df_merge[self.config.TARGET_COL].values

        # Train Ridge Regression
        print("Training Ridge Meta-Learner...")
        self.model.fit(X, y)

        # Evaluate on the training set (which is the OOF set)
        preds = self.model.predict(X)
        mae = get_score(y, preds)

        # Print results
        print(
            f"Meta-Learner Coefficients: Tabular={self.model.coef_[0]}, Vision={self.model.coef_[1]}"
        )
        print(f"Meta-Learner Intercept: {self.model.intercept_}")
        print(f"Ensemble OOF MAE: {mae}")

        return mae

    def predict(self):
        """
        Loads test predictions from both branches, applies the meta-learner,
        and saves the final submission.
        """
        print("Generating final submission...")

        if not os.path.exists(self.tabular_test_path):
            raise FileNotFoundError(
                f"Tabular Test file not found at {self.tabular_test_path}"
            )
        if not os.path.exists(self.vision_test_path):
            raise FileNotFoundError(
                f"Vision Test file not found at {self.vision_test_path}"
            )

        # Load Test Predictions
        df_tab_test = pd.read_csv(self.tabular_test_path)
        df_vis_test = pd.read_csv(self.vision_test_path)

        # Rename
        df_tab_test = df_tab_test.rename(
            columns={self.config.TARGET_COL: "pred_tabular"}
        )
        df_vis_test = df_vis_test.rename(
            columns={self.config.TARGET_COL: "pred_vision"}
        )

        # Merge
        df_test_merge = df_tab_test.merge(
            df_vis_test, on=self.config.SEGMENT_ID_COL, how="inner"
        )

        if len(df_test_merge) == 0:
            raise ValueError(
                "Merging test predictions resulted in empty DataFrame. Check segment_ids."
            )

        print(f"Predicting for {len(df_test_merge)} test segments.")

        # Prepare X
        X_test = df_test_merge[["pred_tabular", "pred_vision"]].values

        # Predict
        final_preds = self.model.predict(X_test)

        # Ensure non-negative predictions (physically impossible to have negative time)
        final_preds = np.maximum(final_preds, 0)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {
                self.config.SEGMENT_ID_COL: df_test_merge[self.config.SEGMENT_ID_COL],
                self.config.TARGET_COL: final_preds,
            }
        )

        # Save
        save_path = os.path.join(self.config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(save_path, index=False)
        print(f"Final submission saved to {save_path}")
