import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from library.config import Config
from library.utils import parameter_aware_cache


class ScoutMiner:
    """
    Implements the Diversity Mining logic to identify Hard Negatives.

    This class uses a trained ensemble of 'Scout' models to scan the set of
    negative examples. Any negative example that is predicted as a positive
    (probability > threshold) by ANY of the scouts is flagged as a 'Hard Negative'.
    These indices are then used to construct the training set for the Expert models.
    """

    def __init__(self, config=Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    @parameter_aware_cache(Config.CACHE_HARD_NEGATIVES, file_format="npy")
    def mine_hard_negatives(self, df, scout_ensemble, load_cached_data=False):
        """
        Identifies hard negatives from the training set using the trained Scout ensemble.

        Args:
            df (pd.DataFrame): The full training dataset (including features and targets).
            scout_ensemble (TriEnsemble): The trained ensemble of Scout models.
            load_cached_data (bool): Whether to load indices from cache if available.

        Returns:
            np.ndarray: An array of indices corresponding to the hard negatives in df.
        """
        self.logger.info("Starting Hard Negative Mining process...")

        # 1. Filter for Negatives only
        # We are exclusively looking for negatives that the models confuse for positives.
        neg_mask = df["contact"] == 0
        df_neg = df[neg_mask]

        if df_neg.empty:
            self.logger.warning("No negative samples found in the provided dataframe.")
            return np.array([])

        self.logger.info(f"Mining from pool of {len(df_neg)} negative samples.")

        # 2. Prepare Features
        # Use the helper from the ensemble to ensure consistent feature selection
        # Accessing protected method _get_feature_cols as we are extending the library logic
        feature_cols = scout_ensemble._get_feature_cols(df_neg)
        X_neg = df_neg[feature_cols]

        # 3. Initialize Hard Negative Mask (Union Logic)
        # Start with all False. If any model says True (Prob > Threshold), it becomes True.
        is_hard_union = np.zeros(len(df_neg), dtype=bool)
        threshold = self.config.SCOUT_PREDICT_THRESHOLD

        if not scout_ensemble.models:
            self.logger.warning(
                "No trained models found in Scout Ensemble. Returning empty indices."
            )
            return np.array([])

        # 4. Iterate through each scout model
        for name, model in scout_ensemble.models.items():
            self.logger.info(f"Scoring negatives with Scout Model: {name}")

            try:
                preds = None

                # Handle model-specific prediction APIs
                if name == "lgbm":
                    # LightGBM Booster.predict returns raw probabilities for binary
                    preds = model.predict(X_neg)

                elif name == "xgb":
                    # XGBoost Booster.predict requires DMatrix
                    dtest = xgb.DMatrix(X_neg)
                    preds = model.predict(dtest)

                elif name == "cat":
                    # CatBoostClassifier.predict_proba returns [p0, p1]
                    preds = model.predict_proba(X_neg)[:, 1]

                else:
                    self.logger.warning(f"Unsupported model type '{name}'. Skipping.")
                    continue

                # Identify candidates from this model
                candidates = preds > threshold
                count = np.sum(candidates)
                self.logger.info(
                    f"  > Scout {name} flagged {count} candidates ({(count/len(df_neg))*100:.2f}%)."
                )

                # Update Union Mask
                is_hard_union = is_hard_union | candidates

            except Exception as e:
                self.logger.error(f"Error predicting with model {name}: {e}")

        # 5. Extract Indices
        # We need the indices from the original dataframe to subset it later in the pipeline
        hard_indices = df_neg.index[is_hard_union].to_numpy()

        self.logger.info(
            f"Mining Complete. Total Hard Negatives found: {len(hard_indices)}"
        )
        self.logger.info(
            f"Hard Negative Rate: {(len(hard_indices)/len(df_neg))*100:.2f}% of negatives."
        )

        return hard_indices
