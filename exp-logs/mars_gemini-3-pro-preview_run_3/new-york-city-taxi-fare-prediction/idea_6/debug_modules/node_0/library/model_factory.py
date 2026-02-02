import xgboost as xgb
import lightgbm as lgb
from library.config import Config


class ModelFactory:
    """
    Factory class for initializing machine learning models with
    configurations defined in library.config.Config.
    """

    @staticmethod
    def create_xgboost(**kwargs):
        """
        Creates and configures an XGBoost Regressor.

        Args:
            **kwargs: Arbitrary keyword arguments to override default Config.XGB_PARAMS.
                      Useful for changing n_estimators, learning_rate, etc.

        Returns:
            xgb.XGBRegressor: The configured XGBoost model.
        """
        # Start with default parameters from Config
        params = Config.XGB_PARAMS.copy()

        # Update with any provided overrides
        if kwargs:
            params.update(kwargs)

        return xgb.XGBRegressor(**params)

    @staticmethod
    def create_lgbm(**kwargs):
        """
        Creates and configures a LightGBM Regressor.

        Args:
            **kwargs: Arbitrary keyword arguments to override default Config.LGBM_PARAMS.

        Returns:
            lgb.LGBMRegressor: The configured LightGBM model.
        """
        # Start with default parameters from Config
        params = Config.LGBM_PARAMS.copy()

        # Update with any provided overrides
        if kwargs:
            params.update(kwargs)

        return lgb.LGBMRegressor(**params)
