from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from library.config import Config


class ModelFactory:
    """
    Factory class for creating model instances based on the configuration.
    Isolates model initialization and hyperparameter management.
    """

    @staticmethod
    def get_base_models(**kwargs):
        """
        Instantiates the Layer 1 base models for the stacking ensemble.

        Args:
            **kwargs: Arbitrary keyword arguments to override default Config parameters
                      for any of the models. Keys should match the parameter names
                      of the underlying estimators.

        Returns:
            dict: A dictionary where keys are model identifiers ('lr', 'mnb', 'xgb')
                  and values are the instantiated model objects.
        """
        models = {}

        # 1. Logistic Regression (for Sparse Features)
        # Merge Config params with any overrides provided in kwargs
        lr_params = Config.LR_PARAMS.copy()
        # We filter kwargs to only apply relevant overrides if necessary,
        # but for simplicity in a factory, we often assume the caller knows what they are passing.
        # Here we allow specific overrides if keys match.
        for k, v in kwargs.items():
            if k in lr_params:
                lr_params[k] = v

        models["lr"] = LogisticRegression(**lr_params)

        # 2. Multinomial Naive Bayes (for Sparse Features)
        mnb_params = Config.MNB_PARAMS.copy()
        for k, v in kwargs.items():
            if k in mnb_params:
                mnb_params[k] = v

        models["mnb"] = MultinomialNB(**mnb_params)

        # 3. XGBoost (for Dense SVD Features)
        xgb_params = Config.XGB_PARAMS.copy()
        for k, v in kwargs.items():
            if k in xgb_params:
                xgb_params[k] = v

        models["xgb"] = XGBClassifier(**xgb_params)

        return models

    @staticmethod
    def get_meta_learner(**kwargs):
        """
        Instantiates the Layer 2 meta-learner (blender).

        Args:
            **kwargs: Arbitrary keyword arguments to override default Config parameters.

        Returns:
            sklearn.linear_model.LogisticRegression: The instantiated meta-learner.
        """
        meta_params = Config.META_LR_PARAMS.copy()

        # Apply overrides
        for k, v in kwargs.items():
            if k in meta_params:
                meta_params[k] = v

        return LogisticRegression(**meta_params)
