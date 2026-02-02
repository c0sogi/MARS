from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler

from library.config import LDA_PARAMS, LOGREG_PARAMS, CALIBRATION_PARAMS


class ExpertFactory:
    """
    Factory class to create un-fitted sklearn estimator objects for the
    Dynamic Ensemble Selection pipeline.
    """

    @staticmethod
    def create_pipeline(preprocessor_type="power", model_type="lda", **kwargs):
        """
        Creates a Pipeline with a preprocessor and a classifier.
        """
        steps = []

        # 1. Preprocessor
        if preprocessor_type == "power":
            steps.append(
                (
                    "preprocessor",
                    PowerTransformer(method="yeo-johnson", standardize=True),
                )
            )
        elif preprocessor_type == "quantile":
            steps.append(
                (
                    "preprocessor",
                    QuantileTransformer(
                        output_distribution="normal", n_quantiles=700, random_state=42
                    ),
                )
            )
        elif preprocessor_type == "standard":
            steps.append(("preprocessor", StandardScaler()))
        else:
            # No preprocessing
            pass

        # 2. Classifier
        if model_type == "lda":
            # Default LDA params from config, can be overridden by kwargs
            params = LDA_PARAMS.copy()
            params.update(kwargs)
            steps.append(("classifier", LinearDiscriminantAnalysis(**params)))
        elif model_type == "logreg":
            base_estimator = LogisticRegression(**LOGREG_PARAMS)
            calibrated_model = CalibratedClassifierCV(
                estimator=base_estimator, **CALIBRATION_PARAMS
            )
            steps.append(("classifier", calibrated_model))

        return Pipeline(steps)
