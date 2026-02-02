import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.covariance import OAS
from library.config import Config


def get_expert_library(debug=False):
    """
    Retrieves the full list of expert configurations for the ensemble library.

    Args:
        debug (bool): If True, initializes the configuration in debug mode.

    Returns:
        list[dict]: A list of dictionaries, each defining an expert's configuration
                    (name, type, view, params).
    """
    config = Config(debug=debug)
    return config.get_expert_library_config()


def create_expert_pipeline(expert_config):
    """
    Constructs the specific machine learning pipeline for a given expert configuration.

    The pipeline consists of:
    1. Gaussianization: PowerTransformer (Yeo-Johnson) to satisfy normality assumptions.
    2. Density Estimation: LDA (with various shrinkage methods) or Regularized QDA.

    Args:
        expert_config (dict): A dictionary containing the expert's 'type' and 'params'.

    Returns:
        sklearn.pipeline.Pipeline: An unfitted pipeline ready for training.
    """
    model_type = expert_config["type"]
    params = expert_config["params"]

    # 1. Preprocessing Step
    # Yeo-Johnson is used to handle potentially negative values (though rare in this dataset)
    # and to strictly enforce the Gaussian distribution assumption of LDA/QDA.
    scaler = PowerTransformer(method="yeo-johnson")

    # 2. Model Instantiation Step
    if model_type == "LDA":
        # Handles 'LDA_Global_Shrinkage_*' and 'LDA_Morph_LedoitWolf'
        # params typically include {'solver': 'lsqr', 'shrinkage': ...}
        estimator = LinearDiscriminantAnalysis(**params)

    elif model_type == "LDA_OAS":
        # Handles 'LDA_Global_OAS'
        # OAS is not directly accessible via the 'shrinkage' string parameter in standard LDA.
        # We inject the OAS covariance estimator class directly.
        estimator = LinearDiscriminantAnalysis(
            solver="lsqr", covariance_estimator=OAS()
        )

    elif model_type == "QDA":
        # Handles 'QDA_Global_Reg_*'
        # params include {'reg_param': ...} to regularize the quadratic boundaries.
        estimator = QuadraticDiscriminantAnalysis(**params)

    else:
        raise ValueError(f"Unknown expert model type: {model_type}")

    # 3. Pipeline Construction
    pipeline = Pipeline([("scaler", scaler), ("model", estimator)])

    return pipeline
