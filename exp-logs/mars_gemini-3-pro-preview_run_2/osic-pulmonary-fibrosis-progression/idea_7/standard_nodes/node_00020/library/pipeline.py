import os
from library.config import SEED
from library.utils import seed_everything
from library.feature_generation import run_feature_generation
from library.preprocessing import run_preprocessing
from library.modeling import train_models, generate_submission


def train_pipeline(load_cached_data=True):
    """
    Orchestrates the training process for the Spatially-Aware Hybrid-Feature Quantile-GLM Pipeline.

    Steps:
    1. Feature Generation: Extracts 3-zone EfficientNet texture features and lung volumes.
    2. Preprocessing: Merges image/tabular data, fits PCA (on train), and creates interaction terms.
    3. Modeling: Trains Median Regressor (FVC) and Gamma GLM (Uncertainty).

    Args:
        load_cached_data (bool): If True, attempts to load intermediate features from disk.

    Returns:
        tuple: (fvc_model, unc_model, data_dict)
            - fvc_model: Trained MedianRegressor instance.
            - unc_model: Trained UncertaintyGLM instance.
            - data_dict: Dictionary containing preprocessed training, validation, and test arrays.
    """
    # 1. Set Seed for Reproducibility
    seed_everything(SEED)

    # 2. Feature Generation (Image Processing)
    # Extracts raw image features or loads them from cache.
    # Returns dictionaries: {PatientID: np.array}
    train_image_feats, val_image_feats, test_image_feats = run_feature_generation(
        load_cached_data=load_cached_data
    )

    # 3. Preprocessing (Tabular + Fusion + Feature Engineering)
    # Combines image features with clinical data, scales, and creates specific feature sets
    # for FVC (interactions) and Uncertainty (horizon) models.
    data_dict = run_preprocessing(
        train_image_feats,
        val_image_feats,
        test_image_feats,
        load_cached_data=load_cached_data,
    )

    # 4. Model Training
    # Trains the models and prints validation metrics (MAE and Laplace Log Likelihood).
    fvc_model, unc_model = train_models(data_dict)

    return fvc_model, unc_model, data_dict


def inference_pipeline(fvc_model, unc_model, data_dict):
    """
    Orchestrates the inference process and submission generation.

    Steps:
    1. Extracts preprocessed test features.
    2. Generates predictions for FVC (Median) and Confidence (Scaled Delta).
    3. Saves the results to submission.csv.

    Args:
        fvc_model: Trained MedianRegressor instance.
        unc_model: Trained UncertaintyGLM instance.
        data_dict: Dictionary containing preprocessed data (must contain 'X_fvc_test' and 'X_unc_test').
    """
    # Extract test feature sets
    X_fvc_test = data_dict["X_fvc_test"]
    X_unc_test = data_dict["X_unc_test"]

    # Generate and save submission
    # This function handles prediction, scaling (sigma = delta * sqrt(2)), and file I/O.
    generate_submission(fvc_model, unc_model, X_fvc_test, X_unc_test)
