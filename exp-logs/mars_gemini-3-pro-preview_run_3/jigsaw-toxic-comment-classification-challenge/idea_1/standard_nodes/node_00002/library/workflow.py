import os
import pandas as pd
import numpy as np
from library.utils import set_seed, get_logger
from library.data_processing import load_data, FeatureEngineer
from library.model import MultiLabelNBSVM

# Constants
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = "submission.csv"

logger = get_logger("workflow")


def train_validate(
    load_cached_data: bool = True,
    sample_size: int = None,
    C: float = 1.0,
    max_iter: int = 100,
    max_features_word: int = None,
    max_features_char: int = None,
    dual: bool = False,
    seed: int = 42,
):
    """
    Orchestrates the training and validation pipeline.

    Args:
        load_cached_data (bool): Whether to use cached data/features.
        sample_size (int, optional): Number of samples to use for debugging.
        C (float): Inverse regularization strength for Logistic Regression.
        max_iter (int): Maximum iterations for the solver.
        dual (bool): Prefer dual formulation for LinearSVC/LogisticRegression.
        max_features_word (int, optional): Max features for word vectorizer.
        max_features_char (int, optional): Max features for char vectorizer.
        seed (int): Random seed.

    Returns:
        tuple: (trained_model, fitted_feature_engineer)
    """
    set_seed(seed)

    # 1. Load Data
    # We load the full data first, then slice if debugging.
    # Note: load_data handles its own caching of the merged dataframe.
    train_df = load_data("train", load_cached_data=load_cached_data)
    val_df = load_data("val", load_cached_data=load_cached_data)

    # 2. Handle Subsampling (Debugging)
    cache_suffix_train = "train"
    cache_suffix_val = "val"

    if sample_size is not None:
        logger.info(f"Subsampling datasets to {sample_size} samples for debugging...")
        train_df = train_df.iloc[:sample_size].copy()
        val_df = val_df.iloc[:sample_size].copy()
        # Change cache suffix to avoid overwriting full feature caches with debug ones
        cache_suffix_train = "train_debug"
        cache_suffix_val = "val_debug"

    # 3. Feature Engineering
    # We pass the specific cache suffix determined above.
    fe = FeatureEngineer(
        max_features_word=max_features_word, max_features_char=max_features_char
    )

    logger.info("Generating training features...")
    X_train = fe.fit_transform(
        train_df["comment_text"],
        load_cached_data=load_cached_data,
        cache_suffix=cache_suffix_train,
    )

    logger.info("Generating validation features...")
    X_val = fe.transform(
        val_df["comment_text"],
        load_cached_data=load_cached_data,
        cache_suffix=cache_suffix_val,
    )

    # 4. Prepare Labels
    Y_train = train_df[LABEL_COLS]
    Y_val = val_df[LABEL_COLS]

    # 5. Model Training
    logger.info(
        f"Training MultiLabelNBSVM (C={C}, dual={dual}, max_iter={max_iter})..."
    )
    model = MultiLabelNBSVM(C=C, dual=dual, max_iter=max_iter, random_state=seed)
    model.fit(X_train, Y_train)

    # 6. Validation
    logger.info("Evaluating model...")
    model.score(X_val, Y_val)

    return model, fe


def generate_submission(
    model, feature_engineer, load_cached_data: bool = True, sample_size: int = None
):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model: Trained MultiLabelNBSVM model.
        feature_engineer: Fitted FeatureEngineer instance.
        load_cached_data (bool): Whether to use cached data/features.
        sample_size (int, optional): If set, predicts only on a subset (mostly for testing flow).
    """
    logger.info("Starting submission generation...")

    # 1. Load Test Data
    test_df = load_data("test", load_cached_data=load_cached_data)

    cache_suffix_test = "test"

    if sample_size is not None:
        logger.info(f"Subsampling test set to {sample_size} samples...")
        test_df = test_df.iloc[:sample_size].copy()
        cache_suffix_test = "test_debug"

    # 2. Transform Features
    # Must use the feature engineer fitted on training data
    logger.info("Transforming test features...")
    X_test = feature_engineer.transform(
        test_df["comment_text"],
        load_cached_data=load_cached_data,
        cache_suffix=cache_suffix_test,
    )

    # 3. Predict
    logger.info("Predicting probabilities...")
    preds = model.predict_proba(X_test)

    # 4. Format Submission
    submission = pd.DataFrame(preds, columns=LABEL_COLS)
    submission["id"] = test_df["id"].values

    # Reorder columns: id, toxic, severe_toxic, ...
    cols = ["id"] + LABEL_COLS
    submission = submission[cols]

    # 5. Save
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    out_path = os.path.join(SUBMISSION_DIR, SUBMISSION_FILE)
    submission.to_csv(out_path, index=False)

    logger.info(f"Submission saved to {out_path}")
    logger.info(f"Submission shape: {submission.shape}")

    # Validation check for submission format
    if submission.isnull().any().any():
        logger.warning("Submission contains NaN values!")
