import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from library.config import CFG


def get_ridge_pipeline(analyzer="word"):
    """
    Creates a Scikit-Learn Pipeline with TfidfVectorizer and Ridge Regression.
    Configures the vectorizer based on the analyzer type (word vs char) using
    settings from the global configuration.

    Args:
        analyzer (str): 'word' or 'char'. Defaults to 'word'.

    Returns:
        sklearn.pipeline.Pipeline: The configured pipeline containing 'tfidf' and 'ridge' steps.
    """
    # Select hyperparameters based on analyzer type
    if analyzer == "word":
        ngram_range = CFG.word_ngram_range
        min_df = CFG.word_min_df
        # Standard sklearn default is \w\w+, which skips 1-letter words.
        # We use \w+ to match the EDA findings (which included 'a', 'I').
        token_pattern = r"(?u)\b\w+\b"
    elif analyzer == "char":
        ngram_range = CFG.char_ngram_range
        min_df = CFG.char_min_df
        # Char analyzer does not use token_pattern
        token_pattern = None
    else:
        raise ValueError(f"Unknown analyzer: {analyzer}. Must be 'word' or 'char'.")

    # Define the vectorizer
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        token_pattern=token_pattern,
        sublinear_tf=True,  # Apply logarithmic scaling to TF (1 + log(tf))
        strip_accents="unicode",  # Normalize characters
        use_idf=True,
        smooth_idf=True,
    )

    # Define the regressor
    regressor = Ridge(
        alpha=CFG.ridge_alpha,
        random_state=CFG.seed,
        solver="auto",  # Let sklearn choose the best solver (often sparse cholesky/lsqr)
    )

    # Construct the pipeline
    pipeline = Pipeline([("tfidf", vectorizer), ("ridge", regressor)])

    return pipeline


def get_lgbm_stacker():
    """
    Creates the LightGBM Regressor for the meta-learning stage.
    Uses parameters defined in CFG.lgbm_params.

    Returns:
        lightgbm.LGBMRegressor: The configured LightGBM model.
    """
    # Unpack parameters from config
    model = lgb.LGBMRegressor(**CFG.lgbm_params)
    return model
