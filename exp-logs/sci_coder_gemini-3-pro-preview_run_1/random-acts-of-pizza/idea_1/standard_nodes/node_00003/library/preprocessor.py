from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from library.config import NUMERIC_COLS, VECTORIZER_PARAMS


def create_preprocessor(numeric_cols=None):
    """
    Constructs a ColumnTransformer to process numerical and text features.

    Args:
        numeric_cols (list, optional): List of numerical column names to process.
                                       Defaults to NUMERIC_COLS from config.

    Returns:
        ColumnTransformer: The configured preprocessor object ready to be fit.
    """
    # Default to the configuration if no specific columns are provided
    if numeric_cols is None:
        numeric_cols = NUMERIC_COLS

    # 1. Numerical Transformer
    # Impute missing values with the median of the column
    numeric_transformer = SimpleImputer(strategy="median")

    # 2. Text Transformer
    # Apply TF-IDF vectorization to the 'combined_text' column.
    text_transformer = TfidfVectorizer(**VECTORIZER_PARAMS)

    # 3. Combine Transformers
    # Use ColumnTransformer to apply the transformations in parallel.
    # 'remainder="drop"' ensures that ID and Target columns are excluded from the feature matrix.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("text", text_transformer, "combined_text"),
        ],
        remainder="drop",
        n_jobs=-1,  # Use all available cores for transformation
        verbose=False,
    )

    return preprocessor
