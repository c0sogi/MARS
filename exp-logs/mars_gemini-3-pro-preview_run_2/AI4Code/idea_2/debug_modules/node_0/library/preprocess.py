import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    VOCAB_SIZE_CODE,
    MAX_CODE_TOKENS,
    SEED,
    DEBUG_SAMPLE_SIZE,
)
from library.utils import read_notebook, seed_everything

seed_everything(SEED)


class CodeContextExtractor:
    """
    Extracts high-value technical keywords from code cells using TF-IDF.
    """

    def __init__(self, max_features=VOCAB_SIZE_CODE, max_tokens=MAX_CODE_TOKENS):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            token_pattern=r"(?u)\b\w\w+\b",  # Ignore single characters
            stop_words="english",
        )
        self.max_tokens = max_tokens
        self.is_fitted = False

    def fit(self, corpus_iterable):
        """
        Fits the TF-IDF vectorizer on a generator of code strings.
        """
        self.vectorizer.fit(corpus_iterable)
        self.is_fitted = True
        return self

    def save(self, path):
        joblib.dump(self.vectorizer, path)

    def load(self, path):
        self.vectorizer = joblib.load(path)
        self.is_fitted = True
        return self

    def extract(self, code_text):
        """
        Transforms a code string into a space-separated string of top-k keywords.
        """
        if not self.is_fitted:
            raise RuntimeError("CodeContextExtractor must be fitted before extraction.")

        if not code_text or not code_text.strip():
            return ""

        # Transform returns a sparse matrix (1, n_features)
        tfidf_matrix = self.vectorizer.transform([code_text])
        feature_names = self.vectorizer.get_feature_names_out()

        # Get indices sorted by score descending
        coo = tfidf_matrix.tocoo()
        # Sort by (score, index) descending
        sorted_items = sorted(
            zip(coo.col, coo.data), key=lambda x: (x[1], x[0]), reverse=True
        )

        # Select top K tokens
        top_indices = [idx for idx, score in sorted_items[: self.max_tokens]]
        top_keywords = [feature_names[idx] for idx in top_indices]

        return " ".join(top_keywords)


def get_notebook_code_text(nb_json):
    """
    Aggregates all code cells in a notebook into a single string.
    """
    if "cell_type" not in nb_json or "source" not in nb_json:
        return ""

    code_cells = []
    # Iterate through cells; order here is not strictly guaranteed to be execution order
    # in the JSON dict, but for Bag-of-Words context, it is acceptable.
    for cell_id, cell_type in nb_json["cell_type"].items():
        if cell_type == "code":
            source = nb_json["source"].get(cell_id, "")
            code_cells.append(source)

    return " ".join(code_cells)


def create_training_dataframe(load_cached_data=True, debug=False):
    """
    Generates or loads the training and validation dataframes.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.
        debug (bool): If True, limits dataset size using DEBUG_SAMPLE_SIZE.

    Returns:
        tuple: (df_train, df_val)
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    train_cache_path = os.path.join(WORKING_DIR, "train_dataframe.parquet")
    val_cache_path = os.path.join(WORKING_DIR, "val_dataframe.parquet")
    vectorizer_path = os.path.join(WORKING_DIR, "tfidf_vectorizer.joblib")

    # 1. Try Loading Cache
    if (
        load_cached_data
        and os.path.exists(train_cache_path)
        and os.path.exists(val_cache_path)
    ):
        print(f"Loading cached training dataframes from {WORKING_DIR}...")
        df_train = pd.read_parquet(train_cache_path)
        df_val = pd.read_parquet(val_cache_path)
        return df_train, df_val

    print("Generating training dataframes from scratch...")

    # 2. Load Metadata
    df_train_meta = pd.read_csv(TRAIN_METADATA_PATH)
    df_val_meta = pd.read_csv(VAL_METADATA_PATH)

    # Apply Debug Sampling
    sample_size = DEBUG_SAMPLE_SIZE if debug else None
    if sample_size is not None:
        print(f"Debug mode: Sampling {sample_size} notebooks.")
        df_train_meta = df_train_meta.iloc[:sample_size]
        df_val_meta = df_val_meta.iloc[:sample_size]

    # 3. Fit or Load Extractor
    extractor = CodeContextExtractor(
        max_features=VOCAB_SIZE_CODE, max_tokens=MAX_CODE_TOKENS
    )

    if load_cached_data and os.path.exists(vectorizer_path):
        print("Loading cached TF-IDF vectorizer...")
        extractor.load(vectorizer_path)
    else:
        print("Fitting TF-IDF vectorizer on training code cells...")

        # Generator to avoid loading all code into memory
        def train_corpus_generator():
            for _, row in df_train_meta.iterrows():
                nb = read_notebook(row["filepath"])
                yield get_notebook_code_text(nb)

        extractor.fit(train_corpus_generator())
        extractor.save(vectorizer_path)

    # 4. Process Splits
    def process_metadata(df_meta, desc):
        print(f"Processing {desc}...")
        data = []
        for _, row in df_meta.iterrows():
            nb_id = row["id"]
            filepath = row["filepath"]
            cell_order_str = row["cell_order"]

            nb = read_notebook(filepath)
            if not nb:
                continue

            # Extract Context
            code_text = get_notebook_code_text(nb)
            context_summary = extractor.extract(code_text)

            # Calculate Rank Targets
            correct_order = cell_order_str.split()
            total_cells = len(correct_order)
            # Map cell_id -> normalized rank [0, 1]
            rank_map = {
                cid: i / (total_cells - 1) if total_cells > 1 else 0.0
                for i, cid in enumerate(correct_order)
            }

            # Extract Markdown Cells
            for cell_id, c_type in nb.get("cell_type", {}).items():
                if c_type == "markdown":
                    source = nb.get("source", {}).get(cell_id, "")
                    # Only include if cell is in the ground truth order
                    if cell_id in rank_map:
                        data.append(
                            {
                                "id": nb_id,
                                "cell_id": cell_id,
                                "text": source,
                                "context": context_summary,
                                "rank": rank_map[cell_id],
                            }
                        )

        df = pd.DataFrame(data)
        if not df.empty:
            df["rank"] = df["rank"].astype(np.float32)
        return df

    df_train = process_metadata(df_train_meta, "train set")
    df_val = process_metadata(df_val_meta, "validation set")

    # 5. Save to Cache
    print("Saving processed dataframes to parquet...")
    df_train.to_parquet(train_cache_path, index=False)
    df_val.to_parquet(val_cache_path, index=False)

    return df_train, df_val


def transform_notebook_features(df_test_meta, load_cached_data=True):
    """
    Applies the fitted feature extraction to the test set.

    Args:
        df_test_meta (pd.DataFrame): Metadata for the test set.
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: Processed test dataframe with 'text' and 'context'.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(WORKING_DIR, "test_dataframe.parquet")
    vectorizer_path = os.path.join(WORKING_DIR, "tfidf_vectorizer.joblib")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached test dataframe from {WORKING_DIR}...")
        return pd.read_parquet(cache_path)

    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            f"Vectorizer not found at {vectorizer_path}. Please run training preparation first."
        )

    print("Generating test dataframe...")

    # Load Extractor
    extractor = CodeContextExtractor(
        max_features=VOCAB_SIZE_CODE, max_tokens=MAX_CODE_TOKENS
    )
    extractor.load(vectorizer_path)

    data = []
    for _, row in df_test_meta.iterrows():
        nb_id = row["id"]
        filepath = row["filepath"]

        nb = read_notebook(filepath)
        if not nb:
            continue

        # Extract Context
        code_text = get_notebook_code_text(nb)
        context_summary = extractor.extract(code_text)

        # Extract Markdown Cells (No rank target for test)
        for cell_id, c_type in nb.get("cell_type", {}).items():
            if c_type == "markdown":
                source = nb.get("source", {}).get(cell_id, "")
                data.append(
                    {
                        "id": nb_id,
                        "cell_id": cell_id,
                        "text": source,
                        "context": context_summary,
                    }
                )

    df_test = pd.DataFrame(data)

    print("Saving test dataframe to parquet...")
    df_test.to_parquet(cache_path, index=False)

    return df_test
