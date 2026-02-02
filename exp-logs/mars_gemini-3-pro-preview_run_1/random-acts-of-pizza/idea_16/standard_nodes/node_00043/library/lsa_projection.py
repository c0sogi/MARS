import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from library import config


class CommunityProjector:
    """
    Projects user subreddit history into a latent semantic space.
    Treats the list of subreddits as a document and applies LSA (TF-IDF + SVD).
    """

    def __init__(self, n_components=20, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.pipeline = None

    def _preprocess(self, subreddit_lists):
        """
        Converts a list of lists of subreddits into a list of space-separated strings.
        Example: [['AskReddit', 'funny'], []] -> ["AskReddit funny", ""]
        """
        processed = []
        for sub_list in subreddit_lists:
            # Handle potential non-list inputs gracefully
            if isinstance(sub_list, list):
                # Join with space. Subreddit names don't contain spaces.
                processed.append(" ".join(str(s) for s in sub_list))
            elif isinstance(sub_list, np.ndarray):
                processed.append(" ".join(str(s) for s in sub_list))
            else:
                processed.append("")
        return processed

    def fit(self, subreddit_lists):
        """
        Fits the TF-IDF + SVD pipeline on the provided subreddit history.
        """
        corpus = self._preprocess(subreddit_lists)

        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        binary=False,  # Use frequency information
                        min_df=2,  # Ignore extremely rare subreddits to reduce noise
                        max_features=10000,  # Cap vocabulary size
                        token_pattern=r"(?u)\b\w+\b",  # Allow single character subreddits if any
                    ),
                ),
                (
                    "svd",
                    TruncatedSVD(
                        n_components=self.n_components,
                        random_state=self.random_state,
                        algorithm="randomized",
                        n_iter=5,
                    ),
                ),
            ]
        )

        self.pipeline.fit(corpus)
        return self

    def transform(self, subreddit_lists):
        """
        Transforms subreddit lists into dense latent vectors.
        """
        if self.pipeline is None:
            raise RuntimeError("Model must be fitted before calling transform.")

        corpus = self._preprocess(subreddit_lists)
        return self.pipeline.transform(corpus)

    def get_feature_names(self):
        return [f"LSA_{i}" for i in range(self.n_components)]


def get_lsa_features(df_train, df_val, df_test, load_cached_data=True):
    """
    Generates or loads Latent Semantic Analysis features for subreddit history.

    Args:
        df_train, df_val, df_test: DataFrames containing 'requester_subreddits_at_request'
        load_cached_data: Boolean to use cached .npy files

    Returns:
        tuple: (train_lsa, val_lsa, test_lsa) as numpy arrays
    """

    # Define cache paths
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_path = os.path.join(cache_dir, "lsa_train.npy")
    val_path = os.path.join(cache_dir, "lsa_val.npy")
    test_path = os.path.join(cache_dir, "lsa_test.npy")

    # Check cache
    if (
        load_cached_data
        and os.path.exists(train_path)
        and os.path.exists(val_path)
        and os.path.exists(test_path)
    ):
        print(f"Loading LSA features from cache: {cache_dir}")
        try:
            train_lsa = np.load(train_path)
            val_lsa = np.load(val_path)
            test_lsa = np.load(test_path)

            # Validate dimensions match current data (Cite debug_lesson_1)
            if (
                train_lsa.shape[0] == len(df_train)
                and val_lsa.shape[0] == len(df_val)
                and test_lsa.shape[0] == len(df_test)
            ):
                return train_lsa, val_lsa, test_lsa
            else:
                print(
                    f"LSA Cache dimension mismatch (Train: {train_lsa.shape[0]} vs {len(df_train)}). Recomputing..."
                )
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    print("Computing LSA features from scratch...")

    col_name = "requester_subreddits_at_request"

    # Extract columns
    # We assume utils.load_data has already parsed these into lists
    train_subs = df_train[col_name].tolist()
    val_subs = df_val[col_name].tolist()
    test_subs = df_test[col_name].tolist()

    # Initialize and Fit Projector
    projector = CommunityProjector(
        n_components=config.LSA_N_COMPONENTS, random_state=config.RANDOM_STATE
    )

    print(f"Fitting LSA on {len(train_subs)} training histories...")
    projector.fit(train_subs)

    # Transform
    print("Transforming datasets...")
    train_lsa = projector.transform(train_subs)
    val_lsa = projector.transform(val_subs)
    test_lsa = projector.transform(test_subs)

    # Save to cache
    print("Saving LSA features to cache...")
    np.save(train_path, train_lsa)
    np.save(val_path, val_lsa)
    np.save(test_path, test_lsa)

    return train_lsa, val_lsa, test_lsa
