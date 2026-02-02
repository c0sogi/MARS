import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from library.data_manager import get_clean_data
from library.feature_engineers import (
    MetadataExtractor,
    TopKSubredditEncoder,
    TextProcessor,
)
from library.utils import Timer, save_submission


class RFPipeline:
    """
    Random Forest Pipeline for Pizza Request Prediction.
    Combines TF-IDF, Metadata, and Top-K Community features into a single dense matrix.
    """

    def __init__(
        self, n_estimators=500, max_depth=None, min_samples_leaf=2, random_state=42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model = None
        self.cache_dir = "./working/idea_25/"

    def get_data(self, load_cached_data=True, debug_mode=False, debug_size=50):
        """
        Prepares the data for the Random Forest model.
        Aggregates features from different extractors and implements caching for the combined matrix.
        """
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Define cache file paths
        files = {
            "X_train": os.path.join(self.cache_dir, "rf_X_train.npy"),
            "y_train": os.path.join(self.cache_dir, "rf_y_train.npy"),
            "X_val": os.path.join(self.cache_dir, "rf_X_val.npy"),
            "y_val": os.path.join(self.cache_dir, "rf_y_val.npy"),
            "X_test": os.path.join(self.cache_dir, "rf_X_test.npy"),
            "test_ids": os.path.join(self.cache_dir, "rf_test_ids.npy"),
        }

        # Determine if we can load from cache
        # We disable cache loading if we are in debug mode to ensure we get the sliced data
        use_cache = load_cached_data and not debug_mode
        all_exist = all(os.path.exists(p) for p in files.values())

        # 1. Load Clean Data First (to establish expected dimensions)
        # We pass use_cache to get_clean_data. If debug_mode is True, get_clean_data handles slicing.
        df_train, df_val, df_test = get_clean_data(
            load_cached_data=load_cached_data,
            debug_mode=debug_mode,
            debug_size=debug_size,
        )

        if use_cache and all_exist:
            print("Loading aggregated RF data from cache...")
            try:
                data = {k: np.load(v, allow_pickle=True) for k, v in files.items()}

                # Validate Dimensions against loaded clean data
                if (
                    len(data["X_train"]) == len(df_train)
                    and len(data["X_val"]) == len(df_val)
                    and len(data["X_test"]) == len(df_test)
                ):
                    return data
                else:
                    print("RF Cache dimensions mismatch. Recomputing...")

            except Exception as e:
                print(f"Failed to load RF cache: {e}. Recomputing...")

        print("Computing RF data from scratch...")
        with Timer("RF Data Aggregation"):
            # 2. Extract Features
            # Note: We pass use_cache to sub-extractors.
            # If debug_mode is True, use_cache is False, forcing them to recompute on sliced data.

            # Metadata
            meta_extractor = MetadataExtractor()
            meta_train, meta_val, meta_test = meta_extractor.process(
                df_train, df_val, df_test, load_cached_data=use_cache
            )

            # Top-K Community Features
            topk_encoder = TopKSubredditEncoder(k=50)
            topk_train, topk_val, topk_test = topk_encoder.process(
                df_train, df_val, df_test, load_cached_data=use_cache
            )

            # TF-IDF Features
            text_processor = TextProcessor()
            tfidf_train, tfidf_val, tfidf_test = text_processor.process_tfidf(
                df_train, df_val, df_test, load_cached_data=use_cache
            )

            # 3. Concatenate Features
            # Convert DataFrames to numpy arrays (float32 for memory efficiency)
            X_meta_train = meta_train.values.astype(np.float32)
            X_meta_val = meta_val.values.astype(np.float32)
            X_meta_test = meta_test.values.astype(np.float32)

            X_topk_train = topk_train.values.astype(np.float32)
            X_topk_val = topk_val.values.astype(np.float32)
            X_topk_test = topk_test.values.astype(np.float32)

            # Stack: TF-IDF + Metadata + TopK
            X_train = np.hstack([tfidf_train, X_meta_train, X_topk_train])
            X_val = np.hstack([tfidf_val, X_meta_val, X_topk_val])
            X_test = np.hstack([tfidf_test, X_meta_test, X_topk_test])

            # 4. Extract Targets and IDs
            y_train = df_train["requester_received_pizza"].astype(int).values
            y_val = df_val["requester_received_pizza"].astype(int).values
            test_ids = df_test["request_id"].values

            data = {
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "X_test": X_test,
                "test_ids": test_ids,
            }

            # 5. Save to Cache (only if not debugging)
            if not debug_mode:
                print("Saving aggregated RF data to cache...")
                for k, v in files.items():
                    np.save(v, data[k])

            print(
                f"RF Data Shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}"
            )
            return data

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains the Random Forest model and evaluates on validation set.
        """
        print(f"Initializing Random Forest (n_estimators={self.n_estimators})...")
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0,
        )

        with Timer("RF Training"):
            self.model.fit(X_train, y_train)

        # Evaluate
        print("Evaluating RF on Validation Set...")
        val_probs = self.model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_probs)
        print(f"Validation AUC: {auc}")

        return auc

    def predict(self, X):
        """
        Generates probability predictions.
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return self.model.predict_proba(X)[:, 1]


def run_rf_pipeline(
    load_cached_data=True, debug_mode=False, debug_size=50, save_output=True
):
    """
    End-to-end execution of the Random Forest pipeline.
    """
    pipeline = RFPipeline(n_estimators=500, min_samples_leaf=2)

    # 1. Get Data
    data = pipeline.get_data(
        load_cached_data=load_cached_data, debug_mode=debug_mode, debug_size=debug_size
    )

    # 2. Train
    val_auc = pipeline.train(
        data["X_train"], data["y_train"], data["X_val"], data["y_val"]
    )

    # 3. Predict on Test
    print("Generating Test Predictions...")
    test_probs = pipeline.predict(data["X_test"])

    # 4. Save Submission
    if save_output:
        save_submission(
            data["test_ids"], test_probs, output_path="./submission/submission.csv"
        )

    return val_auc, test_probs, data["test_ids"]
