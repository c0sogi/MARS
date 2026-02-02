import os
import json
import random
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sentence_transformers import SentenceTransformer

# Define constants
CACHE_DIR = "./working/idea_34"
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_submission(
    request_ids, probabilities, output_path="./submission/submission.csv"
):
    """
    Saves the predictions to a CSV file in the required format.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(
        {"request_id": request_ids, "requester_received_pizza": probabilities}
    )
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


class FeatureLoader:
    """
    Handles loading raw data, computing/caching embeddings, and extracting metadata.
    """

    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def _get_text_data(self, data):
        texts = []
        for entry in data:
            title = entry.get("request_title", "")
            # Use edit aware text if available, else standard text
            body = entry.get("request_text_edit_aware", entry.get("request_text", ""))
            # Concatenate title and body
            full_text = f"{title} {body}".strip()
            texts.append(full_text)
        return texts

    def _get_metadata_features(self, data):
        # Numerical features based on analysis
        num_cols = [
            "requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_comments_in_raop_at_request",
            "requester_number_of_posts_at_request",
            "requester_number_of_posts_on_raop_at_request",
            "requester_number_of_subreddits_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request",
            "unix_timestamp_of_request",
        ]

        features = []
        for entry in data:
            row = [float(entry.get(col, 0)) for col in num_cols]
            features.append(row)
        return np.array(features, dtype=np.float32)

    def _compute_embeddings(self, texts, model_name, cache_name, load_cached):
        cache_path = os.path.join(self.cache_dir, f"{cache_name}.npy")

        if load_cached and os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}")
            return np.load(cache_path)

        print(f"Computing embeddings with {model_name}...")
        model = SentenceTransformer(model_name)
        if torch.cuda.is_available():
            model = model.to("cuda")

        embeddings = model.encode(
            texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )

        # Normalize immediately (L2) as per JBPCE strategy
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        embeddings = embeddings / norms

        np.save(cache_path, embeddings)
        return embeddings

    def load_features(self, load_cached=True):
        # Load Raw Data
        train_raw = self._load_json(os.path.join(INPUT_DIR, "train.json"))
        test_raw = self._load_json(os.path.join(INPUT_DIR, "test.json"))

        # Process Text
        train_texts = self._get_text_data(train_raw)
        test_texts = self._get_text_data(test_raw)

        # Process Metadata
        train_meta = self._get_metadata_features(train_raw)
        test_meta = self._get_metadata_features(test_raw)

        # Extract IDs
        train_ids = [entry["request_id"] for entry in train_raw]
        test_ids = [entry["request_id"] for entry in test_raw]

        # Embeddings - Backbone A (MiniLM)
        train_emb_a = self._compute_embeddings(
            train_texts, "all-MiniLM-L6-v2", "train_emb_minilm", load_cached
        )
        test_emb_a = self._compute_embeddings(
            test_texts, "all-MiniLM-L6-v2", "test_emb_minilm", load_cached
        )

        # Embeddings - Backbone B (MPNet)
        train_emb_b = self._compute_embeddings(
            train_texts, "all-mpnet-base-v2", "train_emb_mpnet", load_cached
        )
        test_emb_b = self._compute_embeddings(
            test_texts, "all-mpnet-base-v2", "test_emb_mpnet", load_cached
        )

        # Concatenate Embeddings (A + B) for Joint PCA
        train_emb_full = np.hstack([train_emb_a, train_emb_b])
        test_emb_full = np.hstack([test_emb_a, test_emb_b])

        # Map to dict for easy retrieval by ID via metadata files
        data_map = {}
        for i, rid in enumerate(train_ids):
            label = int(train_raw[i].get("requester_received_pizza", 0))
            data_map[rid] = {
                "emb": train_emb_full[i],
                "meta": train_meta[i],
                "y": label,
            }

        test_map = {}
        for i, rid in enumerate(test_ids):
            test_map[rid] = {"emb": test_emb_full[i], "meta": test_meta[i]}

        return data_map, test_map, train_emb_full.shape[1]


def get_splits(data_map, test_map):
    """
    Reconstructs train/val/test arrays based on metadata CSVs.
    Merges train and val for full cross-validation.
    """
    df_train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    df_val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    df_test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    def build_arrays(df, source_map, is_test=False):
        X_emb = []
        X_meta = []
        y = []
        ids = []

        for _, row in df.iterrows():
            rid = row["request_id"]
            if rid in source_map:
                item = source_map[rid]
                X_emb.append(item["emb"])
                X_meta.append(item["meta"])
                ids.append(rid)
                if not is_test:
                    y.append(item["y"])

        return (
            np.array(X_emb),
            np.array(X_meta),
            np.array(y) if not is_test else None,
            ids,
        )

    X_train_emb, X_train_meta, y_train, _ = build_arrays(df_train_meta, data_map)
    X_val_emb, X_val_meta, y_val, _ = build_arrays(df_val_meta, data_map)
    X_test_emb, X_test_meta, _, test_ids = build_arrays(
        df_test_meta, test_map, is_test=True
    )

    # Combine Train and Val for 5-Fold Stratified CV
    X_full_emb = np.vstack([X_train_emb, X_val_emb])
    X_full_meta = np.vstack([X_train_meta, X_val_meta])
    y_full = np.hstack([y_train, y_val])

    return X_full_emb, X_full_meta, y_full, X_test_emb, X_test_meta, test_ids


def train_and_predict(X_emb, X_meta, y, X_test_emb, X_test_meta, emb_dim):
    """
    Implements the JBPCE strategy:
    1. Joint PCA on concatenated embeddings.
    2. RankGauss on metadata.
    3. Bagging Logistic Regression.
    4. Grid Search and CV-Bagging inference.
    """
    # Combine Emb and Meta into single matrix for ColumnTransformer
    X_full = np.hstack([X_emb, X_meta])
    X_test_full = np.hstack([X_test_emb, X_test_meta])

    # Define Preprocessing Pipeline
    # Branch 1: Embeddings (Cols 0 to emb_dim) -> PCA -> L2 Norm
    # Branch 2: Metadata (Cols emb_dim to end) -> QuantileTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "emb_pca",
                Pipeline([("pca", PCA(random_state=42)), ("norm", Normalizer())]),
                slice(0, emb_dim),
            ),
            (
                "meta_trans",
                QuantileTransformer(output_distribution="normal", random_state=42),
                slice(emb_dim, X_full.shape[1]),
            ),
        ]
    )

    # Full Pipeline
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "clf",
                BaggingClassifier(
                    estimator=LogisticRegression(solver="liblinear", random_state=42),
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )

    # Hyperparameter Grid
    param_grid = {
        "preprocessor__emb_pca__pca__n_components": [100, 150, 200],
        "clf__estimator__C": [0.1, 1.0, 10.0],
        "clf__estimator__class_weight": ["balanced", None],
        "clf__n_estimators": [20],
        "clf__max_samples": [1.0],
        "clf__bootstrap": [True],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Starting Grid Search with CV...")
    grid = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=10, verbose=1
    )

    grid.fit(X_full, y)

    print(f"Best CV AUC: {grid.best_score_:.6f}")
    print(f"Best Params: {grid.best_params_}")

    # CV-Bagging Inference: Train 5 ensembles (one per fold) using best params and average predictions
    print("Executing CV-Bagging Inference...")
    test_preds_sum = np.zeros(len(X_test_full))
    best_params = grid.best_params_

    for fold, (train_idx, _) in enumerate(cv.split(X_full, y)):
        X_train_fold, y_train_fold = X_full[train_idx], y[train_idx]

        # Create fresh pipeline with best params
        # We must manually reconstruct to ensure clean state
        fold_preprocessor = ColumnTransformer(
            transformers=[
                (
                    "emb_pca",
                    Pipeline(
                        [
                            (
                                "pca",
                                PCA(
                                    n_components=best_params[
                                        "preprocessor__emb_pca__pca__n_components"
                                    ],
                                    random_state=42,
                                ),
                            ),
                            ("norm", Normalizer()),
                        ]
                    ),
                    slice(0, emb_dim),
                ),
                (
                    "meta_trans",
                    QuantileTransformer(output_distribution="normal", random_state=42),
                    slice(emb_dim, X_full.shape[1]),
                ),
            ]
        )

        fold_clf = BaggingClassifier(
            estimator=LogisticRegression(
                solver="liblinear",
                C=best_params["clf__estimator__C"],
                class_weight=best_params["clf__estimator__class_weight"],
                random_state=42,
            ),
            n_estimators=best_params["clf__n_estimators"],
            max_samples=best_params["clf__max_samples"],
            bootstrap=best_params["clf__bootstrap"],
            random_state=42 + fold,  # Vary seed for diversity across folds
            n_jobs=1,
        )

        fold_pipeline = Pipeline(
            [("preprocessor", fold_preprocessor), ("clf", fold_clf)]
        )

        fold_pipeline.fit(X_train_fold, y_train_fold)
        fold_probs = fold_pipeline.predict_proba(X_test_full)[:, 1]
        test_preds_sum += fold_probs

    avg_probs = test_preds_sum / cv.get_n_splits()
    return avg_probs


def run_idea_34(load_cached_data=True):
    """
    Main execution function for Idea 34.
    """
    set_seed(42)

    print("Loading and processing data...")
    loader = FeatureLoader()
    data_map, test_map, emb_dim = loader.load_features(load_cached=load_cached_data)

    print("Preparing data splits...")
    X, X_meta, y, X_test, X_test_meta, test_ids = get_splits(data_map, test_map)

    print(f"Training Data Shape: Embeddings={X.shape}, Metadata={X_meta.shape}")

    probs = train_and_predict(X, X_meta, y, X_test, X_test_meta, emb_dim)

    save_submission(test_ids, probs)
