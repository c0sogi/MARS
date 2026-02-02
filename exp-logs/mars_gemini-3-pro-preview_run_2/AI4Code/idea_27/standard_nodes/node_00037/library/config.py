import os
import json
import gc
import random
import numpy as np
import pandas as pd
import scipy.sparse
from glob import glob
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold
import lightgbm as lgb


# Set fixed seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(42)


class Config:
    # --------------------------------------------------------------------------
    # Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_27"
    SUBMISSION_PATH = "./submission/submission.csv"

    # --------------------------------------------------------------------------
    # Preprocessing & Feature Engineering
    # --------------------------------------------------------------------------
    # TF-IDF Vectorizer Settings
    TFIDF_MAX_FEATURES = 60000
    TFIDF_NGRAM_RANGE = (1, 2)
    TFIDF_SUBLINEAR_TF = True
    TFIDF_USE_IDF = True

    # Truncated SVD Settings
    SVD_COMPONENTS = 128
    SVD_RANDOM_STATE = 42

    # Neighborhood Retrieval
    TOP_K_NEIGHBORS = 10

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------
    # Stage 1: Ridge Regression
    RIDGE_ALPHA = 1.0

    # Stage 2: LightGBM Regressor
    LGBM_PARAMS = {
        "objective": "mae",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 3000,
        "early_stopping_rounds": 50,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }

    # Training Settings
    N_FOLDS = 5

    def __init__(self):
        # Ensure working directories exist
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(self.SUBMISSION_PATH), exist_ok=True)


# ------------------------------------------------------------------------------
# Data Processing Functions
# ------------------------------------------------------------------------------


def read_notebook(filepath):
    """Reads a JSON notebook and returns cell data."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    cell_types = data.get("cell_type", {})
    sources = data.get("source", {})

    # In train data, markdown cells are shuffled at the end.
    # In test data, we assume standard format (mixed) but we treat code as anchors.
    # We return a list of dicts.
    cells = []
    for cell_id, c_type in cell_types.items():
        source = sources.get(cell_id, "")
        cells.append({"cell_id": cell_id, "cell_type": c_type, "source": source})
    return cells


def get_ranks(base, derived):
    """Helper to calculate normalized ranks."""
    return [base.index(d) / (len(base) - 1) for d in derived]


def clean_text(text):
    """Basic text cleaning."""
    return text.lower().strip()


def load_corpus(config, df_meta, mode="train"):
    """
    Loads notebook content.
    Returns:
        df_cells: DataFrame with cell_id, cell_type, source, rank (if train), notebook_id
    """
    cache_path = os.path.join(config.WORKING_DIR, f"{mode}_corpus.parquet")

    # Caching Logic
    if os.path.exists(cache_path):
        print(f"Loading cached {mode} corpus from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Processing {mode} corpus from raw JSONs...")
    data = []

    for _, row in tqdm(
        df_meta.iterrows(), total=len(df_meta), desc=f"Reading {mode} notebooks"
    ):
        nb_id = row["id"]
        filepath = os.path.join(config.INPUT_DIR, row["filepath"])

        cells = read_notebook(filepath)

        # If train, we have ground truth order
        rank_map = {}
        if mode == "train":
            order = row["cell_order"].split()
            rank_map = {cid: i for i, cid in enumerate(order)}
            total_cells = len(order)

        for cell in cells:
            cell_id = cell["cell_id"]
            c_type = cell["cell_type"]
            source = clean_text(cell["source"])

            entry = {
                "id": nb_id,
                "cell_id": cell_id,
                "cell_type": c_type,
                "source": source,
            }

            if mode == "train":
                rank = rank_map.get(cell_id, -1)
                entry["rank"] = rank
                entry["pct_rank"] = rank / (total_cells - 1) if total_cells > 1 else 0.0

            data.append(entry)

    df_cells = pd.DataFrame(data)

    # Save to cache
    df_cells.to_parquet(cache_path, index=False)
    return df_cells


# ------------------------------------------------------------------------------
# Feature Engineering Pipeline
# ------------------------------------------------------------------------------


class FeaturePipeline:
    def __init__(self, config):
        self.config = config
        self.tfidf = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            sublinear_tf=config.TFIDF_SUBLINEAR_TF,
            use_idf=config.TFIDF_USE_IDF,
            strip_accents="unicode",
        )
        self.svd = TruncatedSVD(
            n_components=config.SVD_COMPONENTS, random_state=config.SVD_RANDOM_STATE
        )

    def fit_transform_text(self, df_train_cells):
        print("Fitting TF-IDF on Markdown cells...")
        # Filter for markdown only for vocabulary building
        md_sources = (
            df_train_cells[df_train_cells["cell_type"] == "markdown"]["source"]
            .astype(str)
            .tolist()
        )
        self.tfidf.fit(md_sources)

        print("Fitting SVD on TF-IDF matrix...")
        # Transform a subset or full set to fit SVD
        # To save memory, we can fit SVD on a sample or the sparse matrix directly
        tfidf_mat = self.tfidf.transform(md_sources)
        self.svd.fit(tfidf_mat)

        del md_sources, tfidf_mat
        gc.collect()

    def extract_features(self, df_cells, mode="train"):
        """
        Generates Stage 2 features:
        1. Intrinsic LSA
        2. Lexical Neighbors (TF-IDF)
        3. Latent Neighbors (SVD)
        """
        cache_path = os.path.join(self.config.WORKING_DIR, f"{mode}_features.parquet")
        if os.path.exists(cache_path):
            print(f"Loading cached {mode} features...")
            return pd.read_parquet(cache_path)

        print(f"Generating features for {mode}...")

        # Group by notebook
        nb_groups = df_cells.groupby("id")

        feature_rows = []

        # Pre-transform all text to avoid repeated overhead
        # Note: This might be memory intensive. If OOM, process in chunks.
        # Here we process per notebook to keep logic clean and memory manageable.

        for nb_id, group in tqdm(nb_groups, desc="Feature Extraction"):
            code_cells = group[group["cell_type"] == "code"].reset_index(drop=True)
            md_cells = group[group["cell_type"] == "markdown"].reset_index(drop=True)

            if len(code_cells) == 0 or len(md_cells) == 0:
                # Edge case: no code or no markdown.
                # If no code, we can't rank relative to code. Just return basic features.
                continue

            # 1. Vectorize
            code_sources = code_cells["source"].astype(str).fillna("").tolist()
            md_sources = md_cells["source"].astype(str).fillna("").tolist()

            # TF-IDF
            code_tfidf = self.tfidf.transform(code_sources)
            md_tfidf = self.tfidf.transform(md_sources)

            # SVD
            code_svd = self.svd.transform(code_tfidf)
            md_svd = self.svd.transform(md_tfidf)

            # 2. Compute Similarities (Decoupled Neighborhoods)
            # Lexical (Sparse)
            sim_lex = cosine_similarity(md_tfidf, code_tfidf)
            # Latent (Dense)
            sim_lat = cosine_similarity(md_svd, code_svd)

            # 3. Extract Features per MD cell
            for i in range(len(md_cells)):
                md_row = md_cells.iloc[i]
                row_feat = {
                    "id": nb_id,
                    "cell_id": md_row["cell_id"],
                }

                if mode == "train":
                    row_feat["target"] = md_row["pct_rank"]

                # Metadata
                row_feat["n_code"] = len(code_cells)
                row_feat["n_md"] = len(md_cells)
                row_feat["md_ratio"] = len(md_cells) / (len(code_cells) + len(md_cells))

                # Intrinsic LSA features (first few components)
                for c in range(min(16, self.config.SVD_COMPONENTS)):
                    row_feat[f"lsa_{c}"] = md_svd[i, c]

                # Neighbor Features
                # We look at the code cells that are most similar

                # Helper to get stats
                def get_neighbor_stats(sim_arr, prefix):
                    # argsort gives indices of code cells sorted by similarity (ascending)
                    # we want descending
                    sorted_idx = np.argsort(sim_arr)[::-1]
                    top_k_idx = sorted_idx[: self.config.TOP_K_NEIGHBORS]

                    # Code cell positions (normalized 0..1)
                    # The code cells are anchors, distributed evenly 0..1
                    code_ranks = np.linspace(0, 1, len(code_cells))

                    top_ranks = code_ranks[top_k_idx]
                    top_sims = sim_arr[top_k_idx]

                    # Top-1
                    row_feat[f"{prefix}_top1_rank"] = top_ranks[0]
                    row_feat[f"{prefix}_top1_sim"] = top_sims[0]

                    # Top-K Stats
                    row_feat[f"{prefix}_mean_rank"] = np.mean(top_ranks)
                    row_feat[f"{prefix}_std_rank"] = np.std(top_ranks)
                    row_feat[f"{prefix}_wmean_rank"] = np.average(
                        top_ranks, weights=top_sims + 1e-6
                    )

                get_neighbor_stats(sim_lex[i], "lex")
                get_neighbor_stats(sim_lat[i], "lat")

                feature_rows.append(row_feat)

        df_features = pd.DataFrame(feature_rows)
        df_features.to_parquet(cache_path, index=False)
        return df_features


# ------------------------------------------------------------------------------
# Training & Inference Logic
# ------------------------------------------------------------------------------


def train_model(config):
    # Load Metadata
    df_train_meta = pd.read_csv(os.path.join(config.METADATA_DIR, "train_metadata.csv"))
    df_val_meta = pd.read_csv(os.path.join(config.METADATA_DIR, "val_metadata.csv"))

    # Combine for full training (as per Idea) or keep split.
    # The Idea suggests using full dataset for Stage 1 OOF.
    df_full_meta = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

    # Initialize Pipeline
    pipeline = FeaturePipeline(config)

    # Load Corpus
    df_corpus = load_corpus(config, df_full_meta, mode="train")

    # Fit Vectorizers
    pipeline.fit_transform_text(df_corpus)

    # Generate Stage 2 Features (Intrinsic + Neighbors)
    df_features = pipeline.extract_features(df_corpus, mode="train")

    # --------------------------------------------------------------------------
    # Stage 1: Ridge Regression (Sparse TF-IDF -> Rank)
    # --------------------------------------------------------------------------
    print("Training Stage 1 (Ridge) with 5-Fold CV...")

    # Prepare Sparse Matrix for Ridge
    # We filter df_corpus to match df_features order
    df_features = df_features.merge(
        df_corpus[["cell_id", "source"]], on="cell_id", how="left"
    )

    tfidf_features = pipeline.tfidf.transform(
        df_features["source"].astype(str).tolist()
    )
    y = df_features["target"].values
    groups = df_features["id"].values  # Group by notebook

    # OOF Predictions
    oof_preds = np.zeros(len(df_features))
    gkf = GroupKFold(n_splits=config.N_FOLDS)

    ridge_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_features, y, groups)):
        X_tr, X_val = tfidf_features[train_idx], tfidf_features[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = Ridge(alpha=config.RIDGE_ALPHA)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        # Clip predictions
        preds = np.clip(preds, 0, 1)
        oof_preds[val_idx] = preds

        score = mean_absolute_error(y_val, preds)
        ridge_scores.append(score)
        print(f"Fold {fold+1} Ridge MAE: {score:.5f}")

    print(f"Average Ridge MAE: {np.mean(ridge_scores):.5f}")

    # Add OOF to features
    df_features["ridge_pred"] = oof_preds

    # Retrain Ridge on full data for inference
    final_ridge = Ridge(alpha=config.RIDGE_ALPHA)
    final_ridge.fit(tfidf_features, y)

    # --------------------------------------------------------------------------
    # Stage 2: LightGBM (Stacking)
    # --------------------------------------------------------------------------
    print("Training Stage 2 (LightGBM)...")

    # Prepare Features
    drop_cols = ["id", "cell_id", "target", "source"]
    feature_cols = [c for c in df_features.columns if c not in drop_cols]

    X = df_features[feature_cols]
    y = df_features["target"]

    # Split for LGBM (using GroupKFold again or just a simple split for early stopping)
    # We use one fold for validation to enable early stopping
    train_idx, val_idx = next(gkf.split(X, y, groups))

    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

    model_lgb = lgb.train(
        config.LGBM_PARAMS,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        callbacks=[lgb.log_evaluation(100), lgb.early_stopping(50)],
    )

    return pipeline, final_ridge, model_lgb, feature_cols


def predict_submission(config, pipeline, ridge_model, lgb_model, feature_cols):
    print("Generating predictions for Test Set...")

    # Load Test Data
    df_test_meta = pd.read_csv(os.path.join(config.METADATA_DIR, "test_metadata.csv"))
    df_test_corpus = load_corpus(config, df_test_meta, mode="test")

    # Generate Features
    df_test_features = pipeline.extract_features(df_test_corpus, mode="test")

    # Stage 1: Ridge Inference
    # Need source text again
    df_test_features = df_test_features.merge(
        df_test_corpus[["cell_id", "source"]], on="cell_id", how="left"
    )
    test_tfidf = pipeline.tfidf.transform(
        df_test_features["source"].astype(str).tolist()
    )

    ridge_preds = ridge_model.predict(test_tfidf)
    df_test_features["ridge_pred"] = np.clip(ridge_preds, 0, 1)

    # Stage 2: LGBM Inference
    X_test = df_test_features[feature_cols]
    final_preds = lgb_model.predict(X_test)
    df_test_features["pred_rank"] = final_preds

    # --------------------------------------------------------------------------
    # Post-Processing: Sorting
    # --------------------------------------------------------------------------
    submission = []

    # Group by notebook to reconstruct order
    for nb_id, group in df_test_features.groupby("id"):
        # Get code cells for this notebook
        nb_cells = df_test_corpus[df_test_corpus["id"] == nb_id]
        code_cells = nb_cells[nb_cells["cell_type"] == "code"].copy()

        # Assign fixed ranks to code cells
        n_code = len(code_cells)
        code_cells["pred_rank"] = np.linspace(0, 1, n_code)

        # Get markdown predictions
        md_cells = group[["cell_id", "pred_rank"]].copy()

        # Combine
        combined = pd.concat([code_cells[["cell_id", "pred_rank"]], md_cells])

        # Sort
        combined = combined.sort_values("pred_rank")
        cell_order = " ".join(combined["cell_id"].tolist())

        submission.append({"id": nb_id, "cell_order": cell_order})

    # Save
    df_sub = pd.DataFrame(submission)
    df_sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------


def run_pipeline():
    config = Config()

    # Train
    pipeline, ridge_model, lgb_model, feature_cols = train_model(config)

    # Predict
    predict_submission(config, pipeline, ridge_model, lgb_model, feature_cols)
