import os
import json
import numpy as np
import pandas as pd
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    StandardScaler,
    QuantileTransformer,
    PolynomialFeatures,
    Normalizer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer

# Suppress warnings
warnings.filterwarnings("ignore")


class Config:
    # Directory Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_8"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Input Files
    TRAIN_DATA_PATH = os.path.join(INPUT_DIR, "train.json")
    TEST_DATA_PATH = os.path.join(INPUT_DIR, "test.json")
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Model Hyperparameters
    TRANSFORMER_MODEL = "sentence-transformers/all-mpnet-base-v2"
    N_PCA_COMPONENTS = 256

    # Logistic Regression & Bagging Hyperparameters
    # Using a moderate C to balance signal capture and regularization
    LR_C = 0.5
    LR_MAX_ITER = 2000
    BAGGING_N_ESTIMATORS = 30
    BAGGING_MAX_SAMPLES = 0.7
    BAGGING_MAX_FEATURES = 1.0

    # Random Seed
    RANDOM_SEED = 42

    # Feature Columns
    NUMERIC_COLS = [
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
    ]


def load_metadata(split="train"):
    """Loads the metadata CSV for the specified split."""
    if split == "train":
        return pd.read_csv(Config.TRAIN_META_PATH)
    elif split == "val":
        return pd.read_csv(Config.VAL_META_PATH)
    elif split == "test":
        return pd.read_csv(Config.TEST_META_PATH)
    else:
        raise ValueError(f"Unknown split: {split}")


def get_raw_data(split="train"):
    """
    Loads raw JSON data and filters it based on the metadata split.
    Returns a DataFrame containing the raw data for the requested split.
    """
    df_meta = load_metadata(split)

    # Load appropriate JSON file
    # Note: For efficiency in repeated calls, one might cache the loaded JSON,
    # but here we follow the function scope.

    # Optimization: Load both if needed or just the one required.
    # Since metadata tells us the source, let's load based on that.
    source_files = df_meta["source_file"].unique()

    data_map = {}
    for src in source_files:
        # src is like "input/train.json"
        full_path = os.path.join(".", src)
        if not os.path.exists(full_path):
            # Fallback for relative paths if needed
            full_path = os.path.join(Config.INPUT_DIR, os.path.basename(src))

        with open(full_path, "r") as f:
            raw_list = json.load(f)
            for item in raw_list:
                data_map[item["request_id"]] = item

    # Construct DataFrame
    records = []
    for _, row in df_meta.iterrows():
        rid = row["request_id"]
        if rid in data_map:
            item = data_map[rid].copy()
            # Ensure label is present for train/val
            if "requester_received_pizza" in row:
                item["requester_received_pizza"] = int(row["requester_received_pizza"])
            records.append(item)

    return pd.DataFrame(records)


def get_text_embeddings(df, split, load_cached_data=True):
    """
    Generates or loads MPNet embeddings for the given DataFrame.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, f"{split}_embeddings.npy")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached embeddings for {split} from {cache_path}")
        return np.load(cache_path)

    print(f"Generating MPNet embeddings for {split}...")
    # Prioritize edit_aware text, fallback to raw text
    texts = df["request_text_edit_aware"].fillna("").astype(str).tolist()
    raw_texts = (
        df["request_text"].fillna("").astype(str).tolist()
        if "request_text" in df.columns
        else [""] * len(texts)
    )

    final_texts = []
    for t_edit, t_raw in zip(texts, raw_texts):
        if len(t_edit.strip()) > 0:
            final_texts.append(t_edit)
        else:
            final_texts.append(t_raw)

    model = SentenceTransformer(Config.TRANSFORMER_MODEL)
    embeddings = model.encode(
        final_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
    )

    np.save(cache_path, embeddings)
    return embeddings


def get_metadata_features(df, split, load_cached_data=True):
    """
    Extracts and caches numerical metadata features.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, f"{split}_metadata.npy")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached metadata for {split} from {cache_path}")
        return np.load(cache_path)

    print(f"Extracting metadata features for {split}...")
    # Ensure all columns exist, fill missing with 0
    meta_data = df[Config.NUMERIC_COLS].copy()
    meta_data = meta_data.fillna(0)

    features = meta_data.values.astype(np.float32)
    np.save(cache_path, features)
    return features


class ProjectedEnsemble:
    """
    Implements the Projected Semantic-Interaction Ensemble model.
    """

    def __init__(self):
        self.text_pipeline = None
        self.meta_pipeline = None
        self.classifier = None

    def fit(self, X_text, X_meta, y):
        # 1. Text Branch: PCA -> L2 Normalization
        print("Fitting Text Projection (PCA + L2)...")
        self.text_pipeline = Pipeline(
            [
                (
                    "pca",
                    PCA(
                        n_components=Config.N_PCA_COMPONENTS,
                        random_state=Config.RANDOM_SEED,
                    ),
                ),
                ("l2_norm", Normalizer(norm="l2")),
            ]
        )
        X_text_proj = self.text_pipeline.fit_transform(X_text)

        # 2. Metadata Branch: Quantile -> Interactions -> Scaling
        print("Fitting Metadata Interactions...")
        self.meta_pipeline = Pipeline(
            [
                (
                    "quantile",
                    QuantileTransformer(
                        output_distribution="normal", random_state=Config.RANDOM_SEED
                    ),
                ),
                (
                    "poly",
                    PolynomialFeatures(
                        degree=2, interaction_only=True, include_bias=False
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )
        X_meta_proj = self.meta_pipeline.fit_transform(X_meta)

        # 3. Fusion
        X_combined = np.hstack([X_text_proj, X_meta_proj])

        # 4. Bagged Logistic Regression
        print("Training Bagged Ensemble...")
        base_lr = LogisticRegression(
            class_weight="balanced",
            C=Config.LR_C,
            max_iter=Config.LR_MAX_ITER,
            random_state=Config.RANDOM_SEED,
        )

        self.classifier = BaggingClassifier(
            estimator=base_lr,
            n_estimators=Config.BAGGING_N_ESTIMATORS,
            max_samples=Config.BAGGING_MAX_SAMPLES,
            max_features=Config.BAGGING_MAX_FEATURES,
            random_state=Config.RANDOM_SEED,
            n_jobs=-1,
        )

        self.classifier.fit(X_combined, y)

        # Training Score
        train_probs = self.classifier.predict_proba(X_combined)[:, 1]
        auc = roc_auc_score(y, train_probs)
        print(f"Training ROC AUC: {auc:.15f}")

    def predict_proba(self, X_text, X_meta):
        X_text_proj = self.text_pipeline.transform(X_text)
        X_meta_proj = self.meta_pipeline.transform(X_meta)
        X_combined = np.hstack([X_text_proj, X_meta_proj])
        return self.classifier.predict_proba(X_combined)[:, 1]


def train_and_validate(load_cached_data=True):
    """
    Orchestrates the training and validation process.
    """
    # Load Training Data
    df_train = get_raw_data("train")
    X_text_train = get_text_embeddings(df_train, "train", load_cached_data)
    X_meta_train = get_metadata_features(df_train, "train", load_cached_data)
    y_train = df_train["requester_received_pizza"].values

    # Load Validation Data
    df_val = get_raw_data("val")
    X_text_val = get_text_embeddings(df_val, "val", load_cached_data)
    X_meta_val = get_metadata_features(df_val, "val", load_cached_data)
    y_val = df_val["requester_received_pizza"].values

    # Train Model
    model = ProjectedEnsemble()
    model.fit(X_text_train, X_meta_train, y_train)

    # Validate
    val_probs = model.predict_proba(X_text_val, X_meta_val)
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"Validation ROC AUC: {val_auc:.15f}")

    return model


def generate_submission(model, load_cached_data=True):
    """
    Generates predictions for the test set and saves the submission file.
    """
    df_test = get_raw_data("test")
    X_text_test = get_text_embeddings(df_test, "test", load_cached_data)
    X_meta_test = get_metadata_features(df_test, "test", load_cached_data)

    probs = model.predict_proba(X_text_test, X_meta_test)

    submission = pd.DataFrame(
        {"request_id": df_test["request_id"], "requester_received_pizza": probs}
    )

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
