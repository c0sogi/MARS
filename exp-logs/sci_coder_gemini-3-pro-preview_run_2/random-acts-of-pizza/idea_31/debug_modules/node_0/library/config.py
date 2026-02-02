import os
import json
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, normalize
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import roc_auc_score


# ==========================================
# Configuration
# ==========================================
class Config:
    SEED = 42
    WORKING_DIR = "./working/idea_31"
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    SUBMISSION_DIR = "./submission"

    # Tri-Backbone Architecture
    MODELS = {
        "anchor": "sentence-transformers/all-MiniLM-L6-v2",  # 384d
        "aux1": "sentence-transformers/all-mpnet-base-v2",  # 768d
        "aux2": "sentence-transformers/all-distilroberta-v1",  # 768d
    }

    # Dimensionality Reduction
    PCA_COMPONENTS = 50

    # Training
    N_FOLDS = 5
    N_BAGGING_ESTIMATORS = 20

    # Hyperparameter Grid for Base Learner (Logistic Regression)
    GRID_PARAMS = {
        "C": [0.001, 0.01, 0.1, 1.0, 5.0, 10.0],
        "class_weight": [None, "balanced"],
    }


# ==========================================
# Helper Functions
# ==========================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
    """
    Loads metadata and merges it with raw JSON data.
    Combines Train and Validation metadata for full Cross-Validation.
    """
    print("Loading metadata and raw data...")
    train_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    val_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))
    test_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

    with open(os.path.join(Config.INPUT_DIR, "train.json"), "r") as f:
        train_json = json.load(f)
    with open(os.path.join(Config.INPUT_DIR, "test.json"), "r") as f:
        test_json = json.load(f)

    # Create lookup map
    raw_map = {item["request_id"]: item for item in train_json + test_json}

    def merge(meta_df):
        data = []
        for _, row in meta_df.iterrows():
            rid = row["request_id"]
            item = raw_map[rid]
            merged = item.copy()
            merged.update(row.to_dict())
            data.append(merged)
        return pd.DataFrame(data)

    df_train_part = merge(train_meta)
    df_val_part = merge(val_meta)
    df_test = merge(test_meta)

    # Combine for full CV
    df_full_train = pd.concat([df_train_part, df_val_part], ignore_index=True)

    print(f"Full Train Shape: {df_full_train.shape}")
    print(f"Test Shape: {df_test.shape}")

    return df_full_train, df_test


def get_text_embeddings(df, model_name, cache_prefix, load_cached_data=True):
    """
    Computes or loads cached embeddings for a specific backbone.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    cache_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_{safe_name}.npy")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached embeddings for {model_name} ({cache_prefix})...")
        return np.load(cache_path)

    print(f"Computing embeddings for {model_name} ({cache_prefix})...")

    # Feature Construction: Title + Edit Aware Text
    titles = df["request_title"].fillna("").astype(str)
    texts = df["request_text_edit_aware"].fillna("").astype(str)
    full_texts = (titles + " " + texts).tolist()

    model = SentenceTransformer(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    # Encode
    embeddings = model.encode(
        full_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
    )

    np.save(cache_path, embeddings)
    return embeddings


def get_metadata_features(df):
    """
    Extracts and cleans numerical metadata.
    """
    cols = [
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
    # Fill NaNs with 0 (standard for counts/metrics in this dataset)
    return df[cols].fillna(0).values


# ==========================================
# Main Pipeline
# ==========================================
def main():
    set_seed(Config.SEED)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 1. Load Data
    df_train, df_test = load_data()
    y = df_train["requester_received_pizza"].values

    # 2. Generate Embeddings (Multi-View)
    # We generate raw embeddings first. PCA/Norm happens inside CV to prevent leakage.

    # Train Embeddings
    emb_tr_anchor = get_text_embeddings(
        df_train, Config.MODELS["anchor"], "train_anchor"
    )
    emb_tr_aux1 = get_text_embeddings(df_train, Config.MODELS["aux1"], "train_aux1")
    emb_tr_aux2 = get_text_embeddings(df_train, Config.MODELS["aux2"], "train_aux2")

    # Test Embeddings
    emb_te_anchor = get_text_embeddings(df_test, Config.MODELS["anchor"], "test_anchor")
    emb_te_aux1 = get_text_embeddings(df_test, Config.MODELS["aux1"], "test_aux1")
    emb_te_aux2 = get_text_embeddings(df_test, Config.MODELS["aux2"], "test_aux2")

    # 3. Metadata Extraction
    meta_train = get_metadata_features(df_train)
    meta_test = get_metadata_features(df_test)

    # 4. Cross-Validation & Modeling
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Accumulator for test predictions
    test_preds_accum = np.zeros(len(df_test))
    oof_preds = np.zeros(len(df_train))

    print(f"\nStarting {Config.N_FOLDS}-Fold Stratified CV with Tri-Backbone Fusion...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        print(f"\n--- Fold {fold+1}/{Config.N_FOLDS} ---")

        # A. Split Data
        # Anchor (Raw)
        X_tr_anc = emb_tr_anchor[train_idx]
        X_val_anc = emb_tr_anchor[val_idx]

        # Aux 1 (Raw)
        X_tr_aux1_raw = emb_tr_aux1[train_idx]
        X_val_aux1_raw = emb_tr_aux1[val_idx]

        # Aux 2 (Raw)
        X_tr_aux2_raw = emb_tr_aux2[train_idx]
        X_val_aux2_raw = emb_tr_aux2[val_idx]

        # Metadata (Raw)
        X_tr_meta_raw = meta_train[train_idx]
        X_val_meta_raw = meta_train[val_idx]

        y_tr = y[train_idx]
        y_val = y[val_idx]

        # B. Feature Processing (Fit on Train, Transform Val)

        # 1. PCA for Aux 1
        pca1 = PCA(n_components=Config.PCA_COMPONENTS, random_state=Config.SEED)
        X_tr_aux1 = normalize(pca1.fit_transform(X_tr_aux1_raw))
        X_val_aux1 = normalize(pca1.transform(X_val_aux1_raw))

        # 2. PCA for Aux 2
        pca2 = PCA(n_components=Config.PCA_COMPONENTS, random_state=Config.SEED)
        X_tr_aux2 = normalize(pca2.fit_transform(X_tr_aux2_raw))
        X_val_aux2 = normalize(pca2.transform(X_val_aux2_raw))

        # 3. Normalize Anchor (No PCA, just L2)
        X_tr_anc_norm = normalize(X_tr_anc)
        X_val_anc_norm = normalize(X_val_anc)

        # 4. RankGauss for Metadata
        qt = QuantileTransformer(output_distribution="normal", random_state=Config.SEED)
        X_tr_meta = qt.fit_transform(X_tr_meta_raw)
        X_val_meta = qt.transform(X_val_meta_raw)

        # C. Fusion
        X_tr_final = np.hstack([X_tr_anc_norm, X_tr_aux1, X_tr_aux2, X_tr_meta])
        X_val_final = np.hstack([X_val_anc_norm, X_val_aux1, X_val_aux2, X_val_meta])

        # D. Hyperparameter Tuning (Bagged Logistic Regression)
        best_score = -1
        best_model = None
        best_params = {}

        param_grid = list(ParameterGrid(Config.GRID_PARAMS))

        for params in param_grid:
            # Base Learner
            base_lr = LogisticRegression(
                C=params["C"],
                class_weight=params["class_weight"],
                solver="liblinear",
                random_state=Config.SEED,
            )

            # Bagging Wrapper
            clf = BaggingClassifier(
                estimator=base_lr,
                n_estimators=Config.N_BAGGING_ESTIMATORS,
                random_state=Config.SEED,
                n_jobs=-1,
            )

            clf.fit(X_tr_final, y_tr)
            val_probs = clf.predict_proba(X_val_final)[:, 1]
            score = roc_auc_score(y_val, val_probs)

            if score > best_score:
                best_score = score
                best_model = clf
                best_params = params

        print(f"Best Fold AUC: {best_score:.8f} | Params: {best_params}")

        # E. OOF Predictions
        oof_preds[val_idx] = best_model.predict_proba(X_val_final)[:, 1]

        # F. Test Inference (Using Fold Processors)
        # Transform Test Data
        X_te_aux1 = normalize(pca1.transform(emb_te_aux1))
        X_te_aux2 = normalize(pca2.transform(emb_te_aux2))
        X_te_anc_norm = normalize(emb_te_anchor)
        X_te_meta = qt.transform(meta_test)

        X_te_final = np.hstack([X_te_anc_norm, X_te_aux1, X_te_aux2, X_te_meta])

        # Accumulate
        test_preds_accum += best_model.predict_proba(X_te_final)[:, 1]

    # 5. Final Evaluation & Submission
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\nOverall OOF AUC: {overall_auc:.8f}")

    # Average Test Predictions
    test_preds_avg = test_preds_accum / Config.N_FOLDS

    sub_df = pd.DataFrame(
        {
            "request_id": df_test["request_id"],
            "requester_received_pizza": test_preds_avg,
        }
    )

    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


# Execute
main()
