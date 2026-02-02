import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch

from library.config import Config
from library.utils import set_seed, load_json_data, save_submission
from library.text_encoders import SBERTEmbedder, TFIDFHandler, SentimentAnalyzer
from library.feature_engine import (
    MetadataProcessor,
    HistoryProcessor,
    PrototypeComputer,
)
from library.rf_learner import RFLearner
from library.mlp_learner import MLPLearner


def run():
    # 1. Initialization
    print("Initializing Hybrid Ensemble Pipeline...")
    set_seed(Config.RANDOM_SEED)

    # 2. Load Data
    print("Loading Data...")
    train_df, val_df, test_df = load_json_data(Config, load_cached_data=True)

    y_train = train_df["requester_received_pizza"].astype(int).values
    y_val = val_df["requester_received_pizza"].astype(int).values

    test_ids = test_df["request_id"].values

    # 3. Feature Engineering
    print("Generating Features...")

    # 3.1 SBERT Embeddings (Title, Body, History)
    sbert = SBERTEmbedder()

    # Title
    train_title_emb = sbert.encode_text(train_df, Config.TEXT_COL_TITLE, "train")
    val_title_emb = sbert.encode_text(val_df, Config.TEXT_COL_TITLE, "val")
    test_title_emb = sbert.encode_text(test_df, Config.TEXT_COL_TITLE, "test")

    # Body
    train_body_emb = sbert.encode_text(train_df, Config.TEXT_COL_BODY, "train")
    val_body_emb = sbert.encode_text(val_df, Config.TEXT_COL_BODY, "val")
    test_body_emb = sbert.encode_text(test_df, Config.TEXT_COL_BODY, "test")

    # History (Sequence of embeddings)
    # Note: encode_history returns (embeddings, mask)
    train_hist_emb, train_hist_mask = sbert.encode_history(
        train_df, "requester_subreddits_at_request", "train"
    )
    val_hist_emb, val_hist_mask = sbert.encode_history(
        val_df, "requester_subreddits_at_request", "val"
    )
    test_hist_emb, test_hist_mask = sbert.encode_history(
        test_df, "requester_subreddits_at_request", "test"
    )

    # 3.2 TF-IDF (For RF)
    tfidf_handler = TFIDFHandler()
    train_tfidf, val_tfidf, test_tfidf = tfidf_handler.process(
        train_df, val_df, test_df
    )

    # 3.3 Sentiment (For RF)
    sentiment_analyzer = SentimentAnalyzer()
    train_sentiment = sentiment_analyzer.process(train_df, "train")
    val_sentiment = sentiment_analyzer.process(val_df, "val")
    test_sentiment = sentiment_analyzer.process(test_df, "test")

    # 3.4 Metadata (RF: Raw/Imputed, MLP: Scaled/Arcsinh)
    meta_processor = MetadataProcessor()
    (rf_meta_train, rf_meta_val, rf_meta_test), (
        mlp_meta_train,
        mlp_meta_val,
        mlp_meta_test,
    ) = meta_processor.process(train_df, val_df, test_df)

    # 3.5 History Top-K (For RF)
    hist_processor = HistoryProcessor()
    train_topk, val_topk, test_topk = hist_processor.process(train_df, val_df, test_df)

    # 3.6 Semantic Prototypes (For both)
    # Depends on SBERT embeddings and Train Labels
    proto_computer = PrototypeComputer()
    train_proto, val_proto, test_proto = proto_computer.process(
        train_title_emb,
        train_hist_emb,
        train_hist_mask,
        y_train,  # Using Title as proxy for Request in prototype calc?
        # Actually, the PrototypeComputer.process signature in library/feature_engine.py takes:
        # train_req, train_hist, train_hist_mask, train_y, ...
        # We should probably use the Body embedding or an average of Title+Body for 'Request'.
        # However, looking at the library code, it just takes one request embedding tensor.
        # Let's use Body embedding as it contains the main content, or Title if Body is empty.
        # Given the config uses Title and Body separately in MLP, let's use Body for prototypes as it's richer.
        # Wait, usually prototypes are better on the most informative text. Let's use Body.
        val_body_emb,
        val_hist_emb,
        val_hist_mask,
        test_body_emb,
        test_hist_emb,
        test_hist_mask,
    )
    # Correction: The arguments in the call above were mixed. The function signature is:
    # process(train_req, train_hist, train_hist_mask, train_y, val_req, val_hist, val_hist_mask, test_req, test_hist, test_hist_mask)
    # I will use Body embeddings for the request representation.
    train_proto, val_proto, test_proto = proto_computer.process(
        train_body_emb,
        train_hist_emb,
        train_hist_mask,
        y_train,
        val_body_emb,
        val_hist_emb,
        val_hist_mask,
        test_body_emb,
        test_hist_emb,
        test_hist_mask,
    )

    # 4. Model Training

    # --- Stream A: Random Forest ---
    print("Training Random Forest Stream...")
    rf_train_comps = {
        "tfidf": train_tfidf,
        "metadata": rf_meta_train,
        "top_k": train_topk,
        "prototypes": train_proto,
        "sentiment": train_sentiment,
    }
    rf_val_comps = {
        "tfidf": val_tfidf,
        "metadata": rf_meta_val,
        "top_k": val_topk,
        "prototypes": val_proto,
        "sentiment": val_sentiment,
    }
    rf_test_comps = {
        "tfidf": test_tfidf,
        "metadata": rf_meta_test,
        "top_k": test_topk,
        "prototypes": test_proto,
        "sentiment": test_sentiment,
    }

    rf_learner = RFLearner()
    rf_learner.train(rf_train_comps, y_train, rf_val_comps, y_val)

    rf_val_preds = rf_learner.predict(rf_val_comps)
    rf_test_preds = rf_learner.predict(rf_test_comps)

    # --- Stream B: MLP (Dual-Query) ---
    print("Training MLP Stream...")
    mlp_train_comps = {
        "title_emb": train_title_emb,
        "body_emb": train_body_emb,
        "hist_emb": train_hist_emb,
        "hist_mask": train_hist_mask,
        "metadata": mlp_meta_train,
        "prototypes": train_proto,
    }
    mlp_val_comps = {
        "title_emb": val_title_emb,
        "body_emb": val_body_emb,
        "hist_emb": val_hist_emb,
        "hist_mask": val_hist_mask,
        "metadata": mlp_meta_val,
        "prototypes": val_proto,
    }
    mlp_test_comps = {
        "title_emb": test_title_emb,
        "body_emb": test_body_emb,
        "hist_emb": test_hist_emb,
        "hist_mask": test_hist_mask,
        "metadata": mlp_meta_test,
        "prototypes": test_proto,
    }

    mlp_learner = MLPLearner()
    mlp_learner.train(mlp_train_comps, y_train, mlp_val_comps, y_val)

    mlp_val_preds = mlp_learner.predict(mlp_val_comps)
    mlp_test_preds = mlp_learner.predict(mlp_test_comps)

    # 5. Ensemble & Evaluation
    print("Ensembling...")

    # Weighted Average
    w_rf = Config.ENSEMBLE_WEIGHT_RF
    w_mlp = Config.ENSEMBLE_WEIGHT_MLP

    final_val_preds = (w_rf * rf_val_preds) + (w_mlp * mlp_val_preds)
    final_test_preds = (w_rf * rf_test_preds) + (w_mlp * mlp_test_preds)

    val_auc = roc_auc_score(y_val, final_val_preds)
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude
    errors = np.abs(y_val - final_val_preds)

    # Create a DataFrame for analysis using the numeric metadata features
    # We use the raw/imputed metadata (rf_meta_val) for interpretability
    # The MetadataProcessor outputs numpy arrays, so we need to reconstruct the DF columns for display
    meta_cols = [
        "account_age",
        "days_since_first_post",
        "num_comments",
        "num_comments_raop",
        "num_posts",
        "num_posts_raop",
        "num_subs",
        "up_minus_down",
        "up_plus_down",
        "text_len",
        "word_count",
        "caps_ratio",
    ]
    # Note: rf_meta_val shape matches len(meta_cols)
    analysis_df = pd.DataFrame(rf_meta_val, columns=meta_cols)
    analysis_df["error"] = errors

    # Calculate correlations
    correlations = analysis_df.corr()["error"].sort_values(ascending=False)
    print("Correlation between Error Magnitude and Features:")
    print(correlations.drop("error"))

    # 7. Submission
    threshold = 0.7056961514236341
    if val_auc > threshold:
        print(f"\nValidation metric {val_auc} > {threshold}. Generating submission...")
        save_submission(final_test_preds, test_ids, Config)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(f"\nValidation metric {val_auc} <= {threshold}. Submission skipped.")


if __name__ == "__main__":
    run()
