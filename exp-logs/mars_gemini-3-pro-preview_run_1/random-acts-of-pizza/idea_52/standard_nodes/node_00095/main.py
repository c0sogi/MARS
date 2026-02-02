import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Import configuration and utilities
from library.config import (
    RANDOM_STATE,
    ENSEMBLE_WEIGHTS,
)
from library.utils import set_seed

# Import pipeline components
from library.data_loader import load_data
from library.text_encoder import TextEncoder
from library.feature_engineer import FeatureEngineer
from library.trainer import (
    train_models,
    predict_ensemble,
    save_submission,
    get_mlp_predictions,
)
from library.dataset import PizzaDataset


def main():
    # 1. Setup and Reproducibility
    set_seed(RANDOM_STATE)

    # 2. Load Data
    # Loads train, val, test dataframes from metadata/cache
    print("Loading data...")
    train_df, val_df, test_df = load_data(load_cached_data=True, debug=False)

    # 3. Feature Generation

    # 3.1 Text Embeddings (SBERT)
    # Generates embeddings for Title, Body, and User History
    print("Generating text embeddings (SBERT)...")
    text_encoder = TextEncoder()

    sbert_train = text_encoder.encode(train_df, "train", load_cached_data=True)
    sbert_val = text_encoder.encode(val_df, "val", load_cached_data=True)
    sbert_test = text_encoder.encode(test_df, "test", load_cached_data=True)

    # 3.2 Tabular Features
    # Generates TF-IDF, Metadata, Top-K Flags, Consistency Scalars, and Interactions
    print("Generating tabular features...")
    feature_engineer = FeatureEngineer()

    # Fit on training data (also generates train features)
    # We pass sbert_features to calculate consistency scalars (cosine sim between text and history)
    tabular_train = feature_engineer.generate_features(
        train_df,
        "train",
        sbert_features=sbert_train,
        train_df=train_df,
        load_cached_data=True,
    )

    # Transform validation and test data
    tabular_val = feature_engineer.generate_features(
        val_df, "val", sbert_features=sbert_val, load_cached_data=True
    )

    tabular_test = feature_engineer.generate_features(
        test_df, "test", sbert_features=sbert_test, load_cached_data=True
    )

    # 4. Model Training
    # Trains both the Orthogonal Skip-Gated MLP and Interaction-Enhanced Random Forest
    print("Training models...")
    mlp_model, rf_model = train_models(
        train_df,
        val_df,
        sbert_train,
        sbert_val,
        tabular_train,
        tabular_val,
        save_models=True,
    )

    # 5. Validation Evaluation
    print("\n=== Validation Evaluation ===")

    # Generate predictions for validation set to calculate metrics
    # 5.1 MLP Predictions
    val_dataset = PizzaDataset(sbert_val, tabular_val, labels=None)
    mlp_val_probs = get_mlp_predictions(mlp_model, val_dataset)

    # 5.2 RF Predictions
    rf_val_probs = rf_model.predict_proba(tabular_val)

    # 5.3 Ensemble
    w_rf, w_mlp = ENSEMBLE_WEIGHTS
    total_weight = w_rf + w_mlp
    val_probs = ((w_rf * rf_val_probs) + (w_mlp * mlp_val_probs)) / total_weight

    # 5.4 Calculate Metric
    y_val = val_df["requester_received_pizza"].astype(int).values
    val_auc = roc_auc_score(y_val, val_probs)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate error magnitude
    errors = np.abs(val_probs - y_val)

    # Correlate error with numerical features in validation set
    numeric_cols = val_df.select_dtypes(include=[np.number]).columns
    correlations = {}

    for col in numeric_cols:
        # Skip target and ID columns if they somehow got in
        if col in ["requester_received_pizza", "request_id"]:
            continue

        try:
            # Handle potential NaNs for correlation calculation
            col_data = val_df[col].fillna(0)
            # Check if column has variance
            if col_data.std() > 1e-9:
                corr = np.corrcoef(errors, col_data)[0, 1]
                if not np.isnan(corr):
                    correlations[col] = corr
        except Exception:
            continue

    # Sort correlations
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

    print("Top 5 Features associated with High Error (Positive Correlation):")
    for name, val in sorted_corr[:5]:
        print(f"  {name}: {val}")

    print("Top 5 Features associated with Low Error (Negative Correlation):")
    for name, val in sorted_corr[-5:]:
        print(f"  {name}: {val}")

    # 7. Submission
    THRESHOLD = 0.7135451153926904

    if val_auc > THRESHOLD:
        print(
            f"\nValidation metric {val_auc} exceeds threshold {THRESHOLD}. Generating submission..."
        )
        test_probs = predict_ensemble(
            mlp_model, rf_model, test_df, sbert_test, tabular_test
        )
        save_submission(test_df, test_probs)
    else:
        print(
            f"\nValidation metric {val_auc} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
