import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed, save_submission
from library.feature_engineering import FeatureEngineer
from library.model_rf import train_rf, predict_rf
from library.model_nn import train_nn_model, predict_nn_model


def run():
    # 1. Setup
    set_seed(Config.SEED)

    # 2. Data Processing
    print("Initializing Feature Engineering...")
    fe = FeatureEngineer()
    # Load cached data if available to speed up execution
    data = fe.run(load_cached_data=True)

    # Unpack Data
    # RF Data: (X_train, y_train, X_val, y_val, X_test)
    X_rf_train, y_rf_train, X_rf_val, y_rf_val, X_rf_test = data["rf"]

    # MLP Data: dict with keys 'train', 'val', 'test', 'sub_emb'
    mlp_data = data["mlp"]
    train_data_mlp = mlp_data["train"]  # (title, body, hist, meta, y)
    val_data_mlp = mlp_data["val"]  # (title, body, hist, meta, y)
    test_data_mlp = mlp_data["test"]  # (title, body, hist, meta)
    sub_emb = mlp_data["sub_emb"]

    # IDs for submission
    test_ids = data["ids"]

    # 3. Train Random Forest (Stream A)
    print("\n--- Training Random Forest (Stream A) ---")
    rf_model = train_rf(
        X_rf_train,
        y_rf_train,
        X_rf_val,
        y_rf_val,
        n_estimators=Config.RF_ESTIMATORS,
        min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
        class_weight=Config.RF_CLASS_WEIGHT,
        random_state=Config.SEED,
    )

    # RF Validation Predictions
    rf_val_probs = predict_rf(rf_model, X_rf_val)

    # 4. Train Neural Network (Stream B)
    print("\n--- Training Attention-Gated MLP (Stream B) ---")
    nn_model = train_nn_model(
        train_data_mlp,
        val_data_mlp,
        sub_emb,
        hidden_dim=Config.MLP_HIDDEN_DIM,
        dropout=Config.MLP_DROPOUT,
        lr=Config.MLP_LR,
        epochs=Config.MLP_EPOCHS,
        patience=Config.MLP_PATIENCE,
        batch_size=Config.MLP_BATCH_SIZE,
        seed=Config.SEED,
    )

    # NN Validation Predictions
    # val_data_mlp contains target 'y' at index 4. predict_nn_model expects inputs without y.
    val_features_mlp = val_data_mlp[:4]
    nn_val_probs = predict_nn_model(
        nn_model, val_features_mlp, batch_size=Config.MLP_BATCH_SIZE
    )

    # 5. Ensemble & Evaluation
    print("\n--- Ensembling & Evaluation ---")
    # Simple Weighted Average (0.5 / 0.5)
    ensemble_val_probs = 0.5 * rf_val_probs + 0.5 * nn_val_probs

    # Calculate Metric
    final_auc = roc_auc_score(y_rf_val, ensemble_val_probs)
    print(f"Final Validation Metric: {final_auc}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Load validation metadata for analysis
    if os.path.exists(Config.VAL_PATH):
        df_val = pd.read_csv(Config.VAL_PATH)

        # Calculate Error
        df_val["pred"] = ensemble_val_probs
        df_val["target"] = y_rf_val
        df_val["error"] = np.abs(df_val["target"] - df_val["pred"])

        # Select numerical columns for correlation
        numeric_cols = df_val.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude target, pred, error from features list
        features_to_corr = [
            c
            for c in numeric_cols
            if c not in ["requester_received_pizza", "pred", "error"]
        ]

        # Compute correlations
        correlations = {}
        for col in features_to_corr:
            # Handle potential NaNs for correlation calculation
            correlations[col] = df_val[col].fillna(0).corr(df_val["error"])

        # Sort and print top correlations
        sorted_corr = sorted(
            correlations.items(), key=lambda x: abs(x[1]), reverse=True
        )
        print("Top 5 Features correlated with Error:")
        for name, val in sorted_corr[:5]:
            print(f"{name}: {val:.4f}")
    else:
        print("Validation metadata file not found. Skipping failure analysis.")

    # 7. Submission
    THRESHOLD = 0.6959737721862433
    if final_auc > THRESHOLD:
        print(
            f"\nValidation AUC ({final_auc}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        # RF Test Preds
        rf_test_probs = predict_rf(rf_model, X_rf_test)

        # NN Test Preds
        nn_test_probs = predict_nn_model(
            nn_model, test_data_mlp, batch_size=Config.MLP_BATCH_SIZE
        )

        # Ensemble
        final_test_probs = 0.5 * rf_test_probs + 0.5 * nn_test_probs

        # Save
        save_submission(test_ids, final_test_probs, Config.SUBMISSION_FILE)
    else:
        print(
            f"\nValidation AUC ({final_auc}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    run()
