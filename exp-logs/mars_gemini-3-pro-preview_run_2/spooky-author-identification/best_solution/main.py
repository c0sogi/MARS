import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import warnings

# Import provided library components
from library.utils import set_seed, compute_log_loss, get_device
from library.data_loader import load_and_process_data
from library.feature_extractor import ClassicalFeaturePipeline, NeuralFeaturePipeline
from library.stacking_manager import StackingEnsemble

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
CONFIG = {
    "seed": 42,
    "n_folds": 5,
    "debug": False,  # Use full data to achieve the target metric
    # Classical Model Hyperparameters
    "svd_n_components": 100,
    "lr_C": 1.0,
    "nb_alpha": 0.01,
    "xgb_n_estimators": 200,
    "xgb_max_depth": 6,
    "xgb_lr": 0.05,
    # Neural Model Hyperparameters
    "transformer_model": "microsoft/deberta-v3-large",
    "max_length": 80,
    "batch_size": 16,
    "epochs": 3,
    "learning_rate": 1e-5,
    "patience": 1,
}


def main():
    # 1. Initialization
    set_seed(CONFIG["seed"])
    device = get_device()

    # 2. Data Loading
    # load_and_process_data merges metadata/train.csv and metadata/val.csv for CV
    train_df, test_df, label_classes = load_and_process_data(
        CONFIG, load_cached_data=True
    )
    y = train_df["author_encoded"].values
    folds = train_df["fold"].values

    # Identify the specific hold-out validation set indices
    # We read the metadata file directly to get the IDs
    val_meta_path = "./metadata/val.csv"
    if os.path.exists(val_meta_path):
        val_meta_df = pd.read_csv(val_meta_path)
        val_ids = set(val_meta_df["id"].values)
        is_val_sample = train_df["id"].isin(val_ids).values
    else:
        # Fallback if metadata missing (should not happen per instructions)
        is_val_sample = np.zeros(len(train_df), dtype=bool)

    # 3. Feature Extraction
    classical_pipe = ClassicalFeaturePipeline(CONFIG)
    X_train_sparse, X_train_dense, X_test_sparse, X_test_dense = classical_pipe.execute(
        train_df["text"], test_df["text"], load_cached_data=True
    )

    neural_pipe = NeuralFeaturePipeline(CONFIG)
    X_train_neural, X_test_neural = neural_pipe.execute(
        train_df["text"], test_df["text"], load_cached_data=True
    )

    # 4. Level 1 Model Training (Generate OOF Predictions)
    ensemble = StackingEnsemble(CONFIG)
    level1_oof_preds = []
    level1_test_preds = []

    # Define base models
    models = [
        ("lr", "lr", X_train_sparse, X_test_sparse, False),
        ("nb", "nb", X_train_sparse, X_test_sparse, False),
        ("xgb", "xgb", X_train_dense, X_test_dense, False),
        ("transformer", "transformer", X_train_neural, X_test_neural, True),
    ]

    for name, model_type, X_tr_full, X_te_full, is_neural in models:
        oof, test_pred = ensemble._get_cv_predictions(
            name, model_type, X_tr_full, y, folds, X_te_full, label_classes, is_neural
        )
        level1_oof_preds.append(oof)
        level1_test_preds.append(test_pred)

    # 5. Level 2 Meta-Learner Training
    X_meta_train = np.hstack(level1_oof_preds)
    X_meta_test = np.hstack(level1_test_preds)

    meta_learner = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        multi_class="multinomial",
        random_state=CONFIG["seed"],
        max_iter=1000,
    )

    # 6. Validation Assessment
    # Split OOF predictions into Meta-Train and Meta-Val to avoid leakage
    X_meta_tr_sub = X_meta_train[~is_val_sample]
    y_meta_tr_sub = y[~is_val_sample]
    X_meta_val_sub = X_meta_train[is_val_sample]
    y_meta_val_sub = y[is_val_sample]

    # Train on subset to get valid validation metric
    meta_learner.fit(X_meta_tr_sub, y_meta_tr_sub)
    val_probs = meta_learner.predict_proba(X_meta_val_sub)
    val_labels = y_meta_val_sub

    # Compute metric
    final_val_metric = compute_log_loss(val_labels, val_probs)
    print(f"Final Validation Metric: {final_val_metric}")

    # Refit on full data for submission
    meta_learner.fit(X_meta_train, y)

    # 7. Failure Analysis
    # Replicate rescaling and clipping for analysis consistency
    row_sums = val_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    val_probs_norm = val_probs / row_sums

    eps = 1e-15
    val_probs_clipped = np.clip(val_probs_norm, eps, 1 - eps)

    # Calculate per-sample log loss
    true_class_probs = val_probs_clipped[np.arange(len(val_labels)), val_labels]
    sample_losses = -np.log(true_class_probs)

    # Calculate features for correlation
    val_texts = train_df.loc[is_val_sample, "text"].values
    val_word_counts = np.array([len(str(t).split()) for t in val_texts])
    val_char_counts = np.array([len(str(t)) for t in val_texts])

    corr_word = np.corrcoef(sample_losses, val_word_counts)[0, 1]
    corr_char = np.corrcoef(sample_losses, val_char_counts)[0, 1]

    print(f"Correlation (Loss vs Word Count): {corr_word}")
    print(f"Correlation (Loss vs Char Count): {corr_char}")

    # 8. Submission
    THRESHOLD = 0.23670230565035474
    if final_val_metric < THRESHOLD:
        final_test_preds = meta_learner.predict_proba(X_meta_test)

        sub_df = pd.DataFrame(final_test_preds, columns=label_classes)
        sub_df.insert(0, "id", test_df["id"])

        out_path = "./submission/submission.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sub_df.to_csv(out_path, index=False)
        print(f"Submission saved to {out_path}")


if __name__ == "__main__":
    main()
