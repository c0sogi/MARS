import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import log_loss
from scipy.stats import pearsonr


# Import library modules
from library.config import Config
from library.utils import seed_everything, save_submission
from library.data_loader import load_tabular_data
from library.feature_extractor import get_raw_image_features, reduce_dimensions
from library.model_pipeline import HybridEnsemble


def main():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)

    print("--- Data Loading Phase ---")
    # Determine limit for debug mode
    limit = Config.DEBUG_SAMPLE_SIZE if Config.DEBUG else None

    # Load Tabular Data
    X_train_tab, y_train, ids_train_tab = load_tabular_data("train", limit=limit)
    X_val_tab, y_val, ids_val_tab = load_tabular_data("val", limit=limit)
    X_test_tab, _, ids_test_tab = load_tabular_data("test", limit=limit)

    # Load Image Features
    X_train_img_raw, ids_train_img = get_raw_image_features("train")
    X_val_img_raw, ids_val_img = get_raw_image_features("val")
    X_test_img_raw, ids_test_img = get_raw_image_features("test")

    # Verify alignment
    assert np.array_equal(ids_train_tab, ids_train_img), "Train IDs mismatch"
    assert np.array_equal(ids_val_tab, ids_val_img), "Val IDs mismatch"
    assert np.array_equal(ids_test_tab, ids_test_img), "Test IDs mismatch"

    print("--- Feature Engineering Phase ---")
    # Dimensionality Reduction on Image Features
    X_train_img_pca, X_val_img_pca, X_test_img_pca = reduce_dimensions(
        X_train_img_raw, X_val_img_raw, X_test_img_raw
    )

    print("--- Validation & Metric Calculation Phase ---")
    # We replicate the pipeline logic here to explicitly capture the metric variable

    # 1. Scaling Tabular Data
    scaler = StandardScaler()
    X_train_tab_sc = scaler.fit_transform(X_train_tab)
    X_val_tab_sc = scaler.transform(X_val_tab)

    # 2. Feature Fusion
    X_train_fused = np.hstack([X_train_tab_sc, X_train_img_pca])
    X_val_fused = np.hstack([X_val_tab_sc, X_val_img_pca])

    # 3. Hyperparameter Tuning (LR)
    # We use the same strategy as the pipeline: Tune on Train+Val (using PredefinedSplit)
    # But for the hold-out metric calculation, we need a model trained on Train only.
    # To find the best C, we can run the grid search as intended by the pipeline design.

    X_comb_val = np.vstack([X_train_fused, X_val_fused])
    y_comb_val = np.concatenate([y_train, y_val])
    test_fold = np.concatenate(
        [np.full(X_train_fused.shape[0], -1), np.zeros(X_val_fused.shape[0], dtype=int)]
    )
    ps = PredefinedSplit(test_fold)

    lr_tuner = LogisticRegression(
        solver=Config.LOG_REG_SOLVER,
        multi_class="multinomial",
        max_iter=Config.LOG_REG_MAX_ITER,
        random_state=Config.SEED,
        n_jobs=-1,
    )

    param_grid = {"C": Config.LOG_REG_C_GRID}
    gs = GridSearchCV(
        lr_tuner, param_grid, cv=ps, scoring="neg_log_loss", n_jobs=-1, refit=False
    )
    gs.fit(X_comb_val, y_comb_val)
    best_C = gs.best_params_["C"]

    # 4. Train Models on Train Set Only
    # Logistic Regression
    lr_model = LogisticRegression(
        C=best_C,
        solver=Config.LOG_REG_SOLVER,
        multi_class="multinomial",
        max_iter=Config.LOG_REG_MAX_ITER,
        random_state=Config.SEED,
        n_jobs=-1,
    )
    lr_model.fit(X_train_fused, y_train)

    # LDA
    lda_model = LinearDiscriminantAnalysis(
        solver=Config.LDA_SOLVER, shrinkage=Config.LDA_SHRINKAGE
    )
    lda_model.fit(X_train_fused, y_train)

    # 5. Evaluate on Validation Set
    lr_preds = lr_model.predict_proba(X_val_fused)
    lda_preds = lda_model.predict_proba(X_val_fused)

    # Weighted Ensemble Optimization
    # Cite solution_lesson_node_00013: Avoid unweighted averaging of disparate models
    # Cite solution_lesson_node_00014: Use discrete grid search to ensure boundary conditions (0.0/1.0) are evaluated
    print("Optimizing ensemble weights on validation set...")
    alphas = np.linspace(0, 1, 101)
    best_alpha = 0.5
    val_log_loss = float("inf")

    for alpha in alphas:
        w_lr = alpha
        w_lda = 1.0 - alpha
        blend = w_lr * lr_preds + w_lda * lda_preds
        loss = log_loss(y_val, blend, labels=lr_model.classes_)

        if loss < val_log_loss:
            val_log_loss = loss
            best_alpha = alpha

    # Generate final ensemble predictions with optimal weights
    ensemble_preds = best_alpha * lr_preds + (1.0 - best_alpha) * lda_preds

    print(f"Optimal LR Weight: {best_alpha:.4f}")
    print(f"Final Validation Metric: {val_log_loss}")

    print("--- Failure Analysis ---")
    # Calculate per-sample log loss
    # We need to extract the probability assigned to the true class for each sample
    class_indices = {cls: i for i, cls in enumerate(lr_model.classes_)}
    y_val_indices = np.array([class_indices[label] for label in y_val])

    # Gather probabilities for the true classes
    # ensemble_preds is (N, n_classes)
    # We want preds[i, y_val_indices[i]]
    true_class_probs = ensemble_preds[np.arange(len(y_val)), y_val_indices]

    # Clip for safety (though log_loss handles it, we do it for manual calc)
    epsilon = 1e-15
    true_class_probs = np.clip(true_class_probs, epsilon, 1 - epsilon)
    sample_losses = -np.log(true_class_probs)

    # Correlation Analysis
    # Correlate sample_losses with X_val_fused features
    n_features = X_val_fused.shape[1]
    correlations = []

    for i in range(n_features):
        feat_vals = X_val_fused[:, i]
        # Handle constant features to avoid warning
        if np.std(feat_vals) == 0:
            corr = 0
        else:
            corr, _ = pearsonr(sample_losses, feat_vals)
        correlations.append(corr)

    correlations = np.array(correlations)

    # Get top 10 features positively correlated with error (high feature value -> high error)
    top_error_indices = np.argsort(correlations)[::-1][:10]

    print(
        "Top 10 features associated with high model error (Correlation with Log Loss):"
    )
    for idx in top_error_indices:
        # Determine feature name
        if idx < 192:
            # It's a tabular feature. We need to map index to name.
            # We don't have the column names loaded explicitly in X, but we know the order:
            # Margin (64) -> Shape (64) -> Texture (64)
            if idx < 64:
                feat_name = f"margin_{idx+1}"
            elif idx < 128:
                feat_name = f"shape_{idx-64+1}"
            else:
                feat_name = f"texture_{idx-128+1}"
        else:
            # It's a PCA feature
            feat_name = f"img_pca_{idx-192}"

        print(f"  {feat_name}: {correlations[idx]:.4f}")

    # Submission Threshold Check
    THRESHOLD = 3.715704343830924e-12

    if val_log_loss < THRESHOLD:
        print(
            f"\nValidation metric {val_log_loss} is better than threshold {THRESHOLD}."
        )
        print("Proceeding with full training and submission generation...")

        # Initialize Pipeline
        pipeline = HybridEnsemble()

        # Train (Refit on Train + Val)
        # Note: pipeline.train_models performs the internal tuning again.
        # This is acceptable and ensures the pipeline is self-contained.
        pipeline.train_models(
            X_train_tab, X_train_img_pca, y_train, X_val_tab, X_val_img_pca, y_val
        )

        # Predict on Test
        test_preds, class_names = pipeline.predict_ensemble(X_test_tab, X_test_img_pca)

        # Save Submission
        save_submission(
            test_preds, ids_test_tab, class_names, Config.SUBMISSION_FILE_PATH
        )

    else:
        print(f"\nValidation metric {val_log_loss} did not meet threshold {THRESHOLD}.")
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
