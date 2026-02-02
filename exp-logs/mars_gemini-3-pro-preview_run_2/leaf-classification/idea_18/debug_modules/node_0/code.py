import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# Ensure the current directory is in the path to import library modules
sys.path.append(".")

# Import provided library modules
from library import data_interface
from library import expert_pipelines
from library import ensemble_optimizer


def main():
    print("Starting demonstration run...")

    # ==========================================================================
    # 1. Data Loading
    # ==========================================================================
    print("\n[Step 1] Loading Dataset...")
    # Load data using the interface. Using cached data if available for speed.
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = (
        data_interface.load_dataset(load_cached_data=True)
    )

    # Verification: Check data shapes
    print(f"  Train shape: {X_train.shape}, Labels: {y_train.shape}")
    print(f"  Val shape:   {X_val.shape}, Labels: {y_val.shape}")
    print(f"  Test shape:  {X_test.shape}")
    print(f"  Classes: {len(classes)}")

    assert len(X_train) == len(y_train), "Mismatch in training data and labels"
    assert len(X_val) == len(y_val), "Mismatch in validation data and labels"
    assert (
        X_train.shape[1] == X_test.shape[1]
    ), "Feature mismatch between train and test"

    # ==========================================================================
    # 2. Model Training (Expert Pipelines)
    # ==========================================================================
    print("\n[Step 2] Training Expert Pipelines...")

    val_preds_dict = {}
    test_preds_dict = {}

    # --- Model A: Global LDA ---
    print("  Training Global LDA...")
    lda_pipeline = expert_pipelines.build_global_lda(random_state=42)
    lda_pipeline.fit(X_train, y_train)

    val_preds_dict["lda"] = lda_pipeline.predict_proba(X_val)
    test_preds_dict["lda"] = lda_pipeline.predict_proba(X_test)

    # --- Model B: Denoised LDA (Feature Selection) ---
    print("  Training Denoised LDA (k=100)...")
    # Selecting top 100 features based on Mutual Information
    denoised_lda_pipeline = expert_pipelines.build_denoised_lda(
        k_features=100, random_state=42
    )
    denoised_lda_pipeline.fit(X_train, y_train)

    val_preds_dict["denoised_lda"] = denoised_lda_pipeline.predict_proba(X_val)
    test_preds_dict["denoised_lda"] = denoised_lda_pipeline.predict_proba(X_test)

    # --- Model C: Global Logistic Regression (with CV) ---
    print("  Training Global LR (CV)...")
    # This pipeline performs internal cross-validation to find the best C
    lr_pipeline = expert_pipelines.build_global_lr(random_state=42)
    lr_pipeline.fit(X_train, y_train)

    val_preds_dict["lr_cv"] = lr_pipeline.predict_proba(X_val)
    test_preds_dict["lr_cv"] = lr_pipeline.predict_proba(X_test)

    # Verification: Check prediction shapes
    for name, preds in val_preds_dict.items():
        assert preds.shape == (
            len(X_val),
            len(classes),
        ), f"Val prediction shape mismatch for {name}"

    # Calculate individual scores
    print("\n  Individual Model Validation Log Loss:")
    for name, preds in val_preds_dict.items():
        score = log_loss(y_val, preds, labels=np.arange(len(classes)))
        print(f"    {name}: {score:.5f}")

    # ==========================================================================
    # 3. Ensemble Optimization
    # ==========================================================================
    print("\n[Step 3] Optimizing Ensemble...")

    selector = ensemble_optimizer.GreedyEnsembleSelector(
        max_iterations=20, tolerance=1e-5, random_state=42  # Limit iterations for speed
    )

    # Fit the ensemble selector on validation data
    selector.fit(val_preds_dict, y_val)

    print(f"  Best Ensemble Score: {selector.best_score_:.5f}")
    print(f"  Selected Weights: {selector.weights_}")

    # Verification: Ensemble should be at least as good as the best single model (roughly)
    best_single_model_score = min(
        [
            log_loss(y_val, p, labels=np.arange(len(classes)))
            for p in val_preds_dict.values()
        ]
    )
    # Allow small floating point margin
    assert (
        selector.best_score_ <= best_single_model_score + 1e-9
    ), "Ensemble optimization failed to match or beat best single model."

    # Generate ensemble predictions for test set
    final_test_preds = selector.predict(test_preds_dict)

    # ==========================================================================
    # 4. Final Retraining Demonstration (Fixed LR)
    # ==========================================================================
    print("\n[Step 4] Demonstrating Final Retraining with Fixed Hyperparameters...")

    # Extract the best C found by the LR CV pipeline
    # Structure: Pipeline -> LogisticRegressionCV object
    lr_cv_model = lr_pipeline.named_steps["lr_cv"]
    # C_ contains the best C for each class (OvR) or single C (Multinomial).
    # Since we used multi_class='multinomial', C_ shape is (1,) or (n_classes,) depending on implementation details of sklearn version.
    # Usually for multinomial lbfgs it returns an array of best Cs. We take the mean or the first one as a representative for this demo.
    best_c = np.mean(lr_cv_model.C_)
    print(f"  Extracted Best C from CV: {best_c:.5f}")

    # Build a fixed LR pipeline using this C
    fixed_lr_pipeline = expert_pipelines.build_fixed_lr(C=best_c, random_state=42)

    # In a real scenario, we would retrain on Train + Val. Here we just demonstrate fit on Train.
    fixed_lr_pipeline.fit(X_train, y_train)
    fixed_preds = fixed_lr_pipeline.predict_proba(X_test)

    assert (
        fixed_preds.shape == final_test_preds.shape
    ), "Fixed LR prediction shape mismatch"
    print("  Fixed LR Retraining successful.")

    # ==========================================================================
    # 5. Submission Generation
    # ==========================================================================
    print("\n[Step 5] Saving Submission...")

    submission_path = "./submission/submission.csv"
    data_interface.save_submission(
        predictions=final_test_preds,
        test_ids=test_ids,
        classes=classes,
        output_path=submission_path,
    )

    # Verification: Check if file exists and has correct format
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    print(f"  Submission saved to {submission_path}")
    print(f"  Submission shape: {df_sub.shape}")

    # Check expected columns: 'id' + 99 species
    expected_cols = 1 + 99
    assert (
        df_sub.shape[1] == expected_cols
    ), f"Expected {expected_cols} columns, found {df_sub.shape[1]}"
    assert df_sub.shape[0] == len(
        test_ids
    ), f"Expected {len(test_ids)} rows, found {df_sub.shape[0]}"

    print("\nDemonstration complete. Success!")


if __name__ == "__main__":
    main()
