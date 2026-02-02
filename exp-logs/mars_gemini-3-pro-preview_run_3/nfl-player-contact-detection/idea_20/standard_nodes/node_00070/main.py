import os
import sys
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef

# Import provided library modules
import library.config as C
import library.utils as U
import library.data_manager as DM
import library.features_stream_a as FA
import library.features_stream_b as FB
import library.trainer as TR
import library.inference as INF


def main():
    # 1. Initialization
    U.setup_logging(os.path.join(C.WORKING_DIR, "run.log"))
    np.random.seed(C.SEED)
    print("Starting runfile.py execution...")

    # 2. Data Loading & Feature Generation (Train & Validation)

    # --- Load Training Data ---
    print("\n--- Loading Training Data ---")
    df_merged_train, df_helmets_train = DM.load_and_merge_data(
        mode="train", load_cached_data=True
    )
    df_stream_a_train, df_stream_b_train = DM.split_by_stream(df_merged_train)

    # Clean up merged dataframe to save memory
    del df_merged_train
    gc.collect()

    # Generate Features for Stream A (Train)
    print("Generating Stream A Train Features...")
    X_train_a, y_train_a, ids_train_a = FA.generate_stream_a_features(
        df_stream_a_train, df_helmets_train, mode="train", load_cached_data=True
    )
    del df_stream_a_train
    gc.collect()

    # Generate Features for Stream B (Train)
    print("Generating Stream B Train Features...")
    X_train_b, y_train_b, ids_train_b = FB.generate_stream_b_features(
        df_stream_b_train, mode="train", load_cached_data=True
    )
    del df_stream_b_train, df_helmets_train
    gc.collect()

    # --- Load Validation Data ---
    print("\n--- Loading Validation Data ---")
    df_merged_val, df_helmets_val = DM.load_and_merge_data(
        mode="validation", load_cached_data=True
    )
    df_stream_a_val, df_stream_b_val = DM.split_by_stream(df_merged_val)

    del df_merged_val
    gc.collect()

    # Generate Features for Stream A (Validation)
    print("Generating Stream A Validation Features...")
    X_val_a, y_val_a, ids_val_a = FA.generate_stream_a_features(
        df_stream_a_val, df_helmets_val, mode="validation", load_cached_data=True
    )
    del df_stream_a_val
    gc.collect()

    # Generate Features for Stream B (Validation)
    print("Generating Stream B Validation Features...")
    X_val_b, y_val_b, ids_val_b = FB.generate_stream_b_features(
        df_stream_b_val, mode="validation", load_cached_data=True
    )
    del df_stream_b_val, df_helmets_val
    gc.collect()

    # 3. Model Training
    # Removed uniform subsampling to preserve minority class (Cite 00060)
    # The trainer handles targeted majority undersampling.

    print("\n--- Training Models ---")

    # Train Stream A
    print("Training Stream A...")
    model_a, thresh_a, mcc_a = TR.train_stream_model(
        "A", X_train_a, y_train_a, X_val_a, y_val_a, save_model=True
    )
    # Free training memory
    del X_train_a, y_train_a
    gc.collect()

    # Train Stream B
    print("Training Stream B...")
    model_b, thresh_b, mcc_b = TR.train_stream_model(
        "B", X_train_b, y_train_b, X_val_b, y_val_b, save_model=True
    )
    # Free training memory
    del X_train_b, y_train_b
    gc.collect()

    # 4. Global Validation & Metric Calculation
    print("\n--- Global Validation ---")

    # Predict Stream A Val
    probas_val_a = model_a.predict_proba(X_val_a)[:, 1]
    preds_val_a = (probas_val_a >= thresh_a).astype(int)
    df_res_a = pd.DataFrame(
        {
            "contact_id": ids_val_a,
            "y_true": y_val_a,
            "y_pred": preds_val_a,
            "prob": probas_val_a,
        }
    )

    # Predict Stream B Val
    probas_val_b = model_b.predict_proba(X_val_b)[:, 1]
    preds_val_b = (probas_val_b >= thresh_b).astype(int)
    df_res_b = pd.DataFrame(
        {
            "contact_id": ids_val_b,
            "y_true": y_val_b,
            "y_pred": preds_val_b,
            "prob": probas_val_b,
        }
    )

    # Combine
    df_val_results = pd.concat([df_res_a, df_res_b], ignore_index=True)

    # Calculate Final Metric
    final_mcc = matthews_corrcoef(df_val_results["y_true"], df_val_results["y_pred"])
    print(f"Final Validation Metric: {final_mcc}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Calculate Error Magnitude
    df_val_results["error"] = np.abs(df_val_results["y_true"] - df_val_results["prob"])

    # Map errors back to features for analysis
    # We do this separately for A and B as they have different features

    print("Stream A Error Correlations:")
    df_err_a = df_res_a.copy()
    df_err_a["error"] = np.abs(df_err_a["y_true"] - df_err_a["prob"])
    # Re-attach features (using index alignment, assuming no shuffle in between)
    # Note: X_val_a is a dataframe
    corrs_a = []
    for col in X_val_a.columns:
        if X_val_a[col].std() > 0:
            corr = np.corrcoef(df_err_a["error"], X_val_a[col])[0, 1]
            corrs_a.append((col, corr))

    corrs_a.sort(key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in corrs_a[:5]:
        print(f"  {feat}: {corr:.4f}")

    print("Stream B Error Correlations:")
    df_err_b = df_res_b.copy()
    df_err_b["error"] = np.abs(df_err_b["y_true"] - df_err_b["prob"])
    corrs_b = []
    for col in X_val_b.columns:
        if X_val_b[col].std() > 0:
            corr = np.corrcoef(df_err_b["error"], X_val_b[col])[0, 1]
            corrs_b.append((col, corr))

    corrs_b.sort(key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in corrs_b[:5]:
        print(f"  {feat}: {corr:.4f}")

    # 6. Inference & Submission
    TARGET_METRIC = 0.6968

    if final_mcc > TARGET_METRIC:
        print(f"\nMetric {final_mcc} > {TARGET_METRIC}. Proceeding to Submission...")

        # Load Test Data
        print("Loading Test Data...")
        df_merged_test, df_helmets_test = DM.load_and_merge_data(
            mode="test", load_cached_data=True
        )
        df_stream_a_test, df_stream_b_test = DM.split_by_stream(df_merged_test)

        del df_merged_test
        gc.collect()

        # Generate Test Features
        print("Generating Test Features A...")
        X_test_a, _, ids_test_a = FA.generate_stream_a_features(
            df_stream_a_test, df_helmets_test, mode="test", load_cached_data=True
        )
        del df_stream_a_test
        gc.collect()

        print("Generating Test Features B...")
        X_test_b, _, ids_test_b = FB.generate_stream_b_features(
            df_stream_b_test, mode="test", load_cached_data=True
        )
        del df_stream_b_test, df_helmets_test
        gc.collect()

        # Run Inference Pipeline
        # This re-optimizes thresholds on validation data and generates submission
        INF.run_inference_pipeline(
            X_val_a,
            y_val_a,
            X_val_b,
            y_val_b,
            X_test_a,
            ids_test_a,
            X_test_b,
            ids_test_b,
        )

    else:
        print(f"\nMetric {final_mcc} <= {TARGET_METRIC}. Skipping Submission.")

    print("Done.")


if __name__ == "__main__":
    main()
