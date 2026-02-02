import sys
import os
import pandas as pd
import numpy as np

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, compute_mcc
from library.data_loader import DataLoader
from library.feature_engineering import FeatureEngineer
from library.model_trainer import ModelTrainer


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    print("Initializing pipeline...")

    loader = DataLoader()
    fe = FeatureEngineer()
    trainer = ModelTrainer()

    # 2. Load Data
    print("\n--- Loading Data ---")

    # Load Train
    print("Loading Train Data...")
    df_train_meta, df_train_track, df_train_helmets = loader.load_dataset("train")

    # Load Validation
    print("Loading Validation Data...")
    df_val_meta, df_val_track, df_val_helmets = loader.load_dataset("validation")

    # Load Test
    print("Loading Test Data...")
    df_test_meta, df_test_track, df_test_helmets = loader.load_dataset("test")

    # Partition Streams
    print("Partitioning Streams...")
    train_meta_a, train_meta_b = loader.get_stream_data(df_train_meta)
    val_meta_a, val_meta_b = loader.get_stream_data(df_val_meta)
    test_meta_a, test_meta_b = loader.get_stream_data(df_test_meta)

    # 3. Feature Engineering
    print("\n--- Feature Engineering ---")

    # Stream A: Interaction
    print("Processing Stream A (Train)...")
    X_train_a, y_train_a, ids_train_a = fe.process_stream_a(
        train_meta_a, df_train_track, df_train_helmets
    )
    print("Processing Stream A (Validation)...")
    X_val_a, y_val_a, ids_val_a = fe.process_stream_a(
        val_meta_a, df_val_track, df_val_helmets
    )
    print("Processing Stream A (Test)...")
    X_test_a, _, ids_test_a = fe.process_stream_a(
        test_meta_a, df_test_track, df_test_helmets
    )

    # Stream B: Impact
    print("Processing Stream B (Train)...")
    X_train_b, y_train_b, ids_train_b = fe.process_stream_b(
        train_meta_b, df_train_track
    )
    print("Processing Stream B (Validation)...")
    X_val_b, y_val_b, ids_val_b = fe.process_stream_b(val_meta_b, df_val_track)
    print("Processing Stream B (Test)...")
    X_test_b, _, ids_test_b = fe.process_stream_b(test_meta_b, df_test_track)

    # 4. Model Training
    print("\n--- Model Training ---")

    # Train Stream A
    model_a, thresh_a, mcc_a = trainer.train_stream(
        X_train_a, y_train_a, X_val_a, y_val_a, "A"
    )

    # Train Stream B
    model_b, thresh_b, mcc_b = trainer.train_stream(
        X_train_b, y_train_b, X_val_b, y_val_b, "B"
    )

    # 5. Validation Assessment
    print("\n--- Validation Assessment ---")

    # Predict Stream A Validation
    probs_val_a = model_a.predict_proba(X_val_a)[:, 1]
    preds_val_a = (probs_val_a >= thresh_a).astype(int)

    # Predict Stream B Validation
    probs_val_b = model_b.predict_proba(X_val_b)[:, 1]
    preds_val_b = (probs_val_b >= thresh_b).astype(int)

    # Combine
    y_true_all = np.concatenate([y_val_a, y_val_b])
    y_pred_all = np.concatenate([preds_val_a, preds_val_b])

    final_mcc = compute_mcc(y_true_all, y_pred_all)
    print(f"Final Validation Metric: {final_mcc}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Analyze Stream A Failures
    residuals_a = np.abs(y_val_a - probs_val_a)
    corr_a = X_val_a.corrwith(pd.Series(residuals_a, index=X_val_a.index))
    print("Top 5 Features correlated with Error (Stream A):")
    print(corr_a.abs().sort_values(ascending=False).head(5))

    # Analyze Stream B Failures
    residuals_b = np.abs(y_val_b - probs_val_b)
    corr_b = X_val_b.corrwith(pd.Series(residuals_b, index=X_val_b.index))
    print("\nTop 5 Features correlated with Error (Stream B):")
    print(corr_b.abs().sort_values(ascending=False).head(5))

    # 7. Submission
    if final_mcc > 0.6968:
        print("\n--- Generating Submission ---")
        trainer.predict_and_submit(ids_test_a, X_test_a, ids_test_b, X_test_b)
    else:
        print(
            f"\nValidation Metric ({final_mcc}) did not meet threshold (0.6968). Skipping submission."
        )


if __name__ == "__main__":
    main()
