import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, mae_score, get_device
from library.feature_engineering import FeatureEngineer
from library.dataset import VolcanoDataset
from library.model_vision import EfficientNetFiLM
from library.model_tabular import train_lgbm_fold, predict_lgbm
from library.runner import (
    run_tabular_cv,
    run_vision_cv,
    train_meta_learner,
    generate_submission,
)

# ==========================================
# Configuration Overrides for Fast Baseline
# ==========================================
# Reduce epochs to ensure execution finishes well within the 1-hour limit
Config.EPOCHS = 10
# Keep 5 folds to ensure robust OOF generation for the meta-learner
Config.N_FOLDS = 5


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Running Fast Baseline with Device: {device}")

    # 2. Data Preparation
    fe = FeatureEngineer()

    # Process Training Data
    print("\n--- Processing Training Data ---")
    df_train = fe.process_dataset(
        os.path.join(Config.METADATA_DIR, "train.csv"),
        output_dir_name="train_data",
        load_cached_data=True,
    )
    train_spec_dir = os.path.join(Config.WORKING_DIR, "train_data", "spectrograms")

    # Process Validation Data (Hold-out)
    print("\n--- Processing Validation Data ---")
    df_val = fe.process_dataset(
        os.path.join(Config.METADATA_DIR, "val.csv"),
        output_dir_name="val_data",
        load_cached_data=True,
    )
    val_spec_dir = os.path.join(Config.WORKING_DIR, "val_data", "spectrograms")

    # Process Test Data
    print("\n--- Processing Test Data ---")
    df_test = fe.process_dataset(
        os.path.join(Config.METADATA_DIR, "test.csv"),
        output_dir_name="test_data",
        load_cached_data=True,
    )
    test_spec_dir = os.path.join(Config.WORKING_DIR, "test_data", "spectrograms")

    # 3. Model Training
    print("\n--- Starting Training Phase ---")

    # Branch A: Tabular (LightGBM)
    oof_tab, tab_models, tab_feats = run_tabular_cv(df_train)

    # Branch B: Vision (EfficientNet + FiLM)
    oof_vis, vis_model_paths, vis_scalar_stats = run_vision_cv(df_train, train_spec_dir)

    # Meta-Learner (Ridge Stacking)
    y_train_true = df_train["time_to_eruption"].values
    meta_model = train_meta_learner(oof_tab, oof_vis, y_train_true)

    # 4. Validation on Hold-out Set
    print("\n--- Running Validation on Hold-out Set ---")

    # 4a. Tabular Inference
    print("Generating Tabular Validation Predictions...")
    val_tab_preds = np.zeros(len(df_val))
    for model in tab_models:
        val_tab_preds += predict_lgbm(model, df_val, tab_feats)
    val_tab_preds /= len(tab_models)

    # 4b. Vision Inference
    print("Generating Vision Validation Predictions...")
    val_vis_preds = np.zeros(len(df_val))

    # Iterate through each fold's model to ensemble predictions
    for i, (path, stats) in enumerate(zip(vis_model_paths, vis_scalar_stats)):
        # Initialize dataset with the specific scalar stats from the training fold
        ds = VolcanoDataset(df_val, val_spec_dir, mode="val", scalar_stats=stats)
        loader = DataLoader(
            ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
        )

        scalar_dim = len(ds.scalar_cols)
        model = EfficientNetFiLM(scalar_input_dim=scalar_dim).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        fold_preds_log = []
        with torch.no_grad():
            for spec, scalar, _ in loader:
                spec = spec.to(device)
                scalar = scalar.to(device)
                output = model(spec, scalar).squeeze(1)
                fold_preds_log.append(output.cpu().numpy())

        fold_preds_log = np.concatenate(fold_preds_log)
        # Inverse Transform (expm1)
        val_vis_preds += np.expm1(fold_preds_log)

        # Cleanup to save memory
        del model, ds, loader
        torch.cuda.empty_cache()

    val_vis_preds /= len(vis_model_paths)

    # 4c. Ensemble Inference
    X_meta_val = np.column_stack([val_tab_preds, val_vis_preds])
    val_final_preds = meta_model.predict(X_meta_val)
    val_final_preds = np.maximum(val_final_preds, 0)  # Clip negative predictions

    # 4d. Metric Calculation
    y_val_true = df_val["time_to_eruption"].values
    val_mae = mae_score(y_val_true, val_final_preds)

    print(f"Final Validation Metric: {val_mae}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    errors = np.abs(y_val_true - val_final_preds)
    df_val_analysis = df_val.copy()
    df_val_analysis["error"] = errors

    # Calculate correlations between numeric features and the error magnitude
    numeric_cols = df_val_analysis.select_dtypes(include=[np.number]).columns
    # Drop ID columns and target from correlation check to focus on features
    cols_to_check = [
        c for c in numeric_cols if c not in ["segment_id", "time_to_eruption", "error"]
    ]

    if cols_to_check:
        correlations = (
            df_val_analysis[cols_to_check]
            .corrwith(df_val_analysis["error"])
            .sort_values(ascending=False)
        )
        print("Top 5 Features Positively Correlated with Error:")
        print(correlations.head(5))
        print("\nTop 5 Features Negatively Correlated with Error:")
        print(correlations.tail(5))
    else:
        print("No numeric features available for failure analysis.")

    # 6. Submission Generation
    threshold = 1920624.12
    if val_mae < threshold:
        print(
            f"\nValidation metric {val_mae} is better than threshold {threshold}. Proceeding to submission."
        )
        generate_submission(
            df_test,
            test_spec_dir,
            tab_models,
            tab_feats,
            vis_model_paths,
            vis_scalar_stats,
            meta_model,
        )
    else:
        print(
            f"\nValidation metric {val_mae} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()
