import os
import pandas as pd
import numpy as np
import warnings
from library.config import Config
from library.utils import set_seed, calc_kendall_tau, read_notebook
from library.fine_tuner import FineTuner
from library.feature_extractor import FeatureEngineer
from library.regressor import RankRegressor
from library.inference import Predictor, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    print("Setting up execution environment...")
    set_seed(Config.SEED)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Fine-Tuning Backbone
    print("\n=== Step 1: Fine-Tuning Backbone ===")
    # Initialize FineTuner and train on the subset defined in Config (50k)
    fine_tuner = FineTuner()
    fine_tuner.train(load_cached_data=True)

    # 3. Feature Extraction
    print("\n=== Step 2: Feature Extraction ===")
    feature_engineer = FeatureEngineer()

    # A. Training Data
    # Sample training data to 15k notebooks to ensure fast baseline execution
    REGRESSOR_TRAIN_SIZE = 15000
    train_full_df = pd.read_csv(Config.TRAIN_METADATA_PATH)

    if len(train_full_df) > REGRESSOR_TRAIN_SIZE:
        print(f"Sampling {REGRESSOR_TRAIN_SIZE} notebooks for regressor training...")
        train_sampled_df = train_full_df.sample(
            n=REGRESSOR_TRAIN_SIZE, random_state=Config.SEED
        )
        train_sampled_path = os.path.join(Config.WORKING_DIR, "train_sampled.csv")
        train_sampled_df.to_csv(train_sampled_path, index=False)
    else:
        train_sampled_path = Config.TRAIN_METADATA_PATH

    print("Processing Training Features...")
    # Note: If cache exists at Config.TRAIN_FEATURES_PATH, it will be loaded
    # regardless of the metadata path provided.
    df_train_features = feature_engineer.extract_features(
        metadata_path=train_sampled_path,
        save_path=Config.TRAIN_FEATURES_PATH,
        load_cached_data=True,
    )

    # B. Validation Data
    # Use full validation set for accurate metrics
    print("Processing Validation Features...")
    df_val_features = feature_engineer.extract_features(
        metadata_path=Config.VAL_METADATA_PATH,
        save_path=Config.VAL_FEATURES_PATH,
        load_cached_data=True,
    )

    # 4. Regressor Training
    print("\n=== Step 3: Regressor Training ===")
    regressor = RankRegressor()
    regressor.train(df_train_features, df_val_features)

    # 5. Validation & Metrics
    print("\n=== Step 4: Validation Evaluation ===")
    print("Generating validation predictions...")
    val_preds = regressor.predict(df_val_features)
    df_val_features["pred"] = val_preds

    # Reconstruct cell orders for validation
    print("Reconstructing validation cell orders...")
    val_metadata = pd.read_csv(Config.VAL_METADATA_PATH)

    # Create prediction dictionary for fast lookup
    pred_dict = {}
    for nb_id, group in df_val_features.groupby("id"):
        pred_dict[nb_id] = dict(zip(group["cell_id"], group["pred"]))

    val_pred_rows = []

    # Iterate through validation notebooks
    for _, row in val_metadata.iterrows():
        nb_id = row["id"]
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        try:
            code_cells, md_cells = read_notebook(file_path)
        except:
            continue

        n_code = len(code_cells)
        nb_preds = pred_dict.get(nb_id, {})

        cells_with_ranks = []

        # Rank Code Cells (fixed anchors)
        for i, cell in enumerate(code_cells):
            rank = i + 0.5
            cells_with_ranks.append((rank, cell["id"]))

        # Rank Markdown Cells (predicted)
        for cell in md_cells:
            cell_id = cell["id"]
            pred_ratio = nb_preds.get(cell_id, 1.0)
            rank = pred_ratio * n_code
            cells_with_ranks.append((rank, cell_id))

        # Sort by rank
        cells_with_ranks.sort(key=lambda x: x[0])
        ordered_ids = [c[1] for c in cells_with_ranks]
        cell_order_str = " ".join(ordered_ids)

        val_pred_rows.append({"id": nb_id, "cell_order": cell_order_str})

    df_val_pred = pd.DataFrame(val_pred_rows)

    # Calculate and print Metric
    kendall_tau = calc_kendall_tau(val_metadata, df_val_pred)
    print(f"Final Validation Metric: {kendall_tau}")

    # 6. Failure Analysis
    print("\n=== Step 5: Failure Analysis ===")
    df_val_features["error"] = (
        df_val_features["pred"] - df_val_features["target"]
    ).abs()

    # Calculate correlations between error and numeric features
    numeric_cols = df_val_features.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_exclude = ["target", "pred", "error"]
    feature_cols = [c for c in numeric_cols if c not in cols_to_exclude]

    print("Correlation between Error and Features (Top 10):")
    correlations = {}
    for col in feature_cols:
        corr = df_val_features["error"].corr(df_val_features[col])
        correlations[col] = corr

    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corr[:10]:
        print(f"  {feat}: {corr:.4f}")

    # 7. Submission
    print("\n=== Step 6: Submission Generation ===")
    THRESHOLD = 0.8061

    if kendall_tau > THRESHOLD:
        print(
            f"Validation metric ({kendall_tau:.4f}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        predictor = Predictor()

        # Predict Test Set
        # Note: Predictor uses the saved models from previous steps
        df_test_preds = predictor.predict_test_set(load_cached_data=True)

        # Generate final submission file
        generate_submission(df_test_preds)
    else:
        print(
            f"Validation metric ({kendall_tau:.4f}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
