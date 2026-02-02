import os
import gc
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Suppress warnings
warnings.filterwarnings("ignore")


class Config:
    """
    Configuration class for the Cover Type Prediction Task.
    """

    # Random Seed
    SEED = 42

    # Data Paths
    TRAIN_PATH = "./metadata/train.csv"
    TEST_PATH = "./metadata/test.csv"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = "submission.csv"
    CACHE_DIR = "./working/idea_11"

    # Cross Validation
    N_FOLDS = 5

    # Feature Engineering
    # Number of LDA components = min(n_classes - 1, n_features)
    # Classes are [1, 2, 3, 4, 6, 7] (6 classes) -> 5 components
    LDA_COMPONENTS = 5

    # XGBoost Hyperparameters
    XGB_PARAMS = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 10,
        "learning_rate": 0.05,
        "n_estimators": 3000,
        "early_stopping_rounds": 50,
        "verbosity": 0,
        "n_jobs": 12,
    }


def process_data(load_cached_data=True):
    """
    Loads and processes the data with caching mechanism.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        tuple: (train_df, test_df)
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    train_cache_path = os.path.join(Config.CACHE_DIR, "train_processed.parquet")
    test_cache_path = os.path.join(Config.CACHE_DIR, "test_processed.parquet")

    # Check cache
    if (
        load_cached_data
        and os.path.exists(train_cache_path)
        and os.path.exists(test_cache_path)
    ):
        print(f"Loading cached data from {Config.CACHE_DIR}...")
        try:
            train_df = pd.read_parquet(train_cache_path)
            test_df = pd.read_parquet(test_cache_path)
            return train_df, test_df
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    print("Processing data from scratch...")

    # Load raw metadata
    if not os.path.exists(Config.TRAIN_PATH):
        raise FileNotFoundError(f"{Config.TRAIN_PATH} not found.")
    if not os.path.exists(Config.TEST_PATH):
        raise FileNotFoundError(f"{Config.TEST_PATH} not found.")

    train_df = pd.read_csv(Config.TRAIN_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # --- Feature Engineering ---
    def engineer_features(df):
        # 1. Physics-Informed Features
        # Euclidean Distance to Hydrology
        h_dist = df["Horizontal_Distance_To_Hydrology"]
        v_dist = df["Vertical_Distance_To_Hydrology"]
        df["Hydrology_Distance"] = np.sqrt(h_dist**2 + v_dist**2)

        # Relative Elevation
        df["Relative_Elevation"] = df["Elevation"] - v_dist

        # Cyclic Aspect
        # Convert degrees to radians
        aspect_rad = np.radians(df["Aspect"])
        df["Aspect_Sin"] = np.sin(aspect_rad)
        df["Aspect_Cos"] = np.cos(aspect_rad)

        # 2. Robust Densification (Dot Product)
        # Soil_Type
        soil_cols = [c for c in df.columns if c.startswith("Soil_Type")]
        if soil_cols:
            # Create index vector [1, 2, ..., N]
            soil_indices = np.arange(1, len(soil_cols) + 1)
            # Dot product: If one-hot, result is the index. If all zero (missing), result is 0.
            df["Soil_Type_Index"] = df[soil_cols].dot(soil_indices)

        # Wilderness_Area
        wild_cols = [c for c in df.columns if c.startswith("Wilderness_Area")]
        if wild_cols:
            wild_indices = np.arange(1, len(wild_cols) + 1)
            df["Wilderness_Area_Index"] = df[wild_cols].dot(wild_indices)

        return df

    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    # Save to cache
    print(f"Saving processed data to {Config.CACHE_DIR}...")
    train_df.to_parquet(train_cache_path)
    test_df.to_parquet(test_cache_path)

    return train_df, test_df


def run_training_pipeline():
    """
    Executes the full training and inference pipeline:
    1. Data Loading & Processing
    2. Stratified K-Fold Cross Validation
    3. Dynamic LDA Projection (Leakage-Free)
    4. XGBoost Training with Early Stopping
    5. Soft Voting Ensemble
    6. Submission Generation
    """
    print("Starting Training Pipeline...")

    # 1. Load Data
    train_df, test_df = process_data(load_cached_data=True)

    # 2. Prepare Data for Training
    # Save Test Ids for submission
    if "Id" in test_df.columns:
        test_ids = test_df["Id"].values
        X_test_base = test_df.drop(columns=["Id"])
    else:
        # Fallback if Id missing
        test_ids = np.arange(4000000, 4000000 + len(test_df))
        X_test_base = test_df

    # Prepare Train
    target_col = "Cover_Type"
    if "Id" in train_df.columns:
        train_df = train_df.drop(columns=["Id"])

    y = train_df[target_col].values
    X = train_df.drop(columns=[target_col])

    # Encode Target (Map to 0..N-1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)
    print(f"Number of classes: {n_classes}")

    # 3. Stratified K-Fold CV
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Initialize Test Predictions Accumulator
    test_probs_sum = np.zeros((len(X_test_base), n_classes))

    # Loop Folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
        print(f"\n--- Fold {fold + 1} / {Config.N_FOLDS} ---")

        # Split Data
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y_encoded[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y_encoded[val_idx]

        # --- Supervised Linear Projection (LDA) ---
        # Fit on Train Fold ONLY to prevent leakage
        print("Fitting LDA...")
        lda = LinearDiscriminantAnalysis(n_components=Config.LDA_COMPONENTS)
        lda.fit(X_train_fold, y_train_fold)

        # Transform Train, Val, Test
        X_train_lda = lda.transform(X_train_fold)
        X_val_lda = lda.transform(X_val_fold)
        X_test_lda = lda.transform(X_test_base)

        # Concatenate LDA features to original features
        # Using numpy hstack for efficiency
        X_train_final = np.hstack([X_train_fold.values, X_train_lda])
        X_val_final = np.hstack([X_val_fold.values, X_val_lda])
        X_test_final = np.hstack([X_test_base.values, X_test_lda])

        # Clean up intermediate dataframes to free memory
        del X_train_fold, X_val_fold
        gc.collect()

        # --- XGBoost Training ---
        print("Training XGBoost...")
        model = xgb.XGBClassifier(**Config.XGB_PARAMS, random_state=Config.SEED)

        model.fit(
            X_train_final,
            y_train_fold,
            eval_set=[(X_val_final, y_val_fold)],
            verbose=False,
        )

        # Validation Metrics
        val_preds = model.predict(X_val_final)
        acc = accuracy_score(y_val_fold, val_preds)
        print(f"Fold {fold + 1} Accuracy: {acc:.8f}")

        # Inference on Test
        test_probs = model.predict_proba(X_test_final)
        test_probs_sum += test_probs

        # Cleanup
        del (
            model,
            X_train_final,
            X_val_final,
            X_test_final,
            X_train_lda,
            X_val_lda,
            X_test_lda,
        )
        gc.collect()

    # 4. Aggregate Predictions (Soft Voting)
    print("\nAggregating predictions...")
    avg_test_probs = test_probs_sum / Config.N_FOLDS
    final_pred_indices = np.argmax(avg_test_probs, axis=1)

    # Decode labels
    final_preds = le.inverse_transform(final_pred_indices)

    # 5. Create Submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(Config.SUBMISSION_DIR, Config.SUBMISSION_FILE)

    submission_df = pd.DataFrame({"Id": test_ids, "Cover_Type": final_preds})

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
