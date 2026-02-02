import numpy as np
import pandas as pd
import xgboost as xgb
from library.config import Config
from library.utils import seed_everything, compute_log_loss, save_submission
from library.features import extract_meta_features
from library.dataset import create_folds


def run_meta_learner(
    expert_a_oof, expert_a_test, expert_b_oof, expert_b_test, debug=False
):
    """
    Trains the Level 2 Meta-Learner (XGBoost) on expert predictions and meta-features.

    Args:
        expert_a_oof (np.array): OOF predictions from Expert A (N_train, 3).
        expert_a_test (np.array): Test predictions from Expert A (N_test, 3).
        expert_b_oof (np.array): OOF predictions from Expert B (N_train, 3).
        expert_b_test (np.array): Test predictions from Expert B (N_test, 3).
        debug (bool): Whether to run in debug mode (subset of data).
    """
    seed_everything(Config.SEED)

    print("\n[Meta-Learner] Initializing...")

    # --------------------------------------------------------------------------
    # 1. Prepare Training Data (OOF + Meta Features)
    # --------------------------------------------------------------------------
    # Load folds to get ground truth and text for meta-features
    # create_folds handles debug sampling internally
    df_train = create_folds(load_cached_data=True, debug=debug)

    # Map string labels to integers
    y_train = df_train["author"].map(Config.LABEL2ID).values

    # Extract meta-features for training data
    print("[Meta-Learner] Extracting meta-features for training data...")
    meta_train_df = extract_meta_features(df_train)
    X_meta_train = meta_train_df.values.astype(np.float32)

    # --------------------------------------------------------------------------
    # 2. Prepare Test Data
    # --------------------------------------------------------------------------
    # Load test data directly to access text
    df_test = pd.read_csv(Config.TEST_DATA_PATH)

    # Apply debug sampling if necessary (Must match logic in get_test_dataset)
    if debug:
        df_test = df_test.sample(
            n=min(len(df_test), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    test_ids = df_test["id"].values

    # Extract meta-features for test data
    print("[Meta-Learner] Extracting meta-features for test data...")
    meta_test_df = extract_meta_features(df_test)
    X_meta_test = meta_test_df.values.astype(np.float32)

    # --------------------------------------------------------------------------
    # 3. Construct Feature Matrices (Stacking)
    # --------------------------------------------------------------------------
    # Concatenate: [Expert A (3), Expert B (3), Meta Features (3)]
    print("[Meta-Learner] Stacking features...")
    X_train_full = np.hstack([expert_a_oof, expert_b_oof, X_meta_train])
    X_test_full = np.hstack([expert_a_test, expert_b_test, X_meta_test])

    print(f"  Training Input Shape: {X_train_full.shape}")
    print(f"  Test Input Shape:     {X_test_full.shape}")

    # --------------------------------------------------------------------------
    # 4. Train with Early Stopping (Hold-out Validation)
    # --------------------------------------------------------------------------
    # We use Fold 0 as the validation set for the meta-learner to tune n_estimators
    val_fold = 0
    train_mask = df_train["fold"] != val_fold
    val_mask = df_train["fold"] == val_fold

    X_train_split = X_train_full[train_mask]
    y_train_split = y_train[train_mask]

    X_val_split = X_train_full[val_mask]
    y_val_split = y_train[val_mask]

    print(
        f"\n[Meta-Learner] Tuning on split (Train: Folds 1-4, Val: Fold {val_fold})..."
    )

    # Update params for training
    xgb_params = Config.XGB_PARAMS.copy()
    xgb_params["n_estimators"] = 2000  # High number to allow early stopping

    model = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)

    model.fit(
        X_train_split,
        y_train_split,
        eval_set=[(X_val_split, y_val_split)],
        verbose=False,
    )

    # Evaluate
    val_preds = model.predict_proba(X_val_split)
    val_score = compute_log_loss(y_val_split, val_preds)

    # Retrieve best iteration
    try:
        best_iteration = model.best_iteration
    except AttributeError:
        best_iteration = model.get_booster().best_iteration

    print(f"  Best Iteration: {best_iteration}")
    print(f"  Validation Log Loss: {val_score}")

    # --------------------------------------------------------------------------
    # 5. Retrain on Full Data & Predict
    # --------------------------------------------------------------------------
    print(
        f"\n[Meta-Learner] Retraining on full OOF dataset with {best_iteration} rounds..."
    )

    final_model = xgb.XGBClassifier(**xgb_params)
    # Set the optimal number of trees found
    final_model.set_params(n_estimators=best_iteration)

    final_model.fit(X_train_full, y_train, verbose=False)

    print("[Meta-Learner] Generating test predictions...")
    test_probs = final_model.predict_proba(X_test_full)

    # --------------------------------------------------------------------------
    # 6. Save Submission
    # --------------------------------------------------------------------------
    save_submission(test_ids, test_probs)
    print(f"[Meta-Learner] Submission saved to {Config.SUBMISSION_FILE_PATH}")
