import os
import numpy as np
import pandas as pd
import xgboost as xgb
import scipy.sparse
from sklearn.model_selection import StratifiedKFold
from library.utils import seed_everything, calculate_log_loss, ensure_directory
from library.feature_engineering import extract_meta_features
from library.data_loader import LABEL_MAP, create_stratified_folds

# Define directories
WORKING_DIR = "./working/idea_5/"
SUBMISSION_DIR = "./submission/"
INPUT_DIR = "./metadata/"

# Expert file names expected in WORKING_DIR
EXPERT_FILES = {
    "linear": {"oof": "oof_linear.npy", "test": "test_linear.npy"},
    "transformer": {"oof": "oof_transformer.npy", "test": "test_transformer.npy"},
}


def load_expert_preds(expert_name, is_test=False):
    """
    Loads predictions from a specific expert.
    """
    key = "test" if is_test else "oof"
    filename = EXPERT_FILES[expert_name][key]
    path = os.path.join(WORKING_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Prediction file for {expert_name} ({key}) not found at {path}. "
            "Ensure Level 1 experts have been run."
        )

    return np.load(path)


def prepare_stacking_data(n_folds=5, seed=42, load_cached_data=True, debug=False):
    """
    Aggregates OOF predictions, Test predictions, and Meta-features.
    Caches the resulting feature matrices.

    Args:
        n_folds (int): Number of folds used in upstream experts.
        seed (int): Random seed used in upstream experts.
        load_cached_data (bool): Whether to load from cache.
        debug (bool): If True, processes a subset of data.

    Returns:
        tuple: (X_train, y_train, X_test, test_ids)
    """
    ensure_directory(WORKING_DIR)

    # Cite debug_lesson_1: Validate artifact's metadata against runtime environment (debug vs full)
    suffix = "_debug" if debug else "_full"
    cache_train_path = os.path.join(WORKING_DIR, f"stacking_X_train{suffix}.npz")
    cache_test_path = os.path.join(WORKING_DIR, f"stacking_X_test{suffix}.npz")
    cache_y_path = os.path.join(WORKING_DIR, f"stacking_y_train{suffix}.npy")
    cache_ids_path = os.path.join(WORKING_DIR, f"stacking_test_ids{suffix}.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(cache_train_path)
            and os.path.exists(cache_test_path)
            and os.path.exists(cache_y_path)
            and os.path.exists(cache_ids_path)
        ):
            try:
                print("Loading cached stacking data...")
                X_train = scipy.sparse.load_npz(cache_train_path).toarray()
                X_test = scipy.sparse.load_npz(cache_test_path).toarray()
                y_train = np.load(cache_y_path)
                test_ids = np.load(cache_ids_path, allow_pickle=True)

                if debug:
                    return (
                        X_train[:1000],
                        y_train[:1000],
                        X_test[:1000],
                        test_ids[:1000],
                    )
                return X_train, y_train, X_test, test_ids
            except Exception as e:
                print(f"Failed to load stacking cache: {e}. Recomputing...")

    print("Preparing stacking data from scratch...")

    # 2. Load Metadata (Ground Truth & IDs)
    # Use create_stratified_folds to ensure we get the same subset/order as experts if debug=True
    train_df = create_stratified_folds(
        data_path=os.path.join(INPUT_DIR, "train.csv"),
        n_folds=n_folds,
        seed=seed,
        load_cached_data=load_cached_data,
        debug=debug,
    )

    test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
    if debug:
        # Match the sampling logic of experts (Linear uses head(1000))
        test_df = test_df.head(1000)
        print(f"Debug mode: Sampled {len(test_df)} test rows for Stacking.")

    y_train = train_df["author"].map(LABEL_MAP).values
    test_ids = test_df["id"].values

    # 3. Load Expert Predictions
    # Shape: (n_samples, 3)
    oof_linear = load_expert_preds("linear", is_test=False)
    test_linear = load_expert_preds("linear", is_test=True)

    oof_trans = load_expert_preds("transformer", is_test=False)
    test_trans = load_expert_preds("transformer", is_test=True)

    # Verify shapes match metadata
    if len(oof_linear) != len(train_df):
        raise ValueError(
            f"Mismatch: OOF Linear size {len(oof_linear)} != Train size {len(train_df)}"
        )
    if len(test_linear) != len(test_df):
        raise ValueError(
            f"Mismatch: Test Linear size {len(test_linear)} != Test size {len(test_df)}"
        )

    # 4. Extract Meta-Features
    # Note: We use the feature_engineering library which handles its own caching
    # Adjust dataset_id for debug to avoid cache collisions with full run
    train_id = "train_debug" if debug else "train"
    test_id = "test_debug" if debug else "test"

    meta_train_df = extract_meta_features(
        train_df, train_id, load_cached_data=load_cached_data
    )
    meta_test_df = extract_meta_features(
        test_df, test_id, load_cached_data=load_cached_data
    )

    # Select specific columns as per strategy
    meta_cols = ["char_len", "word_count", "punct_density"]
    meta_train = meta_train_df[meta_cols].values
    meta_test = meta_test_df[meta_cols].values

    # 5. Concatenate Features
    # Feature Order: [Linear_Probs(3), Transformer_Probs(3), Meta_Features(3)]
    X_train = np.hstack([oof_linear, oof_trans, meta_train])
    X_test = np.hstack([test_linear, test_trans, meta_test])

    print(f"Stacking Training Shape: {X_train.shape}")
    print(f"Stacking Test Shape: {X_test.shape}")

    # 6. Save to Cache
    try:
        scipy.sparse.save_npz(cache_train_path, scipy.sparse.csr_matrix(X_train))
        scipy.sparse.save_npz(cache_test_path, scipy.sparse.csr_matrix(X_test))
        np.save(cache_y_path, y_train)
        np.save(cache_ids_path, test_ids)
        print("Saved stacking data to cache.")
    except Exception as e:
        print(f"Warning: Could not save stacking cache: {e}")

    if debug:
        return X_train[:1000], y_train[:1000], X_test[:1000], test_ids[:1000]

    return X_train, y_train, X_test, test_ids


class XGBoostStacker:
    """
    Wrapper for XGBoost Classifier to handle stacking logic.
    """

    def __init__(self, params=None, n_folds=5, seed=42):
        self.params = params if params else {}
        self.n_folds = n_folds
        self.seed = seed
        self.models = []

    def train(self, X, y):
        """
        Trains the stacker using Stratified K-Fold CV.
        """
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed
        )

        oof_preds = np.zeros((len(X), 3))
        scores = []

        print(f"\nTraining Meta-Learner (XGBoost) with {self.n_folds} folds...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold, y_train_fold = X[train_idx], y[train_idx]
            X_val_fold, y_val_fold = X[val_idx], y[val_idx]

            # Initialize model
            model = xgb.XGBClassifier(
                **self.params,
                random_state=self.seed,
                callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)],
            )

            # Fit with early stopping
            model.fit(
                X_train_fold,
                y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False,
            )

            # Predict
            val_probs = model.predict_proba(X_val_fold)
            oof_preds[val_idx] = val_probs

            # Score
            fold_loss = calculate_log_loss(y_val_fold, val_probs)
            scores.append(fold_loss)

            # Store model
            self.models.append(model)

            # Retrieve best iteration for logging
            best_iter = model.best_iteration
            print(
                f"Meta-Fold {fold+1} Log Loss: {fold_loss:.10f} (Best Iter: {best_iter})"
            )

        overall_loss = calculate_log_loss(y, oof_preds)
        print(f"\nMeta-Learner Overall CV Log Loss: {overall_loss:.10f}")
        print(f"Average Meta-Fold Log Loss: {np.mean(scores):.10f}")

        return overall_loss

    def predict(self, X_test):
        """
        Generates predictions by averaging outputs from all fold models (Bagging).
        """
        if not self.models:
            raise RuntimeError("Model not trained yet.")

        test_preds_accum = np.zeros((len(X_test), 3))

        for model in self.models:
            test_preds_accum += model.predict_proba(X_test)

        return test_preds_accum / len(self.models)


def run_meta_learner(n_folds=5, seed=42, debug=False, load_cached_data=True):
    """
    Main execution function for the Meta-Learner module.
    """
    seed_everything(seed)
    ensure_directory(SUBMISSION_DIR)

    # 1. Prepare Data
    X_train, y_train, X_test, test_ids = prepare_stacking_data(
        n_folds=n_folds, seed=seed, load_cached_data=load_cached_data, debug=debug
    )

    # 2. Define Hyperparameters
    # Tuned for small feature set (9 features)
    xgb_params = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "n_jobs": -1,
        "verbosity": 0,
    }

    if debug:
        xgb_params["n_estimators"] = 50

    # 3. Train Stacker
    stacker = XGBoostStacker(params=xgb_params, n_folds=n_folds, seed=seed)
    stacker.train(X_train, y_train)

    # 4. Predict on Test
    print("Generating final test predictions...")
    final_probs = stacker.predict(X_test)

    # 5. Create Submission
    submission = pd.DataFrame(final_probs, columns=["EAP", "HPL", "MWS"])
    submission.insert(0, "id", test_ids)

    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    print("Preview:")
    print(submission.head())

    return submission
