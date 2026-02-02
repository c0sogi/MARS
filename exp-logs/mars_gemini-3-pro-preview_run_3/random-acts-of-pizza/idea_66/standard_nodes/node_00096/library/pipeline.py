import os
import numpy as np
import pandas as pd
import joblib
from library.config import Config
from library.data_loader import load_dataset
from library.feature_engineering import FeatureFactory
from library.model_factory import ModelFactory
from library.trainer import Trainer


def _get_feature_set_for_model(model_name, features_dict):
    """
    Maps a model name to its specific input feature set based on the architecture.

    Architecture Mapping:
    - Lexical Branch -> Lexical Sparse Matrix + Metadata
    - Behavioral Branch -> Community Sparse Matrix + Metadata
    - Semantic Branch -> Semantic Dense Matrix + Metadata
    - Metadata Branch -> Metadata Only
    """
    X_meta = features_dict["meta"]

    if "lexical" in model_name:
        return features_dict["lexical"], X_meta
    elif "community" in model_name:
        return features_dict["community"], X_meta
    elif "semantic" in model_name:
        return features_dict["semantic"], X_meta
    elif "metadata" in model_name or "temporal" in model_name:
        # For metadata/temporal models, the "main" feature is None,
        # so the trainer will just use X_meta.
        return None, X_meta
    else:
        raise ValueError(f"Unknown model branch for {model_name}")


def _is_volatile_model(model_name):
    """
    Determines if a model is 'Volatile' (Gradient Boosting) or 'Stable' (Linear/Bagging).
    Volatile models use CV-Bagging (average of fold models).
    Stable models use Full-Retraining.
    """
    # Based on Config naming conventions
    volatile_keywords = ["booster", "gradient"]
    return any(k in model_name for k in volatile_keywords)


def run_training_pipeline(debug_size=None, load_cached_data=True):
    """
    Executes the full training pipeline:
    1. Load Data
    2. Generate Features
    3. Train Level 1 Base Models (Hybrid Protocol)
    4. Train Level 2 Meta-Learner
    """
    print("Starting Training Pipeline...")

    # 1. Load Data
    train_df, test_df = load_dataset(load_cached_data=load_cached_data)

    if debug_size is not None:
        print(f"DEBUG: Truncating training data to {debug_size} samples")
        train_df = train_df.iloc[:debug_size]
        # We don't truncate test_df here usually, but for feature consistency in debug:
        test_df = test_df.iloc[:debug_size]

    y_train = train_df[Config.TARGET_COL].values

    # 2. Feature Engineering
    ff = FeatureFactory()

    print("Generating Features...")
    X_train_lex, X_test_lex = ff.make_lexical(
        train_df, test_df, load_cached_data, debug_size
    )
    X_train_com, X_test_com = ff.make_behavioral(
        train_df, test_df, load_cached_data, debug_size
    )
    X_train_sem, X_test_sem = ff.make_semantic(
        train_df, test_df, load_cached_data, debug_size
    )
    X_train_meta, X_test_meta = ff.make_metadata(
        train_df, test_df, load_cached_data, debug_size
    )

    # Store in dict for easy access
    features_train = {
        "lexical": X_train_lex,
        "community": X_train_com,
        "semantic": X_train_sem,
        "meta": X_train_meta,
    }

    # 3. Train Level 1 Models
    trainer = Trainer()
    base_models = ModelFactory.get_base_models()

    # Placeholder for OOF predictions (N_samples x N_models)
    oof_preds_df = pd.DataFrame(index=range(len(y_train)))

    for model_name, model_instance in base_models.items():
        # We need a factory function to pass to the trainer so it can instantiate fresh models per fold
        # We use a lambda that returns a clone or new instance
        # Since sklearn models are mutable, we must be careful.
        # ModelFactory returns new instances, but here we have the instance.
        # We'll rely on the Trainer to handle cloning or we pass a lambda.
        # However, ModelFactory.get_base_models returns instances.
        # Let's create a factory lambda for the specific model type based on Config.

        # Helper to get a fresh instance based on name
        def model_factory_func(name=model_name):
            models = ModelFactory.get_base_models()
            return models[name]

        X_main, X_meta = _get_feature_set_for_model(model_name, features_train)

        if _is_volatile_model(model_name):
            # Volatile: Train CV, save all folds, return OOF
            oof = trainer.train_cv_volatile(
                model_name, model_factory_func, X_main, X_meta, y_train
            )
        else:
            # Stable: Train CV for OOF, then Train Full for Inference
            oof = trainer.train_cv_stable(
                model_name, model_factory_func, X_main, X_meta, y_train
            )
            trainer.train_full_stable(
                model_name, model_factory_func, X_main, X_meta, y_train
            )

        oof_preds_df[model_name] = oof

    # 4. Train Level 2 Meta-Learner
    print("\nPreparing Meta-Learner...")
    X_oof = oof_preds_df.values
    meta_model = ModelFactory.get_meta_learner()
    meta_model, meta_oof_preds = trainer.train_meta_learner(meta_model, X_oof, y_train)

    # Save OOF predictions for analysis/validation
    oof_path = os.path.join(Config.CACHE_DIR, "oof_predictions.csv")
    pd.DataFrame({"y_true": y_train, "y_pred": meta_oof_preds}).to_csv(
        oof_path, index=False
    )
    print(f"Saved OOF predictions to {oof_path}")

    print("Training Pipeline Completed.")


def run_inference_pipeline(debug_size=None, load_cached_data=True):
    """
    Executes the inference pipeline:
    1. Load Test Data & Features
    2. Generate Level 1 Predictions (Hybrid Inference)
    3. Generate Level 2 Predictions
    4. Save Submission
    """
    print("\nStarting Inference Pipeline...")

    # 1. Load Data & Features
    # We reload to ensure clean state, utilizing cache
    train_df, test_df = load_dataset(load_cached_data=load_cached_data)

    if debug_size is not None:
        test_df = test_df.iloc[:debug_size]
        train_df = train_df.iloc[
            :debug_size
        ]  # Needed for feature factory fit check if not cached

    ff = FeatureFactory()
    # Note: make_* methods handle caching. If trained, cache exists.
    _, X_test_lex = ff.make_lexical(train_df, test_df, load_cached_data, debug_size)
    _, X_test_com = ff.make_behavioral(train_df, test_df, load_cached_data, debug_size)
    _, X_test_sem = ff.make_semantic(train_df, test_df, load_cached_data, debug_size)
    _, X_test_meta = ff.make_metadata(train_df, test_df, load_cached_data, debug_size)

    features_test = {
        "lexical": X_test_lex,
        "community": X_test_com,
        "semantic": X_test_sem,
        "meta": X_test_meta,
    }

    trainer = (
        Trainer()
    )  # Used for helper methods like _concat_features and path resolution
    base_models = ModelFactory.get_base_models()
    level1_preds = pd.DataFrame(index=range(len(test_df)))

    # 2. Level 1 Inference
    for model_name in base_models.keys():
        print(f"Predicting with {model_name}...")
        X_main, X_meta = _get_feature_set_for_model(model_name, features_test)
        X_test_fold = trainer._concat_features(X_main, X_meta)

        if _is_volatile_model(model_name):
            # Hybrid Inference: Average of 5 fold models
            fold_preds = []
            for fold in range(Config.N_FOLDS):
                path = trainer._get_model_path(model_name, fold)
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Model file not found: {path}")

                model = joblib.load(path)
                # Predict proba
                pred = model.predict_proba(X_test_fold)[:, 1]
                fold_preds.append(pred)

            # Average predictions
            avg_pred = np.mean(fold_preds, axis=0)
            level1_preds[model_name] = avg_pred

        else:
            # Hybrid Inference: Use single full model
            path = trainer._get_model_path(model_name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")

            model = joblib.load(path)
            pred = model.predict_proba(X_test_fold)[:, 1]
            level1_preds[model_name] = pred

    # 3. Level 2 Inference
    print("Predicting with Meta-Learner...")
    meta_path = trainer._get_model_path("meta_learner")
    meta_model = joblib.load(meta_path)

    final_preds = meta_model.predict_proba(level1_preds.values)[:, 1]

    # 4. Save Submission
    submission = pd.DataFrame(
        {Config.ID_COL: test_df[Config.ID_COL], Config.TARGET_COL: final_preds}
    )

    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Inference Pipeline Completed.")
