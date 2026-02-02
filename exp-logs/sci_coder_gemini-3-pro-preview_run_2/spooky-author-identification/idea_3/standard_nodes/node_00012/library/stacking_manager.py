import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from library.utils import set_seed, compute_log_loss, generate_config_hash, get_device
from library.data_loader import load_and_process_data
from library.feature_extractor import ClassicalFeaturePipeline, NeuralFeaturePipeline
from library.model_zoo import ClassicalModelWrapper, TransformerClassifier
from library.training_engine import (
    train_classical_model,
    train_neural_model,
    validate_neural,
    _create_dataloader,
)


class StackingEnsemble:
    """
    Orchestrates the Hybrid Classical-Neural Stacking Ensemble.
    Manages data loading, feature extraction, cross-validation of base models,
    meta-learner training, and submission generation.
    """

    def __init__(self, config):
        self.config = config
        self.seed = config.get("seed", 42)
        self.cache_dir = "./working/idea_3"

        # Ensure reproducibility and directory existence
        set_seed(self.seed)
        os.makedirs(self.cache_dir, exist_ok=True)

    def run(self):
        """
        Executes the full stacking pipeline.
        """
        print("Initializing Stacking Ensemble...")

        # 1. Load Data
        train_df, test_df, label_classes = load_and_process_data(self.config)
        y = train_df["author_encoded"].values
        folds = train_df["fold"].values

        # 2. Feature Extraction
        print("\n--- Feature Extraction ---")

        # Classical Features (Sparse TF-IDF & Dense SVD)
        classical_pipe = ClassicalFeaturePipeline(self.config)
        X_train_sparse, X_train_dense, X_test_sparse, X_test_dense = (
            classical_pipe.execute(train_df["text"], test_df["text"])
        )

        # Neural Features (Token IDs & Masks)
        neural_pipe = NeuralFeaturePipeline(self.config)
        X_train_neural, X_test_neural = neural_pipe.execute(
            train_df["text"], test_df["text"]
        )

        # 3. Level 1 Model Training (Base Learners)
        print("\n--- Level 1 Model Training ---")
        level1_oof_preds = []
        level1_test_preds = []

        # Define models: (Name, Type, Train Features, Test Features, Is Neural?)
        models = [
            ("lr", "lr", X_train_sparse, X_test_sparse, False),
            ("nb", "nb", X_train_sparse, X_test_sparse, False),
            ("xgb", "xgb", X_train_dense, X_test_dense, False),
            ("transformer", "transformer", X_train_neural, X_test_neural, True),
        ]

        for name, model_type, X_tr_full, X_te_full, is_neural in models:
            oof, test_pred = self._get_cv_predictions(
                name,
                model_type,
                X_tr_full,
                y,
                folds,
                X_te_full,
                label_classes,
                is_neural,
            )
            level1_oof_preds.append(oof)
            level1_test_preds.append(test_pred)

            # Print metric for individual model
            loss = compute_log_loss(y, oof)
            print(f"Model [{name}] OOF Log Loss: {loss}")

        # 4. Level 2 Meta-Learner Training
        print("\n--- Level 2 Meta-Learner Training ---")

        # Stack predictions horizontally
        X_meta_train = np.hstack(level1_oof_preds)
        X_meta_test = np.hstack(level1_test_preds)

        # Train Logistic Regression Meta-Learner
        meta_learner = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=self.seed,
            max_iter=1000,
        )
        meta_learner.fit(X_meta_train, y)

        # Sanity check on meta-learner performance
        meta_train_probs = meta_learner.predict_proba(X_meta_train)
        meta_loss = compute_log_loss(y, meta_train_probs)
        print(f"Meta-Learner Training Log Loss: {meta_loss}")

        # 5. Generate Final Predictions
        final_preds = meta_learner.predict_proba(X_meta_test)

        # 6. Save Submission
        self._save_submission(test_df, final_preds, label_classes)

    def _get_cv_predictions(
        self,
        name,
        model_type,
        X,
        y,
        folds,
        X_test,
        label_classes,
        is_neural,
        transformer_model_name=None,
    ):
        """
        Performs Stratified K-Fold CV to generate OOF predictions and averages Test predictions.
        Implements caching to avoid re-training if config hasn't changed.
        """
        # Generate hash based on full config
        config_hash = generate_config_hash(self.config)
        oof_cache_path = os.path.join(self.cache_dir, f"oof_{name}_{config_hash}.npy")
        test_cache_path = os.path.join(self.cache_dir, f"test_{name}_{config_hash}.npy")

        # Check Cache
        if os.path.exists(oof_cache_path) and os.path.exists(test_cache_path):
            print(f"Loading cached predictions for {name}...")
            return np.load(oof_cache_path), np.load(test_cache_path)

        print(f"Training {name} from scratch (CV)...")

        n_samples = len(y)
        n_classes = len(label_classes)
        n_folds = self.config.get("n_folds", 5)

        # Determine test set size
        if is_neural:
            n_test = X_test["input_ids"].shape[0]
        else:
            n_test = X_test.shape[0]

        # Initialize arrays
        oof_preds = np.zeros((n_samples, n_classes))
        test_preds_accum = np.zeros((n_test, n_classes))

        for fold in range(n_folds):
            print(f"  > Fold {fold + 1}/{n_folds}")

            # Split Data
            val_idx = folds == fold
            train_idx = ~val_idx

            y_tr, y_val = y[train_idx], y[val_idx]

            # Slice features based on type
            if is_neural:
                X_tr = {k: v[train_idx] for k, v in X.items()}
                X_val = {k: v[val_idx] for k, v in X.items()}
            else:
                X_tr = X[train_idx]
                X_val = X[val_idx]

            # Train and Predict
            if is_neural:
                # Initialize new model instance
                model_name = (
                    transformer_model_name
                    if transformer_model_name
                    else self.config.get("transformer_model", "roberta-base")
                )
                model = TransformerClassifier(
                    model_name,
                    num_classes=n_classes,
                )
                # Train
                trained_model, val_probs, _ = train_neural_model(
                    model, X_tr, y_tr, X_val, y_val, self.config
                )

                # Predict on Test Set
                device = get_device()
                test_loader = _create_dataloader(
                    X_test, None, self.config.get("batch_size", 16), shuffle=False
                )
                _, fold_test_probs = validate_neural(trained_model, test_loader, device)

            else:
                # Classical Model
                wrapper = ClassicalModelWrapper(model_type, self.config)
                _, val_probs, _ = train_classical_model(
                    wrapper, X_tr, y_tr, X_val, y_val
                )
                fold_test_probs = wrapper.predict_proba(X_test)

            # Store predictions
            oof_preds[val_idx] = val_probs
            test_preds_accum += fold_test_probs

        # Average test predictions across folds
        test_preds_avg = test_preds_accum / n_folds

        # Save to Cache
        print(f"Saving predictions for {name} to cache.")
        np.save(oof_cache_path, oof_preds)
        np.save(test_cache_path, test_preds_avg)

        return oof_preds, test_preds_avg

    def _save_submission(self, test_df, preds, classes):
        """
        Formats and saves the submission file.
        """
        sub_df = pd.DataFrame(preds, columns=classes)
        sub_df.insert(0, "id", test_df["id"])

        out_path = "./submission/submission.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        sub_df.to_csv(out_path, index=False)
        print(f"Submission saved to {out_path}")
