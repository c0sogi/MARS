import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

import library.config as conf
import library.data_loader as data_loader
import library.model_factory as model_factory
import library.ensemble as ensemble


class PhaseManager:
    """
    Orchestrates the Dynamic Multi-View Ensemble Selection pipeline.
    Manages data loading, two-stage training (Selection & Retraining), and inference.
    """

    def __init__(self):
        self.data_manager = data_loader.LeafDataManager()
        self.selector = ensemble.GreedyEnsembleSelector(n_iterations=100)

    def execute(self):
        """
        Runs the full pipeline: Data Loading -> Phase A (Selection) -> Phase B (Retraining) -> Inference.
        """
        # 1. Load Data
        print("Loading and processing data...")
        # Use caching to save time if re-running
        (
            X_train_views,
            y_train,
            X_val_views,
            y_val,
            X_test_views,
            test_ids,
            classes,
        ) = self.data_manager.load_all_data(load_cached_data=True)

        # 2. Phase A: Selection (Train on Train, Evaluate on Val)
        print("\n" + "=" * 40)
        print("PHASE A: Expert Selection")
        print("=" * 40)

        # Initialize pool
        pool = model_factory.get_expert_pool()

        # Train all candidates on Training set
        print(f"Training {len(pool)} candidate experts on Training set...")
        fitted_models_a = self.train_pool(pool, X_train_views, y_train)

        # Generate predictions on Validation set
        print("Generating validation predictions...")
        val_preds = self.generate_predictions(fitted_models_a, X_val_views)

        # Run Greedy Forward Selection
        print("Running Greedy Ensemble Selection...")
        self.selector.fit(val_preds, y_val)

        selected_names = self.selector.selected_models_
        unique_selected = sorted(list(set(selected_names)))
        print(f"Selected {len(unique_selected)} unique models for final ensemble.")

        # 3. Phase B: Final Retraining (Train on Train + Val)
        print("\n" + "=" * 40)
        print("PHASE B: Final Retraining")
        print("=" * 40)

        # Combine Train and Val data
        X_full_views = {}
        for view_name in X_train_views.keys():
            X_full_views[view_name] = np.vstack(
                [X_train_views[view_name], X_val_views[view_name]]
            )

        y_full = np.concatenate([y_train, y_val])

        # Retrain only selected models
        # We pass the fitted models from Phase A to extract hyperparameters (specifically C for LR)
        final_models = self.retrain_selected(
            unique_selected, fitted_models_a, X_full_views, y_full
        )

        # 4. Inference
        print("\n" + "=" * 40)
        print("INFERENCE & SUBMISSION")
        print("=" * 40)

        print("Generating test predictions...")
        test_preds_dict = self.generate_predictions(final_models, X_test_views)

        print("Aggregating predictions...")
        final_probs = self.selector.predict(test_preds_dict)

        # 5. Save Submission
        self.save_submission(test_ids, classes, final_probs)

    def train_pool(self, pool, views, y):
        """
        Trains a dictionary of models on their respective views.
        """
        trained_models = {}
        for name, model in pool.items():
            # Extract view name (e.g., "Global_LR" -> "Global")
            view_name = name.split("_")[0]
            if view_name not in views:
                raise ValueError(
                    f"View '{view_name}' required for model '{name}' not found in data."
                )

            print(f"  Training {name}...")
            model.fit(views[view_name], y)
            trained_models[name] = model

        return trained_models

    def generate_predictions(self, models, views):
        """
        Generates probabilities for a dictionary of models.
        """
        preds = {}
        for name, model in models.items():
            view_name = name.split("_")[0]
            preds[name] = model.predict_proba(views[view_name])
        return preds

    def retrain_selected(self, selected_names, phase_a_models, full_views, y_full):
        """
        Retrains selected models on the full dataset.
        Transfers hyperparameters (C) from Phase A LogisticRegressionCV models.
        """
        retrained = {}
        # Get a fresh pool to ensure clean state for non-LR models
        fresh_pool = model_factory.get_expert_pool()

        for name in selected_names:
            print(f"  Retraining {name} on combined dataset...")
            view_name = name.split("_")[0]
            X = full_views[view_name]

            old_model = phase_a_models[name]

            # Handle Hyperparameter Transfer
            if isinstance(old_model, LogisticRegressionCV):
                # Extract the best C found during Phase A
                # We take the first element as a robust approximation for 'best C'
                try:
                    best_c = old_model.C_[0]
                    # Create a standard LogisticRegression with this C
                    new_model = LogisticRegression(
                        C=best_c,
                        solver=conf.LR_SOLVER,
                        max_iter=conf.LR_MAX_ITER,
                        n_jobs=-1,
                        random_state=conf.RANDOM_SEED,
                    )
                except Exception as e:
                    print(
                        f"    - Warning: Could not extract C from {name} ({e}). Re-running CV."
                    )
                    new_model = fresh_pool[name]
            else:
                # For LDA and CalibratedRF, we use the fresh instance.
                # LDA: No hyperparams to transfer (auto shrinkage).
                # CalibratedRF: Base estimator params are fixed in config. Calibration must be redone on full data.
                new_model = fresh_pool[name]

            new_model.fit(X, y_full)
            retrained[name] = new_model

        return retrained

    def save_submission(self, ids, classes, probs):
        """
        Saves the predictions to a CSV file in the required format.
        """
        # Ensure submission directory exists
        os.makedirs(os.path.dirname(conf.SUBMISSION_PATH), exist_ok=True)

        # Create DataFrame
        df = pd.DataFrame(probs, columns=classes)
        df.insert(0, conf.ID_COL, ids)

        # Save
        df.to_csv(conf.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {conf.SUBMISSION_PATH}")
        print("First 5 rows of submission:")
        print(df.head())
