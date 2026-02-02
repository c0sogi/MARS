import os
import numpy as np
import pandas as pd
from library import config
from library import utils
from library import data_processing
from library import modeling
from library import ensemble


class ExperimentManager:
    """
    Orchestrates the Stratified-Topology Precision-Generative Ensemble (STPGE) workflow.
    Manages data loading, expert training, ensemble selection, and final submission generation.
    """

    def __init__(self):
        self.data_manager = data_processing.DatasetManager()
        utils.set_seed(config.RANDOM_SEED)

    def _get_expert_id(self, expert_def):
        """Generates a unique string identifier for an expert configuration."""
        return (
            f"{expert_def['scope']}_{expert_def['topology']}_{expert_def['shrinkage']}"
        )

    def _get_cache_path(self, expert_id, split):
        """Constructs the file path for caching predictions."""
        filename = f"preds_{split}_{expert_id}.npy"
        return os.path.join(config.WORKING_DIR, filename)

    def run_selection_phase(self, load_cached_preds=True):
        """
        Phase 1: Train all candidates on Train split, evaluate on Val split,
        and run Greedy Forward Selection to find the best ensemble.

        Args:
            load_cached_preds (bool): If True, tries to load validation predictions from disk.

        Returns:
            dict: A dictionary mapping expert IDs to their ensemble weights.
        """
        print("Starting Phase 1: Expert Selection...")

        # 1. Load Data
        df_train = self.data_manager.get_data("train")
        df_val = self.data_manager.get_data("val")

        y_train = self.data_manager.get_targets(df_train)
        y_val = self.data_manager.get_targets(df_val)

        # 2. Generate/Load Predictions for all Experts
        val_predictions = {}

        # We need to capture the classes from at least one model to ensure alignment later
        self.classes_ = np.unique(y_train)

        for i, expert_def in enumerate(config.EXPERT_DEFINITIONS):
            expert_id = self._get_expert_id(expert_def)
            cache_path = self._get_cache_path(expert_id, "val")

            if load_cached_preds and os.path.exists(cache_path):
                # Load from cache
                preds = np.load(cache_path)
                val_predictions[expert_id] = preds
            else:
                # Train and Predict
                print(
                    f"Training Expert {i+1}/{len(config.EXPERT_DEFINITIONS)}: {expert_id}"
                )

                # Extract specific scope features
                X_train = self.data_manager.get_scope_slice(
                    df_train, expert_def["scope"]
                )
                X_val = self.data_manager.get_scope_slice(df_val, expert_def["scope"])

                # Initialize and Fit
                model = modeling.ExpertPipeline(
                    topology=expert_def["topology"], shrinkage=expert_def["shrinkage"]
                )
                model.fit(X_train, y_train)

                # Predict
                preds = model.predict_proba(X_val)

                # Save to cache
                np.save(cache_path, preds)
                val_predictions[expert_id] = preds

        # 3. Run Greedy Forward Selection
        print("Running Greedy Forward Selection...")
        selector = ensemble.GreedyForwardSelector(verbose=True)
        weights = selector.fit(val_predictions, y_val)

        print("Phase 1 Complete.")
        return weights

    def run_final_phase(self, weights):
        """
        Phase 2: Retrain selected experts on Combined (Train + Val) data,
        predict on Test data, and generate submission.

        Args:
            weights (dict): The selected experts and their weights from Phase 1.
        """
        print("Starting Phase 2: Final Retraining and Inference...")

        if not weights:
            raise ValueError("No experts selected. Cannot proceed to Phase 2.")

        # 1. Load All Data
        df_train = self.data_manager.get_data("train")
        df_val = self.data_manager.get_data("val")
        df_test = self.data_manager.get_data("test")

        # Combine Train and Val
        df_combined = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
        y_combined = self.data_manager.get_targets(df_combined)

        # Test IDs for submission
        test_ids = self.data_manager.get_ids(df_test)

        # 2. Retrain Selected Experts and Predict Test
        test_predictions_dict = {}

        # Identify unique experts to retrain (keys of the weights dict)
        # We map the ID back to the definition
        selected_ids = list(weights.keys())

        # Create a lookup for definitions
        def_lookup = {self._get_expert_id(d): d for d in config.EXPERT_DEFINITIONS}

        for expert_id in selected_ids:
            if expert_id not in def_lookup:
                raise ValueError(
                    f"Selected expert ID {expert_id} not found in definitions."
                )

            expert_def = def_lookup[expert_id]
            print(f"Retraining Selected Expert: {expert_id}")

            # Extract Features
            X_combined = self.data_manager.get_scope_slice(
                df_combined, expert_def["scope"]
            )
            X_test = self.data_manager.get_scope_slice(df_test, expert_def["scope"])

            # Train
            model = modeling.ExpertPipeline(
                topology=expert_def["topology"], shrinkage=expert_def["shrinkage"]
            )
            model.fit(X_combined, y_combined)

            # Predict
            preds = model.predict_proba(X_test)
            test_predictions_dict[expert_id] = preds

            # Ensure classes match expectations (sanity check)
            if not np.array_equal(model.pipeline.classes_, self.classes_):
                # In case a class was missing in train but present in combined (unlikely due to stratification)
                # or sorting order changed (unlikely with sklearn)
                print(f"Warning: Class mismatch for expert {expert_id}")

        # 3. Ensemble Aggregation
        print("Aggregating predictions...")
        # We can use the selector instance or just the logic since we have the weights
        # Re-instantiate a selector just to use the predict utility
        selector = ensemble.GreedyForwardSelector(verbose=False)
        # Manually inject weights
        selector.weights = weights

        final_proba = selector.predict(test_predictions_dict)

        # 4. Generate Submission
        print("Generating submission file...")
        self._save_submission(test_ids, final_proba, self.classes_)
        print("Phase 2 Complete.")

    def _save_submission(self, ids, proba, classes):
        """
        Formats and saves the submission CSV.
        """
        # Create DataFrame
        submission_df = pd.DataFrame(proba, columns=classes)
        submission_df.insert(0, "id", ids)

        # Load sample submission to verify column order and format
        sample_sub_path = os.path.join(config.INPUT_DIR, "sample_submission.csv")
        if os.path.exists(sample_sub_path):
            sample_df = pd.read_csv(sample_sub_path)
            # Ensure columns match sample submission (except ID)
            expected_cols = list(sample_df.columns)

            # Check if we have all columns
            # Note: sample_submission might have different order
            if set(expected_cols) != set(submission_df.columns):
                print(
                    "Warning: Submission columns do not match sample submission columns exactly."
                )

            # Reorder columns to match sample submission
            submission_df = submission_df[expected_cols]

        # Save
        output_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")

        # Validation check
        print(f"Submission shape: {submission_df.shape}")
        print(f"First 5 rows:\n{submission_df.head()}")
