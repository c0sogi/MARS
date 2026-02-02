import os
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import setup_logger, set_seed
from library.data_loader import DataLoader
from library.preprocessing import RobustPreprocessor
from library.models import ExpertLibrary
from library.ensemble_selection import GreedyEnsembleSelector

# Initialize logger
logger = setup_logger("pipeline")


class Pipeline:
    """
    Orchestrates the Orthogonal-Basis Hybrid-Complexity Ensemble (OBHCE) pipeline.

    Stages:
    1. Data Loading & Feature Extraction (via DataLoader)
    2. Phase 1: Model Selection & Weight Optimization (Train/Val split)
    3. Phase 2: Final Retraining (Train+Val combined)
    4. Inference: Test set prediction and submission generation
    """

    def __init__(self):
        self.data_loader = DataLoader()
        self.expert_lib = ExpertLibrary()
        self.selector = GreedyEnsembleSelector()

        # Data containers
        self.data = None

        # Preprocessors (Phase 1)
        self.p1_global_scaler = RobustPreprocessor()
        self.p1_zernike_scaler = RobustPreprocessor()

        # Preprocessors (Phase 2)
        self.p2_global_scaler = RobustPreprocessor()
        self.p2_zernike_scaler = RobustPreprocessor()

        # State
        self.ensemble_weights = {}
        self.retrained_models = {}

    def run(self, load_cached=True):
        """
        Executes the full pipeline.

        Args:
            load_cached (bool): Whether to load cached Zernike features.
        """
        set_seed(Config.RANDOM_SEED)

        # 1. Load Data
        logger.info("Initializing Data Loading...")
        self.data = self.data_loader.load_data(load_cached=load_cached)

        # 2. Phase 1: Selection
        self._run_phase_1_selection()

        # 3. Phase 2: Retraining
        self._run_phase_2_retraining()

        # 4. Inference
        self._run_inference()

        logger.info("Pipeline execution completed successfully.")

    def _run_phase_1_selection(self):
        """
        Phase 1: Train all candidates on Train split, evaluate on Val split,
        and select the best ensemble using Greedy Forward Selection.
        """
        logger.info("\n=== Phase 1: Selection ===")

        # Unpack data
        X_train_g = self.data["train"]["X_global"]
        X_train_z = self.data["train"]["X_zernike"]
        y_train = self.data["train"]["y"]

        X_val_g = self.data["val"]["X_global"]
        X_val_z = self.data["val"]["X_zernike"]
        y_val = self.data["val"]["y"]

        # 1. Fit Preprocessors on Train
        logger.info("Fitting preprocessors on Training split...")
        X_train_g_trans = self.p1_global_scaler.fit_transform(X_train_g)
        X_train_z_trans = self.p1_zernike_scaler.fit_transform(X_train_z)

        # 2. Transform Validation
        X_val_g_trans = self.p1_global_scaler.transform(X_val_g)
        X_val_z_trans = self.p1_zernike_scaler.transform(X_val_z)

        # 3. Instantiate Experts
        tier1_experts = self.expert_lib.get_tier1_experts()
        tier2_experts = self.expert_lib.get_tier2_experts()

        val_predictions = {}

        # 4. Train Tier 1 (LDA on Global)
        logger.info(f"Training {len(tier1_experts)} Tier 1 (LDA) experts...")
        for name, model in tier1_experts.items():
            model.fit(X_train_g_trans, y_train)
            preds = model.predict_proba(X_val_g_trans)
            val_predictions[name] = preds

        # 5. Train Tier 2 (QDA on Zernike)
        logger.info(f"Training {len(tier2_experts)} Tier 2 (QDA) experts...")
        for name, model in tier2_experts.items():
            model.fit(X_train_z_trans, y_train)
            preds = model.predict_proba(X_val_z_trans)
            val_predictions[name] = preds

        # 6. Run Selection
        logger.info("Running Greedy Ensemble Selection...")
        self.selector.fit(val_predictions, y_val)
        self.ensemble_weights = self.selector.get_selected_experts()

        if not self.ensemble_weights:
            raise RuntimeError("Ensemble selection failed to select any models.")

        logger.info(f"Phase 1 Complete. Selected {len(self.ensemble_weights)} experts.")

    def _run_phase_2_retraining(self):
        """
        Phase 2: Retrain ONLY the selected experts on the combined Train + Val dataset.
        """
        logger.info("\n=== Phase 2: Retraining ===")

        # 1. Combine Data
        X_full_g = np.vstack(
            [self.data["train"]["X_global"], self.data["val"]["X_global"]]
        )
        X_full_z = np.vstack(
            [self.data["train"]["X_zernike"], self.data["val"]["X_zernike"]]
        )
        y_full = np.concatenate([self.data["train"]["y"], self.data["val"]["y"]])

        logger.info(f"Combined Training Set Size: {len(y_full)}")

        # 2. Fit New Preprocessors on Full Data
        logger.info("Fitting preprocessors on Combined (Train + Val) data...")
        X_full_g_trans = self.p2_global_scaler.fit_transform(X_full_g)
        X_full_z_trans = self.p2_zernike_scaler.fit_transform(X_full_z)

        # 3. Retrain Selected Experts
        # We need to re-instantiate to ensure clean state
        all_tier1 = self.expert_lib.get_tier1_experts()
        all_tier2 = self.expert_lib.get_tier2_experts()

        self.retrained_models = {}

        for name in self.ensemble_weights.keys():
            logger.info(f"Retraining selected expert: {name}")

            if name in all_tier1:
                # Tier 1: LDA -> Global Features
                model = all_tier1[name]
                model.fit(X_full_g_trans, y_full)
                self.retrained_models[name] = model

            elif name in all_tier2:
                # Tier 2: QDA -> Zernike Features
                model = all_tier2[name]
                model.fit(X_full_z_trans, y_full)
                self.retrained_models[name] = model

            else:
                logger.warning(
                    f"Selected model {name} not found in library during retraining."
                )

    def _run_inference(self):
        """
        Inference: Generate predictions for the Test set using retrained models
        and aggregated using the learned ensemble weights.
        """
        logger.info("\n=== Inference ===")

        X_test_g = self.data["test"]["X_global"]
        X_test_z = self.data["test"]["X_zernike"]
        ids_test = self.data["test"]["ids"]

        # 1. Transform Test Data (using Phase 2 preprocessors)
        X_test_g_trans = self.p2_global_scaler.transform(X_test_g)
        X_test_z_trans = self.p2_zernike_scaler.transform(X_test_z)

        # 2. Generate Predictions
        test_predictions = {}

        for name, model in self.retrained_models.items():
            if name.startswith("LDA"):
                preds = model.predict_proba(X_test_g_trans)
            elif name.startswith("QDA"):
                preds = model.predict_proba(X_test_z_trans)
            else:
                continue
            test_predictions[name] = preds

        # 3. Aggregate Predictions
        # Using the selector's predict method which handles weighting, normalization and clipping
        final_probs = self.selector.predict(test_predictions)

        # 4. Create Submission DataFrame
        le = self.data["label_encoder"]
        species_names = le.classes_

        # Verify shape
        if final_probs.shape[1] != len(species_names):
            logger.error(
                f"Prediction shape {final_probs.shape} does not match class count {len(species_names)}"
            )
            raise ValueError("Dimension mismatch in predictions.")

        submission_df = pd.DataFrame(final_probs, columns=species_names)
        submission_df.insert(0, "id", ids_test)

        # 5. Save Submission
        save_path = Config.SUBMISSION_PATH
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        submission_df.to_csv(save_path, index=False)

        logger.info(f"Submission saved to {save_path}")
        logger.info(f"Submission shape: {submission_df.shape}")

        # Print first few rows for verification
        logger.info("First 5 rows of submission:")
        logger.info(submission_df.head().to_string())
