import os
import pandas as pd
from library.config import SUBMISSION_PATH, ID_COL, TARGET_COL
from library.utils import set_seed, time_execution
from library.features import FeaturePipeline
from library.training import StackingTrainer


class Predictor:
    """
    Orchestrator for the Symmetric Dual-Topology Stacking Ensemble pipeline.
    Handles data loading, model training, and submission generation.
    """

    def __init__(self, debug_sample_size=None):
        """
        Args:
            debug_sample_size (int, optional): If set, limits the dataset size for debugging.
        """
        self.debug_sample_size = debug_sample_size

    @time_execution
    def generate_predictions(self):
        """
        Executes the full pipeline:
        1. Feature Engineering (loading cached views).
        2. Level 1 Stacking (CV) & Meta-Learner Training.
        3. Final Base Model Retraining (Validation-Guided).
        4. Test Set Prediction & Submission Generation.
        """
        # Ensure reproducibility
        set_seed()

        print("Initializing Feature Pipeline...")
        pipeline = FeaturePipeline(debug_sample_size=self.debug_sample_size)

        # ---------------------------------------------------------------------
        # 1. Load Feature Views
        # ---------------------------------------------------------------------
        # We load all 5 views for Train, Val, and Test splits

        # Contextual View (Metadata)
        meta_tr, meta_val, meta_te = pipeline.get_metadata_view()

        # Lexical Views (Text)
        lex_sp_tr, lex_sp_val, lex_sp_te = pipeline.get_lexical_sparse_view()
        lex_dn_tr, lex_dn_val, lex_dn_te = pipeline.get_lexical_dense_view()

        # Behavioral Views (Subreddit History)
        beh_sp_tr, beh_sp_val, beh_sp_te = pipeline.get_behavioral_sparse_view()
        beh_dn_tr, beh_dn_val, beh_dn_te = pipeline.get_behavioral_dense_view()

        # Get Targets and IDs
        y_train, y_val = pipeline.get_targets()
        test_ids = pipeline.get_test_ids()

        # Organize into dictionaries for the Trainer
        X_train_dict = {
            "metadata": meta_tr,
            "lexical_sparse": lex_sp_tr,
            "lexical_dense": lex_dn_tr,
            "behavioral_sparse": beh_sp_tr,
            "behavioral_dense": beh_dn_tr,
        }

        X_val_dict = {
            "metadata": meta_val,
            "lexical_sparse": lex_sp_val,
            "lexical_dense": lex_dn_val,
            "behavioral_sparse": beh_sp_val,
            "behavioral_dense": beh_dn_val,
        }

        X_test_dict = {
            "metadata": meta_te,
            "lexical_sparse": lex_sp_te,
            "lexical_dense": lex_dn_te,
            "behavioral_sparse": beh_sp_te,
            "behavioral_dense": beh_dn_te,
        }

        # ---------------------------------------------------------------------
        # 2. Model Training
        # ---------------------------------------------------------------------
        print("Initializing Stacking Trainer...")
        trainer = StackingTrainer(X_train_dict, y_train, X_val_dict, y_val)

        # Step A: Run Cross-Validation to get OOF preds and train Meta-Learner
        trainer.run_cv_and_meta_training()

        # Step B: Retrain base models on full data (or train+val) for final inference
        # This uses the validation set for early stopping in XGBoost models
        trainer.train_final_base_models()

        # ---------------------------------------------------------------------
        # 3. Prediction & Submission
        # ---------------------------------------------------------------------
        final_probs = trainer.predict(X_test_dict)

        # Create submission DataFrame
        submission_df = pd.DataFrame({ID_COL: test_ids, TARGET_COL: final_probs})

        # Save to disk
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(SUBMISSION_PATH, index=False)

        print(f"Submission generated successfully with {len(submission_df)} rows.")
        print(f"Saved to: {SUBMISSION_PATH}")
