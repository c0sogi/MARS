import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import format_submission, kendall_tau
from library.backbone import BackboneTrainer
from library.feature_extractor import FeatureEngineer
from library.ranker import LGBMRanker
from library.data_loader import NotebookTextLoader


class InferencePipeline:
    """
    Orchestrates the inference process for the Multi-Scale Structural Heatmap Regressor.
    Handles feature extraction, prediction, order reconstruction, and submission generation.
    """

    def __init__(self):
        """
        Initializes the pipeline components.
        """
        # Initialize Backbone (loads fine-tuned model if available)
        self.backbone = BackboneTrainer()

        # Initialize Feature Engineer
        self.feature_engineer = FeatureEngineer(backbone_model=self.backbone)

        # Initialize Ranker (will load model lazily during predict)
        self.ranker = LGBMRanker()

    def run_test_inference(self, load_cached_features=True):
        """
        Runs the full inference pipeline on the test set.

        1. Extracts features for test notebooks.
        2. Predicts markdown positions using LightGBM.
        3. Reconstructs the full cell order.
        4. Generates the submission CSV.

        Args:
            load_cached_features (bool): Whether to use cached features if available.
        """
        print("Starting Test Inference...")

        # 1. Extract Features
        # The FeatureEngineer handles caching internally based on the mode='test' and Config paths.
        test_features = self.feature_engineer.extract_features(
            metadata_path=Config.TEST_PATH,
            mode="test",
            load_cached_data=load_cached_features,
        )

        # 2. Generate Predictions
        print("Predicting markdown positions...")
        if len(test_features) > 0:
            predictions = self.ranker.predict(test_features)
            test_features["pred"] = predictions
        else:
            print(
                "Warning: No test features found (possibly empty notebooks or no markdown)."
            )
            # Create empty column if dataframe is empty but exists, or handle downstream
            if "pred" not in test_features.columns:
                test_features["pred"] = []

        # 3. Reconstruct Cell Orders
        print("Reconstructing cell orders...")

        # Load test metadata to iterate through notebooks
        loader = NotebookTextLoader(Config.TEST_PATH)

        # Group predictions by notebook ID for efficient retrieval
        if "id" in test_features.columns and not test_features.empty:
            grouped_preds = test_features.groupby("id")
        else:
            grouped_preds = None

        submission_ids = []
        submission_orders = []

        for i in range(len(loader)):
            nb_data = loader[i]
            nb_id = nb_data["id"]

            # Get the fixed code cells (anchors)
            # nb_data['code_cells'] is list of (id, source)
            code_cells = [c[0] for c in nb_data["code_cells"]]
            n_code = len(code_cells)

            # Prepare list of (rank, cell_id)
            ranked_cells = []

            # Assign ranks to code cells: index + 0.5
            # This places code cell 0 at 0.5, code cell 1 at 1.5, etc.
            # This ensures they maintain their relative order while allowing markdown to slot in between.
            for idx, cid in enumerate(code_cells):
                ranked_cells.append((float(idx) + 0.5, cid))

            # Assign ranks to markdown cells based on predictions
            if grouped_preds is not None and nb_id in grouped_preds.groups:
                nb_pred_df = grouped_preds.get_group(nb_id)

                for _, row in nb_pred_df.iterrows():
                    # Prediction y is normalized [0, 1] representing relative position among code cells.
                    # Rank = y * n_code.
                    # Example: y=0 -> rank 0 (before code 0). y=1 -> rank n_code (after last code).
                    rank = row["pred"] * n_code
                    ranked_cells.append((rank, row["cell_id"]))
            else:
                # Fallback: If no predictions (e.g. no markdown, or n_code=0 skipped in FE),
                # append any remaining markdown cells at the end.

                # Get markdown IDs from loader
                md_cells = [m[0] for m in nb_data["markdown_cells"]]

                for cid in md_cells:
                    # Put at the end
                    ranked_cells.append((float(n_code) + 1.0, cid))

            # Sort by rank
            ranked_cells.sort(key=lambda x: x[0])

            # Extract IDs
            final_order = [x[1] for x in ranked_cells]

            submission_ids.append(nb_id)
            submission_orders.append(final_order)

        # 4. Save Submission
        format_submission(submission_ids, submission_orders, Config.SUBMISSION_PATH)
        print(f"Inference complete. Submission saved to {Config.SUBMISSION_PATH}")

    def run_validation_inference(self, load_cached_features=True):
        """
        Runs inference on the validation set and computes the Kendall Tau score.

        Args:
            load_cached_features (bool): Whether to use cached features.

        Returns:
            float: The Kendall Tau score.
        """
        print("Starting Validation Inference...")

        val_features = self.feature_engineer.extract_features(
            metadata_path=Config.VAL_PATH,
            mode="val",
            load_cached_data=load_cached_features,
        )

        if len(val_features) == 0:
            print("No validation features found.")
            return 0.0

        predictions = self.ranker.predict(val_features)
        val_features["pred"] = predictions

        loader = NotebookTextLoader(Config.VAL_PATH)
        grouped_preds = val_features.groupby("id")

        predicted_orders = []
        ground_truths = []

        print("Reconstructing validation orders...")
        for i in range(len(loader)):
            nb_data = loader[i]
            nb_id = nb_data["id"]
            gt_order = nb_data["cell_order"]

            # If ground truth is missing, skip
            if not gt_order:
                continue

            code_cells = [c[0] for c in nb_data["code_cells"]]
            n_code = len(code_cells)

            ranked_cells = []
            for idx, cid in enumerate(code_cells):
                ranked_cells.append((float(idx) + 0.5, cid))

            if nb_id in grouped_preds.groups:
                nb_pred_df = grouped_preds.get_group(nb_id)
                for _, row in nb_pred_df.iterrows():
                    rank = row["pred"] * n_code
                    ranked_cells.append((rank, row["cell_id"]))
            else:
                md_cells = [m[0] for m in nb_data["markdown_cells"]]
                for cid in md_cells:
                    ranked_cells.append((float(n_code) + 1.0, cid))

            ranked_cells.sort(key=lambda x: x[0])
            final_order = [x[1] for x in ranked_cells]

            predicted_orders.append(final_order)
            ground_truths.append(gt_order)

        score = kendall_tau(ground_truths, predicted_orders)
        print(f"Validation Kendall Tau: {score}")
        return score
