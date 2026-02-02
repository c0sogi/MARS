import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import DataCacher, MRIDataset, get_transforms
from library.model import AsymmetricEfficientNet

logger = get_logger(__name__)


class Predictor:
    """
    Inference engine for Glioblastoma MGMT detection.
    Implements the 'Matched Ensemble' strategy:
    1. Generates multi-scale inputs (Stride 2 and Stride 5).
    2. Applies Test Time Augmentation (TTA).
    3. Aggregates predictions for robust output.
    """

    def __init__(self):
        """
        Initialize the Predictor.
        Sets up configuration, device, and ensures output directories exist.
        """
        Config.setup()
        seed_everything(Config.SEED)
        self.device = torch.device(Config.DEVICE)
        self.model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

        if not os.path.exists(self.model_path):
            logger.warning(
                f"Model file not found at {self.model_path}. Inference may fail if not running in a sequence."
            )

    def load_model(self):
        """
        Loads the AsymmetricEfficientNet architecture and restores weights.
        """
        logger.info(f"Loading model from {self.model_path}...")
        model = AsymmetricEfficientNet()

        # Load state dictionary
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
        else:
            logger.error("Model weights file is missing!")

        model.to(self.device)
        model.eval()
        return model

    def run(self):
        """
        Executes the full inference pipeline.

        Steps:
        1. Load Test Metadata.
        2. Cache Test Data (Circuit-Breaked & Fidelity-Aligned).
        3. Initialize Datasets for Stride 2 and Stride 5.
        4. Run Inference with TTA (Original, H-Flip, V-Flip).
        5. Aggregate and Save Submission.
        """
        logger.info("Starting Inference Pipeline...")

        # 1. Load Metadata
        if not os.path.exists(Config.TEST_METADATA):
            logger.error(f"Test metadata not found at {Config.TEST_METADATA}")
            return

        df_test = pd.read_csv(Config.TEST_METADATA)
        logger.info(f"Loaded metadata for {len(df_test)} test subjects.")

        # 2. Data Caching
        # We use the existing DataCacher which implements the required caching logic.
        test_cache = DataCacher.process_data(
            df_test, cache_key="test", load_cached_data=True
        )

        # 3. Datasets
        # We need two datasets to easily retrieve Stride 2 and Stride 5 volumes for the same subject.
        # transform=get_transforms("test") applies ToTensorV2 only.
        ds_stride2 = MRIDataset(
            data_cache=test_cache,
            metadata_df=df_test,
            transform=get_transforms("test"),
            stride_mode=2,
        )

        ds_stride5 = MRIDataset(
            data_cache=test_cache,
            metadata_df=df_test,
            transform=get_transforms("test"),
            stride_mode=5,
        )

        # 4. Load Model
        model = self.load_model()

        # 5. Inference Loop
        results = []

        with torch.no_grad():
            for i in range(len(df_test)):
                row = df_test.iloc[i]
                subject_id = row["BraTS21ID"]

                # Retrieve base tensors (C, H, W)
                # Dataset returns (image, label), we ignore the label
                img_s2, _ = ds_stride2[i]
                img_s5, _ = ds_stride5[i]

                # Construct TTA Batch
                # We want 6 predictions:
                # Stride 2: [Original, H-Flip, V-Flip]
                # Stride 5: [Original, H-Flip, V-Flip]

                tta_batch = []

                for img in [img_s2, img_s5]:
                    # Original
                    tta_batch.append(img)

                    # Horizontal Flip (Flip width dimension: dim 2)
                    tta_batch.append(torch.flip(img, dims=[2]))

                    # Vertical Flip (Flip height dimension: dim 1)
                    tta_batch.append(torch.flip(img, dims=[1]))

                # Stack into a single batch: Shape (6, 12, H, W)
                batch_tensor = torch.stack(tta_batch).to(self.device)

                # Forward Pass
                logits = model(batch_tensor)
                probs = torch.sigmoid(logits)  # Shape (6, 1)

                # Matched Ensemble Aggregation
                avg_prob = torch.mean(probs).item()

                results.append({"BraTS21ID": subject_id, "MGMT_value": avg_prob})

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(df_test)} subjects.")

        # 6. Save Submission
        submission_df = pd.DataFrame(results)

        # Ensure correct column order and sorting
        submission_df = submission_df[["BraTS21ID", "MGMT_value"]]
        submission_df = submission_df.sort_values("BraTS21ID")

        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_FILE}")

        # Print first few rows for verification
        logger.info("Submission Head:")
        print(submission_df.head())
