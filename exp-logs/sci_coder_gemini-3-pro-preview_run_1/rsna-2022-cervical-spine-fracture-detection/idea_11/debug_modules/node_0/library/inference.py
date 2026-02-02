import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from library.config import Config
from library.utils import get_device, load_checkpoint, format_submission
from library.models import UNetLocalizer, DualStreamEncoder, AnatomicalGRU
from library.training import extract_features_and_cache
from library.data import get_sequence_dataloader


class InferencePipeline:
    """
    Orchestrates the end-to-end prediction pipeline for the RSNA Cervical Spine Fracture Detection task.
    Executes the 3-stage architecture:
    1. Anatomical Localization (U-Net)
    2. Dual-Stream Feature Encoding (ResNet)
    3. Sequence Aggregation (Bi-GRU)
    """

    def __init__(self):
        self.device = get_device()
        self.stage1_model = None
        self.stage2_model = None
        self.stage3_model = None

    def load_models(self):
        """
        Initializes model architectures and loads weights from checkpoints.
        """
        print("Loading models...")

        # Stage 1: Localizer (U-Net)
        # Used to find ROIs and generate anatomical probability profiles
        self.stage1_model = UNetLocalizer(
            num_classes=Config.STAGE1_NUM_CLASSES, pretrained=False
        ).to(self.device)
        load_checkpoint(self.stage1_model, Config.STAGE1_CHECKPOINT_PATH)
        self.stage1_model.eval()

        # Stage 2: Encoder (Dual-Stream ResNet)
        # Used to extract visual features from Local (cropped) and Global (resized) views
        self.stage2_model = DualStreamEncoder(pretrained=False).to(self.device)
        load_checkpoint(self.stage2_model, Config.STAGE2_CHECKPOINT_PATH)
        self.stage2_model.eval()

        # Stage 3: Aggregator (Bi-GRU)
        # Used to aggregate sequence features into final probabilities
        # Input dim: 1024 (Visual Features) + 8 (Anatomical Profile)
        self.stage3_model = AnatomicalGRU(input_dim=1032).to(self.device)
        load_checkpoint(self.stage3_model, Config.STAGE3_CHECKPOINT_PATH)
        self.stage3_model.eval()

        print("All models loaded successfully.")

    def run(self, debug=Config.DEBUG):
        """
        Executes the inference pipeline:
        1. Feature Extraction (Stage 1 & 2) -> Caches features to disk
        2. Sequence Aggregation (Stage 3) -> Generates probabilities
        3. Submission Formatting -> Saves CSV

        Args:
            debug (bool): If True, runs on a small subset of data.
        """
        if self.stage1_model is None:
            self.load_models()

        # ---------------------------------------------------------------------
        # Step 1: Feature Extraction
        # ---------------------------------------------------------------------
        # This step processes the raw DICOM images. It uses the helper function
        # from library.training which encapsulates the logic for:
        # - Loading 3D volumes
        # - Running Stage 1 to get ROIs
        # - Cropping/Resizing for Stage 2
        # - Running Stage 2 to get features
        # - Caching the resulting feature sequences to .npy files
        print(f"Starting feature extraction (Debug={debug})...")
        extract_features_and_cache(
            self.stage1_model, self.stage2_model, split="test", debug=debug
        )

        # ---------------------------------------------------------------------
        # Step 2: Sequence Inference
        # ---------------------------------------------------------------------
        print("Starting sequence inference...")

        # Load the test dataset using the cached features
        # The dataloader handles padding variable length sequences
        test_loader = get_sequence_dataloader(
            batch_size=Config.STAGE3_BATCH_SIZE, split="test"
        )

        all_preds = []

        with torch.no_grad():
            # Iterate through patients
            for features, _, lengths in tqdm(
                test_loader, desc="Predicting", disable=False
            ):
                features = features.to(self.device)

                # Forward pass through Bi-GRU
                # Returns logits for C1-C7 and patient_overall
                logits = self.stage3_model(features, lengths)

                # Apply Sigmoid to get probabilities
                probs = torch.sigmoid(logits)

                all_preds.append(probs.cpu().numpy())

        if not all_preds:
            print("Warning: No predictions generated.")
            return

        # Concatenate all batches: (N_patients, 8)
        predictions = np.concatenate(all_preds, axis=0)

        # ---------------------------------------------------------------------
        # Step 3: Submission Generation
        # ---------------------------------------------------------------------
        print("Generating submission file...")

        # Retrieve StudyInstanceUIDs to ensure alignment
        # The dataloader loads based on test_metadata.csv order, so we load the same list
        test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
        study_ids = test_meta["StudyInstanceUID"].tolist()

        # Verify alignment
        if len(predictions) != len(study_ids):
            print(
                f"Note: Prediction count ({len(predictions)}) vs Study count ({len(study_ids)})."
            )
            # In debug mode, this might differ if metadata wasn't filtered,
            # but get_sequence_dataloader creates dummy zero-features for missing cache files,
            # so lengths should strictly align.

        # Format and save
        # This function expands the (N, 8) array into the required row-based format
        format_submission(study_ids, predictions, output_path=Config.SUBMISSION_PATH)
        print(f"Inference complete. Submission saved to {Config.SUBMISSION_PATH}")
