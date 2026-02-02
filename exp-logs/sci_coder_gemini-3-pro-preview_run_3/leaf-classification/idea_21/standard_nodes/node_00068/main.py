import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
from tqdm import tqdm
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

# Import provided libraries
from library.config import Config
from library.utils import setup_logger, seed_everything
from library.feature_extraction import FeatureExtractor
from library.data_processing import OrthogonalDataManager
from library.training import OSLDETrainer
from library.inference import predict_test_set


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    logger = setup_logger("RunFile")
    logger.info("Starting OS-LDE Pipeline...")

    # 2. Feature Extraction (Train)
    # This step extracts features for the training set and caches them.
    logger.info("--- Step 1: Feature Extraction (Train) ---")
    extractor = FeatureExtractor()
    extractor.extract_features(dataset_type="train", load_cached_data=True)

    # 3. Training
    # Trains the expert ensemble using Stratified K-Fold.
    logger.info("--- Step 2: Training ---")
    trainer = OSLDETrainer()
    cv_score = trainer.train_and_evaluate()
    logger.info(f"Cross-Validation Log Loss: {cv_score}")

    # 4. Hold-out Validation
    # We must evaluate on the separate validation set defined in metadata/val.csv
    logger.info("--- Step 3: Hold-out Validation ---")

    if not os.path.exists(Config.VAL_METADATA_PATH):
        logger.error(f"Validation metadata not found at {Config.VAL_METADATA_PATH}")
        return

    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Extract features for validation set
    # We reuse the extractor instance. Since the library doesn't support 'val' key directly,
    # we process the batch manually.
    logger.info(f"Extracting features for {len(val_df)} validation images...")

    val_img_features = []
    val_ids = []
    val_labels = []

    # Extract Tabular Features for Val
    margin_cols = [c for c in val_df.columns if c.startswith("margin")]
    shape_cols = [c for c in val_df.columns if c.startswith("shape")]
    texture_cols = [c for c in val_df.columns if c.startswith("texture")]
    feature_cols = margin_cols + shape_cols + texture_cols
    val_tabular = val_df[feature_cols].values.astype(np.float32)

    device = extractor.device

    # Loop over validation images
    for idx, row in tqdm(
        val_df.iterrows(), total=len(val_df), desc="Val Extraction", mininterval=30.0
    ):
        img_id = row["id"]
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        species = row["species"]

        try:
            # Get batch of 12 rotated views
            img_batch = extractor.process_image_batch(full_path)
            img_batch = img_batch.to(device)

            with torch.no_grad():
                # Stream 1: DINOv2
                feats_dino = extractor.model_dino(img_batch)
                # Stream 2: ConvNeXt
                feats_conv = extractor.model_convnext(img_batch)
                # Concatenate
                combined = torch.cat([feats_dino, feats_conv], dim=1)  # (12, 2560)

            val_img_features.append(combined.cpu().numpy())
            val_ids.append(img_id)
            val_labels.append(species)

        except Exception as e:
            logger.error(f"Error processing val image {img_id}: {e}")
            # Fallback: Zero vector
            val_img_features.append(np.zeros((12, 2560), dtype=np.float32))
            val_ids.append(img_id)
            val_labels.append(species)

    val_img_features = np.array(val_img_features, dtype=np.float32)
    val_labels = np.array(val_labels)

    # Partition Validation Features into Centroids A, B, C
    data_manager = OrthogonalDataManager()
    # Accessing protected method to reuse logic
    val_centroids = data_manager._partition_and_aggregate(val_img_features)

    # Load Label Encoder
    le_path = os.path.join(Config.WORKING_DIR, "label_encoder.pkl")
    if not os.path.exists(le_path):
        logger.error("Label encoder not found. Training might have failed.")
        return

    label_encoder = joblib.load(le_path)
    val_y_encoded = label_encoder.transform(val_labels)
    classes = label_encoder.classes_

    # Inference on Validation Set
    logger.info("Running inference on validation set...")
    val_probs = np.zeros((len(val_labels), len(classes)))
    models_count = 0

    # Ensemble over all folds and experts
    for fold in range(Config.N_FOLDS):
        for expert_name in ["A", "B", "C"]:
            model_filename = f"model_fold_{fold}_expert_{expert_name}.pkl"
            model_path = os.path.join(Config.WORKING_DIR, model_filename)

            if os.path.exists(model_path):
                pipeline = joblib.load(model_path)

                # Prepare input: [Centroid | Tabular]
                X_val_expert = np.hstack([val_centroids[expert_name], val_tabular])

                probs = pipeline.predict_proba(X_val_expert)
                val_probs += probs
                models_count += 1

    if models_count == 0:
        logger.error("No trained models found.")
        return

    val_probs /= models_count

    # Post-processing (Clip & Normalize)
    val_probs = np.clip(val_probs, Config.PROB_CLIP_MIN, Config.PROB_CLIP_MAX)
    val_probs /= val_probs.sum(axis=1, keepdims=True)

    # Metric Calculation
    final_val_metric = log_loss(
        val_y_encoded, val_probs, labels=list(range(len(classes)))
    )

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_val_metric}")

    # 5. Failure Analysis
    logger.info("--- Step 4: Failure Analysis ---")

    # Calculate error per sample: -log(p_correct)
    # Get the probability assigned to the true class
    true_probs = val_probs[np.arange(len(val_y_encoded)), val_y_encoded]
    sample_errors = -np.log(true_probs)

    # Correlate with Tabular Features
    analysis_df = pd.DataFrame(val_tabular, columns=feature_cols)
    analysis_df["error_magnitude"] = sample_errors

    correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")
    top_correlations = correlations.abs().sort_values(ascending=False).head(5)

    print("\nTop 5 Features Correlated with Error Magnitude:")
    print(top_correlations)

    # 6. Submission Generation
    logger.info("--- Step 5: Submission Generation ---")

    # Extract features for test set
    extractor.extract_features(dataset_type="test", load_cached_data=True)

    # Generate submission file
    predict_test_set(load_cached_data=True)

    logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
