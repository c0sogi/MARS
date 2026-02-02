import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import DataCacher, MRIDataset, get_transforms
from library.model import AsymmetricEfficientNet
from library.train import train_model
from library.inference import Predictor


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    # Initialize Config and Directories
    Config.setup()

    # Set reproducibility
    seed_everything(Config.SEED)

    # Configure for Fast Baseline
    # We reduce epochs to ensure the run completes quickly while still learning.
    # The dataset is small (~500 samples), so 10 epochs is sufficient for a baseline.
    Config.EPOCHS = 10

    logger = get_logger(__name__)
    logger.info("Starting runfile.py execution...")

    # --------------------------------------------------------------------------
    # 2. Training Pipeline
    # --------------------------------------------------------------------------
    logger.info("--- Initiating Training Phase ---")
    # train_model() handles data loading, model init, training loop, and saving best_model.pth
    train_model()

    # --------------------------------------------------------------------------
    # 3. Validation & Failure Analysis
    # --------------------------------------------------------------------------
    logger.info("--- Initiating Validation & Failure Analysis Phase ---")

    device = torch.device(Config.DEVICE)
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        logger.error("Best model not found! Training might have failed.")
        sys.exit(1)

    # Load the best model
    model = AsymmetricEfficientNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load Validation Metadata and Cache
    # Note: DataCacher will load the cache created during training (efficient)
    df_val = pd.read_csv(Config.VAL_METADATA)
    val_cache = DataCacher.process_data(df_val, cache_key="val", load_cached_data=True)

    # Initialize Datasets for Matched Ensemble (Stride 2 and Stride 5)
    ds_s2 = MRIDataset(
        val_cache, df_val, transform=get_transforms("val"), stride_mode=2
    )
    ds_s5 = MRIDataset(
        val_cache, df_val, transform=get_transforms("val"), stride_mode=5
    )

    y_true = []
    y_pred = []
    errors = []

    # Storage for feature extraction (Slice counts per modality)
    # We will correlate error with the amount of data available (depth)
    meta_features = {
        "depth_FLAIR": [],
        "depth_T1w": [],
        "depth_T1wCE": [],
        "depth_T2w": [],
    }

    logger.info("Running Validation Inference...")

    with torch.no_grad():
        for i in range(len(ds_s2)):
            # 1. Get Data
            # Note: ds_s2 and ds_s5 are aligned by index because they use the same metadata df
            img_s2, label_tensor = ds_s2[i]
            img_s5, _ = ds_s5[i]

            label = label_tensor.item()

            # 2. Construct TTA Batch
            # Strategy: [Stride2_Orig, Stride2_HFlip, Stride2_VFlip, Stride5_Orig, Stride5_HFlip, Stride5_VFlip]
            batch_imgs = []
            for img in [img_s2, img_s5]:
                batch_imgs.append(img)
                batch_imgs.append(torch.flip(img, dims=[2]))  # Horizontal Flip (W)
                batch_imgs.append(torch.flip(img, dims=[1]))  # Vertical Flip (H)

            batch_tensor = torch.stack(batch_imgs).to(device)

            # 3. Inference
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()

            # 4. Ensemble Aggregation
            avg_prob = np.mean(probs)

            # 5. Store Results
            y_true.append(label)
            y_pred.append(avg_prob)

            # 6. Error Calculation for Analysis
            error = abs(label - avg_prob)
            errors.append(error)

            # 7. Extract Features for Analysis
            subject_id = ds_s2.metadata.iloc[i]["BraTS21ID"]
            subj_data = val_cache["images"][subject_id]
            for mod in Config.MODALITIES:
                # Shape is (Depth, Height, Width)
                depth = subj_data[mod].shape[0]
                meta_features[f"depth_{mod}"].append(depth)

    # Calculate Final Metric
    final_auc = roc_auc_score(y_true, y_pred)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # Failure Analysis: Correlation
    print("\nFailure Analysis (Correlation between Error Magnitude and Volume Depth):")
    error_arr = np.array(errors)

    for feat_name, values in meta_features.items():
        val_arr = np.array(values)
        # Handle constant input case to avoid NaN
        if np.std(val_arr) == 0 or np.std(error_arr) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(error_arr, val_arr)[0, 1]
        print(f"Correlation with {feat_name}: {corr:.6f}")

    # --------------------------------------------------------------------------
    # 4. Submission Generation
    # --------------------------------------------------------------------------
    # Threshold defined in task requirements
    THRESHOLD = 0.6321818181818182

    if final_auc > THRESHOLD:
        logger.info(
            f"Validation AUC ({final_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        predictor = Predictor()
        predictor.run()
    else:
        logger.info(
            f"Validation AUC ({final_auc}) does not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
