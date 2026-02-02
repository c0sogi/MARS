import os
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library.config import Config
from library.dataset import get_dataloader
from library.model import BraTS2DCNN
from library.trainer import run_training, generate_submission, set_seed


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for Fast Baseline Execution
    Config.NUM_EPOCHS = 20  # Increased to allow convergence with more input channels
    Config.BATCH_SIZE = 32  # Maintained for stability

    # Set global seeds
    set_seed(Config.SEED)

    # ==========================================
    # 2. Training
    # ==========================================
    # run_training handles the loop and saves the best model to Config.MODEL_SAVE_PATH
    run_training()

    # ==========================================
    # 3. Validation & Metric Calculation
    # ==========================================
    # We reload the best model to perform specific analysis and get the exact metric
    device = Config.DEVICE
    model = BraTS2DCNN().to(device)

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print("Error: Model file not found. Training may have failed.")
        return

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Load Validation Data
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)
    val_loader = get_dataloader(
        df_val,
        split="val",
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
        debug=Config.DEBUG,
    )

    all_targets = []
    all_probs = []

    # Inference loop (No Grad)
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)

            # Use autocast for inference
            with torch.cuda.amp.autocast():
                logits = model(inputs)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_targets.extend(targets.numpy().flatten())
            all_probs.extend(probs)

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Compute and Print Metric
    val_auc = roc_auc_score(all_targets, all_probs)
    print(f"Final Validation Metric: {val_auc}")

    # ==========================================
    # 4. Failure Analysis
    # ==========================================
    print("Performing Failure Analysis...")

    # Calculate absolute error
    errors = np.abs(all_targets - all_probs)

    # Extract features for correlation analysis
    # We will count the number of files for each modality in the validation set
    # to see if sequence length/data quantity correlates with error.
    meta_features = []

    # Iterate through the dataframe used for validation
    # Note: If DEBUG is True, df_val might be larger than what was processed by loader if loader subsetted it.
    # But get_dataloader with debug=True subsets the df.
    # To be safe, we assume the loader processed df_val.head(len(errors))

    df_val_analyzed = df_val.iloc[: len(errors)].copy()

    for _, row in df_val_analyzed.iterrows():
        f_feats = {}
        for mod in Config.SELECTED_MODALITIES:
            # Construct path: input/train/00xxx/MODALITY
            # Metadata contains relative path 'train/00xxx/MODALITY'
            rel_path = row[f"{mod.lower()}_path"]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)

            count = 0
            if os.path.exists(full_path):
                try:
                    # Fast count of files
                    count = len(
                        [f for f in os.listdir(full_path) if f.endswith(".dcm")]
                    )
                except OSError:
                    pass
            f_feats[f"{mod}_count"] = count
        meta_features.append(f_feats)

    df_features = pd.DataFrame(meta_features)
    df_features["error"] = errors

    # Calculate correlation
    corr = df_features.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Input Features:")
    print(corr)

    # ==========================================
    # 5. Submission
    # ==========================================
    THRESHOLD = 0.5698181818181819

    if val_auc > THRESHOLD:
        generate_submission()
    else:
        print(
            f"Validation AUC ({val_auc}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
