import os
import sys
import gc
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import nltk
import cv2

# Add current directory to sys.path to ensure library imports work correctly
sys.path.append(os.getcwd())

from library.config import Config
from library.train import run_training
from library.retrieval import (
    build_index,
    extract_embeddings,
    query_index,
    run_retrieval_inference,
)
from library.dataset import ChemicalDataset
from library.model import StoichiometryEncoder


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # Set up environment
    set_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 1. Training
    # -------------------------------------------------------------------------
    print("=" * 40)
    print("Step 1: Training Stoichiometry Encoder")
    print("=" * 40)
    # We use 2 epochs to ensure the pipeline finishes well within the 2-hour limit.
    # The model learns to count atoms, which converges relatively quickly.
    run_training(debug=False, epochs=2)

    # -------------------------------------------------------------------------
    # 2. Validation Assessment
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("Step 2: Validation Assessment")
    print("=" * 40)

    # Load Metadata
    print("Loading metadata...")
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    # Initialize Model and Load Weights
    device = Config.DEVICE
    print(f"Loading best model from {Config.MODEL_PATH}...")
    model = StoichiometryEncoder(
        backbone_name=Config.BACKBONE,
        pretrained=False,
        embedding_dim=Config.EMBEDDING_DIM,
        num_atoms=Config.NUM_ATOMS,
    )
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    else:
        print("CRITICAL WARNING: Model weights not found. Using random weights.")

    model = model.to(device)
    model.eval()

    # Build Training Index (Reference)
    # We use the 'val' mode for dataset to ensure deterministic transforms (no augmentation)
    print("Building retrieval index from training set...")
    train_dataset = ChemicalDataset(df_train, mode="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    # This builds the index and caches it to Config.TRAIN_EMBEDDINGS_PATH
    train_index = build_index(model, train_loader, device, load_cached_data=True)

    # Extract Validation Embeddings (Query)
    print("Extracting embeddings for validation set...")
    val_dataset = ChemicalDataset(df_val, mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_embeddings = extract_embeddings(model, val_loader, device)

    # Perform Retrieval
    print("Retrieving nearest neighbors...")
    # query_index returns indices relative to the train_index tensor
    nearest_indices = query_index(val_embeddings, train_index, device)

    # Map Indices to Labels
    print("Mapping indices to InChI labels...")
    train_labels = df_train["InChI"].values
    val_ground_truth = df_val["InChI"].values

    predicted_indices = nearest_indices.numpy()
    predicted_inchis = train_labels[predicted_indices]

    # Calculate Metric: Mean Levenshtein Distance
    print("Calculating Levenshtein distance...")
    distances = []
    # NLTK's edit_distance is robust. Loop is acceptable for ~380k items (~few minutes).
    for pred, truth in zip(predicted_inchis, val_ground_truth):
        d = nltk.edit_distance(pred, truth)
        distances.append(d)

    mean_levenshtein = np.mean(distances)
    print(f"Final Validation Metric: {mean_levenshtein}")

    # -------------------------------------------------------------------------
    # 3. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("Step 3: Failure Analysis")
    print("=" * 40)

    # Prepare analysis dataframe
    analysis_df = df_val.copy()
    analysis_df["levenshtein_dist"] = distances
    analysis_df["target_len"] = analysis_df["InChI"].apply(len)

    # 3a. Correlation with Target Length (Computed on full validation set)
    corr_len, _ = pearsonr(analysis_df["target_len"], analysis_df["levenshtein_dist"])
    print("Correlation with Error Magnitude:")
    print(f"  Target String Length: {corr_len:.4f}")

    # 3b. Correlation with Image Features (Computed on a subset for speed)
    # Reading 380k images from disk is too slow for the time limit, so we sample.
    sample_size = 2000
    print(f"  Sampling {sample_size} images to analyze visual feature correlations...")

    subset = analysis_df.sample(
        n=min(sample_size, len(analysis_df)), random_state=Config.SEED
    ).copy()
    widths = []
    heights = []
    aspect_ratios = []

    for _, row in subset.iterrows():
        path = os.path.join(Config.INPUT_DIR, row["file_path"])
        try:
            # Read image dimensions without loading full pixel data if possible,
            # but cv2.imread is standard.
            img = cv2.imread(path)
            if img is not None:
                h, w = img.shape[:2]
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h if h > 0 else 0)
            else:
                widths.append(np.nan)
                heights.append(np.nan)
                aspect_ratios.append(np.nan)
        except Exception:
            widths.append(np.nan)
            heights.append(np.nan)
            aspect_ratios.append(np.nan)

    subset["width"] = widths
    subset["height"] = heights
    subset["aspect_ratio"] = aspect_ratios

    # Drop failures
    subset = subset.dropna(subset=["width", "height"])

    if len(subset) > 0:
        corr_w, _ = pearsonr(subset["width"], subset["levenshtein_dist"])
        corr_h, _ = pearsonr(subset["height"], subset["levenshtein_dist"])
        corr_ar, _ = pearsonr(subset["aspect_ratio"], subset["levenshtein_dist"])

        print(f"  Image Width: {corr_w:.4f}")
        print(f"  Image Height: {corr_h:.4f}")
        print(f"  Aspect Ratio: {corr_ar:.4f}")
    else:
        print("  Could not compute image correlations (image loading failed).")

    # Cleanup to free memory for inference
    del train_index, val_embeddings, analysis_df, subset, distances, predicted_inchis
    torch.cuda.empty_cache()
    gc.collect()

    # -------------------------------------------------------------------------
    # 4. Submission Generation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("Step 4: Generating Submission")
    print("=" * 40)

    # run_retrieval_inference handles the full test set prediction pipeline.
    # It will reload the cached training index we built in Step 2.
    run_retrieval_inference(
        model_path=Config.MODEL_PATH,
        batch_size=Config.VAL_BATCH_SIZE,
        device=Config.DEVICE,
        load_cached_index=True,
        debug=False,
    )

    print("\nPipeline execution complete.")


if __name__ == "__main__":
    main()
