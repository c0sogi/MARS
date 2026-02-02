import os
import sys
import torch
import numpy as np
import pandas as pd
import cv2
import gc

# Add current directory to path to ensure library imports work correctly
sys.path.append(os.getcwd())

from library.config import Config, seed_everything
from library.trainer import Trainer
from library.inference import generate_submission, extract_features
from library.dataset import get_loaders
from library.model import WhaleModel


def main():
    # 1. Initialization
    seed_everything(Config.seed)
    print("Starting Whale Identification Pipeline...")

    # 2. Training
    # Initialize Trainer with debug=False to use the full dataset
    trainer = Trainer(debug=False)

    # Run training loop
    print("Initiating training...")
    trainer.fit()

    # Retrieve and print the final best metric
    final_metric = trainer.best_score
    print(f"Final Validation Metric: {final_metric:.10f}")

    # Clean up training resources to free memory for analysis
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Load the best model
    device = Config.device
    model_path = Config.model_path

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Cannot perform analysis.")
        return

    print(f"Loading model from {model_path}...")
    model = WhaleModel(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load DataLoaders for analysis (Validation and Gallery)
    # We use load_cached_data=True to speed up loading
    _, gallery_loader, val_loader, _, id2label = get_loaders(
        debug=False, load_cached_data=True
    )

    # Extract Embeddings
    print("Extracting embeddings for failure analysis...")
    val_feats, val_labels = extract_features(model, val_loader, device)
    gal_feats, gal_labels = extract_features(model, gallery_loader, device)

    # Compute Cosine Similarity Matrix
    # Move to GPU for efficient matrix multiplication
    val_tensor = torch.from_numpy(val_feats).to(device)
    gal_tensor = torch.from_numpy(gal_feats).to(device)

    # Similarity = Dot product of normalized vectors
    sim_matrix = torch.matmul(val_tensor, gal_tensor.t())

    # Calculate Per-Sample Error Magnitude
    print("Calculating per-sample error metrics...")

    # Retrieve top 100 neighbors to find the rank
    top_k = 100
    _, top_indices = torch.topk(sim_matrix, k=top_k, dim=1, largest=True)
    top_indices = top_indices.cpu().numpy()

    scores = []

    for i in range(len(val_labels)):
        true_label = val_labels[i]
        neighbors = top_indices[i]
        neighbor_labels = gal_labels[neighbors]

        # Find the rank of the true label in the neighbor list
        # np.where returns indices where condition is true
        matches = np.where(neighbor_labels == true_label)[0]

        if len(matches) > 0:
            rank = matches[0]  # 0-indexed rank
            # MAP@5 logic: Score is 1/(rank+1) if rank < 5, else 0
            if rank < 5:
                score = 1.0 / (rank + 1)
            else:
                score = 0.0
        else:
            # True label not in top K
            score = 0.0

        scores.append(score)

    scores = np.array(scores)
    # Error Magnitude: 1.0 is worst (score 0), 0.0 is best (score 1)
    error_magnitude = 1.0 - scores

    # Load Metadata to correlate features
    df_val = pd.read_csv(Config.meta_val_path)
    df_train = pd.read_csv(Config.meta_train_path)

    # Feature 1: Image Dimensions (Width, Height, Aspect Ratio)
    widths = []
    heights = []

    # Read image files to get dimensions
    for _, row in df_val.iterrows():
        fpath = os.path.join(Config.input_dir, row["file_path"])
        try:
            # Read image header or full image
            img = cv2.imread(fpath)
            if img is not None:
                h, w = img.shape[:2]
                widths.append(w)
                heights.append(h)
            else:
                widths.append(0)
                heights.append(0)
        except Exception:
            widths.append(0)
            heights.append(0)

    widths = np.array(widths)
    heights = np.array(heights)
    # Avoid division by zero
    aspect_ratios = np.divide(
        widths, heights, out=np.zeros_like(widths, dtype=float), where=heights != 0
    )

    # Feature 2: Class Frequency in Training Data
    train_counts = df_train["Id"].value_counts()
    class_freqs = []
    for _, row in df_val.iterrows():
        lbl = row["Id"]
        freq = train_counts.get(lbl, 0)
        class_freqs.append(freq)
    class_freqs = np.array(class_freqs)

    # Compute and Print Correlations
    def print_correlation(name, feature_values):
        if len(feature_values) != len(error_magnitude):
            return

        # Check for constant arrays to avoid NaN in correlation
        if np.std(feature_values) == 0 or np.std(error_magnitude) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(error_magnitude, feature_values)[0, 1]

        print(f"Correlation between Error Magnitude and {name}: {corr:.6f}")

    print_correlation("Image Width", widths)
    print_correlation("Image Height", heights)
    print_correlation("Aspect Ratio", aspect_ratios)
    print_correlation("Class Frequency", class_freqs)

    # 4. Submission Generation
    THRESHOLD = 0.638101

    if final_metric > THRESHOLD:
        print(
            f"\nValidation Metric ({final_metric:.6f}) exceeds threshold ({THRESHOLD})."
        )
        print("Generating submission file...")

        # Free memory before inference
        del model, val_feats, gal_feats, sim_matrix, top_indices, val_tensor, gal_tensor
        gc.collect()
        torch.cuda.empty_cache()

        # Generate submission
        generate_submission(load_cached_data=True)
    else:
        print(
            f"\nValidation Metric ({final_metric:.6f}) does not exceed threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
