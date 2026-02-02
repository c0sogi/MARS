import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import (
    get_dataloaders,
    get_label_mapping,
    HotelDataset,
    get_transforms,
)
from library.model import HotelModel
from library.engine import run_training, generate_embeddings
from library.inference import predict


def main():
    # --- 1. Configure for Optimized Training ---
    # Override default config for performance within time limit
    Config.EPOCHS = 10
    Config.WARMUP_EPOCHS = 1  # Warmup head first to stabilize ArcFace training
    Config.BATCH_SIZE = 8  # Reduced to fit 16GB VRAM
    Config.NUM_WORKERS = 12  # Maximize CPU usage
    Config.DEBUG = False  # Use full dataset for valid metrics

    # Ensure output directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(
        f"Configuration: Epochs={Config.EPOCHS}, Batch={Config.BATCH_SIZE}, Device={Config.DEVICE}"
    )

    # --- 2. Setup & Data Loading ---
    seed_everything(Config.SEED)

    # Load DataLoaders
    # get_dataloaders returns (train_loader, val_loader, test_loader, num_classes)
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(debug=False)

    # --- 3. Model Initialization ---
    model = HotelModel(num_classes=num_classes)
    model.to(Config.DEVICE)

    # --- 4. Training ---
    print("\n=== Starting Training ===")
    run_training(
        model, train_loader, val_loader, num_epochs=Config.EPOCHS, device=Config.DEVICE
    )

    # --- 5. Validation Inference (MAP@5 Calculation) ---
    print("\n=== Starting Validation Inference ===")

    # Load the best model saved during training
    best_model_path = Config.BEST_MODEL_PATH
    print(f"Loading best model from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
    model.eval()

    # Prepare Label Mappings
    id_to_idx, idx_to_id = get_label_mapping(load_cached_data=True)

    # A. Generate Gallery Embeddings (Train Set)
    # We need a clean loader for the gallery (no shuffle, val transforms)
    print("Generating Gallery Embeddings (Train Set)...")
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    train_df["label_idx"] = train_df["hotel_id"].map(id_to_idx).astype(int)

    gallery_dataset = HotelDataset(
        train_df, transform=get_transforms(mode="val"), mode="train"
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    gallery_emb, gallery_labels = generate_embeddings(
        model, gallery_loader, Config.DEVICE
    )

    # B. Generate Query Embeddings (Validation Set)
    print("Generating Query Embeddings (Validation Set)...")
    # val_loader already uses val transforms and shuffle=False
    query_emb, query_labels = generate_embeddings(model, val_loader, Config.DEVICE)

    # C. Retrieval (Cosine Similarity)
    print("Computing Similarity Matrix...")
    gallery_tensor = torch.from_numpy(gallery_emb).to(Config.DEVICE)
    query_tensor = torch.from_numpy(query_emb).to(Config.DEVICE)

    # Matrix Multiplication: (N_query, Dim) @ (Dim, N_gallery) -> (N_query, N_gallery)
    # Note: Embeddings are normalized in generate_embeddings, so this is Cosine Similarity
    sim_matrix = torch.matmul(query_tensor, gallery_tensor.t())

    # Retrieve Top K (Retrieve 50 to ensure we find 5 unique hotels)
    print("Retrieving Top Candidates...")
    topk_vals, topk_indices = torch.topk(sim_matrix, k=50, dim=1)
    topk_indices = topk_indices.cpu().numpy()

    # D. Calculate MAP@5
    print("Calculating MAP@5...")
    ap_scores = []

    for i in range(len(query_labels)):
        # Ground Truth
        true_label_idx = query_labels[i]
        true_hotel_id = idx_to_id[true_label_idx]

        # Predictions
        indices = topk_indices[i]
        retrieved_label_indices = gallery_labels[indices]

        # Get top 5 unique hotels
        unique_preds = []
        seen = set()
        for l_idx in retrieved_label_indices:
            if l_idx not in seen:
                pred_hotel_id = idx_to_id[l_idx]
                unique_preds.append(pred_hotel_id)
                seen.add(l_idx)
                if len(unique_preds) == 5:
                    break

        # Calculate AP@5 for this single query
        # Formula: sum(precision@k * rel@k) / min(m, 5)
        # Since there is only 1 relevant item (the true hotel),
        # AP is simply 1/rank if found in top 5, else 0.
        score = 0.0
        num_hits = 0.0

        for k, p in enumerate(unique_preds):
            if p == true_hotel_id:
                num_hits += 1.0
                score += num_hits / (k + 1.0)
                # Since only 1 ground truth, we can break early if found?
                # Standard APK definition sums over all predictions.

        # Normalize by min(len(actual), 5). len(actual) is 1.
        ap_scores.append(score / 1.0)

    final_metric = np.mean(ap_scores)
    print(f"Final Validation Metric: {final_metric}")

    # --- 6. Failure Analysis ---
    print("\n=== Failure Analysis ===")
    # Correlate Error (1 - AP) with Input Features (Training Class Frequency)

    # Get training counts per class
    train_counts = train_df["hotel_id"].value_counts().to_dict()

    # Create analysis dataframe
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    val_df["ap"] = ap_scores
    val_df["error"] = 1.0 - val_df["ap"]
    val_df["train_samples"] = val_df["hotel_id"].map(train_counts).fillna(0)

    # Calculate correlation
    correlation = val_df["error"].corr(val_df["train_samples"])
    print(
        f"Correlation between Error Magnitude (1-AP) and Training Samples (Class Freq): {correlation}"
    )
    print(
        "Interpretation: A negative correlation indicates that classes with more training samples tend to have lower error."
    )

    # --- 7. Submission ---
    threshold = 0.596182394593466
    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({threshold}). Generating Submission..."
        )

        # Use the inference module's predict function
        # We force reload=False to ensure we use the model we just trained/loaded
        # Note: predict() handles gallery generation internally, but we can pass cached paths if we wanted.
        # Here we just let it run. It will use the model passed to it.

        # Ensure test loader is ready
        predict(
            model,
            test_loader,
            device=Config.DEVICE,
            load_cached_data=False,  # Recompute to be safe with new model
            debug=False,
        )
    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({threshold}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
