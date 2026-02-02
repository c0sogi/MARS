import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Add current directory to path to ensure imports work
sys.path.append(os.getcwd())

# Import provided libraries
from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.train import run_training
from library.inference import (
    get_gallery_features,
    extract_features,
    predict,
    run_inference,
)
from library.dataset import HotelDataset, get_transforms, get_class_to_idx
from library.model import HotelRecognitionModel


def calculate_map5(predictions, ground_truth):
    """
    Calculates Mean Average Precision @ 5.
    predictions: List of space-delimited strings of hotel IDs.
    ground_truth: List or array of actual hotel IDs.
    """
    scores = []
    for pred_str, gt in zip(predictions, ground_truth):
        pred_list = pred_str.split()
        # Convert to string for comparison
        gt = str(gt)

        if gt in pred_list:
            rank = pred_list.index(gt) + 1
            scores.append(1.0 / rank)
        else:
            scores.append(0.0)

    return np.mean(scores), np.array(scores)


def main():
    # 1. Setup
    seed_everything(Config.seed)

    # Override Config for fast baseline
    # 4 epochs should be sufficient for a baseline on A100 (approx 15-20 mins total)
    Config.epochs = 4

    print("=== Starting Runfile Execution ===")

    # 2. Training
    print(f"Starting training for {Config.epochs} epochs...")
    run_training(epochs=Config.epochs)

    # Clear memory
    torch.cuda.empty_cache()

    # 3. Validation Inference
    print("=== Starting Validation ===")

    # Load Metadata
    val_df = pd.read_csv(Config.val_metadata_path)
    train_df = pd.read_csv(
        Config.train_metadata_path
    )  # Needed for class frequency analysis

    # Initialize Model
    device = Config.device
    model = HotelRecognitionModel(
        n_classes=Config.num_classes,
        backbone_name=Config.backbone_name,
        pretrained=False,
        embedding_size=Config.embedding_size,
    )
    model.to(device)

    # Load Best Checkpoint
    # library.utils.save_checkpoint saves 'best_model.pth' in the same dir
    best_model_path = os.path.join(
        os.path.dirname(Config.model_save_path), "best_model.pth"
    )
    if not os.path.exists(best_model_path):
        best_model_path = Config.model_save_path

    print(f"Loading model from {best_model_path}")
    checkpoint = load_checkpoint(model, best_model_path, device)

    # Generate Gallery (Train) Embeddings
    # This uses caching logic inside the function
    gallery_embs, gallery_ids = get_gallery_features(
        model, device, load_cached_data=True
    )

    # Generate Query (Validation) Embeddings
    print("Generating validation embeddings...")
    # We need class_to_idx to initialize dataset, though we won't use the labels for extraction
    class_to_idx = get_class_to_idx(train_df)

    val_dataset = HotelDataset(
        df=val_df,
        transform=get_transforms(mode="val"),
        data_root=Config.input_dir,
        mode="val",
        class_to_idx=class_to_idx,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    val_embs, val_labels = extract_features(val_loader, model, device)

    # Run Retrieval
    print("Running retrieval on validation set...")
    val_preds = predict(
        query_embeddings=val_embs,
        gallery_embeddings=gallery_embs,
        gallery_ids=gallery_ids,
        knn=Config.knn,
        top_k=Config.top_k,
        device=device,
    )

    # 4. Metric Calculation
    print("Calculating MAP@5...")
    # val_labels from extract_features are the mapped indices, but predict returns hotel_ids.
    # We need the original hotel_ids from val_df.
    val_gt_ids = val_df["hotel_id"].values

    map5, ap_scores = calculate_map5(val_preds, val_gt_ids)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {map5}")

    # 5. Failure Analysis
    print("=== Failure Analysis ===")
    # Calculate Class Frequency in Training
    train_counts = train_df["hotel_id"].value_counts().to_dict()

    # Map validation samples to their training class frequency
    # Use a default of 0 if not found (though split ensures presence)
    val_class_freqs = [train_counts.get(hid, 0) for hid in val_gt_ids]

    # Error Magnitude = 1 - AP
    error_magnitudes = 1.0 - ap_scores

    # Calculate Correlation
    # We use Log(Frequency) because frequency follows a power law
    log_freqs = np.log1p(val_class_freqs)

    correlation, _ = pearsonr(error_magnitudes, log_freqs)

    print(
        f"Correlation between Error Magnitude (1-AP) and Log(Class Frequency): {correlation}"
    )
    print(
        "Interpretation: Negative correlation implies higher frequency classes have lower error."
    )

    # 6. Submission
    THRESHOLD = 0.004247584426088712

    if map5 > THRESHOLD:
        print(f"Validation metric {map5} > {THRESHOLD}. Generating submission...")
        # Clear memory before inference
        del gallery_embs, val_embs, val_loader, val_dataset
        torch.cuda.empty_cache()

        # Run inference pipeline (Test set)
        # Note: run_inference reloads the model from Config.model_save_path.
        # We should ensure Config.model_save_path points to the best model or overwrite it.
        # Since run_inference loads Config.model_save_path, let's ensure it's correct.
        if best_model_path != Config.model_save_path:
            import shutil

            shutil.copy(best_model_path, Config.model_save_path)

        run_inference(load_cached_data=True)
        print("Submission generated successfully.")
    else:
        print(f"Validation metric {map5} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
