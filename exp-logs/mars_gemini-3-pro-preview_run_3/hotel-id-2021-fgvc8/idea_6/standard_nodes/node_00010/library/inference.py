import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from library.config import Config
from library.dataset import get_dataloaders, get_label_encoder
from library.model import HotelRecognitionModel


def predict_and_submit(
    model_path=Config.MODEL_PATH,
    output_file=Config.SUBMISSION_FILE,
    device=Config.DEVICE,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
):
    """
    Generates predictions for the test set using the trained model with Test-Time Augmentation (TTA).

    Args:
        model_path (str): Path to the saved model checkpoint.
        output_file (str): Path where the submission CSV will be saved.
        device (str): Device to run inference on ('cuda' or 'cpu').
        batch_size (int): Batch size for inference.
        num_workers (int): Number of workers for data loading.
    """
    print(f"Starting inference pipeline...")
    print(f"Model Path: {model_path}")
    print(f"Device: {device}")

    # 1. Load Data
    # We use the factory function but only need the test_loader and classes
    # debug=False ensures we predict on the full test set
    _, _, test_loader, classes = get_dataloaders(debug=False, load_cached_data=True)

    # 2. Initialize Model
    # We must initialize the model with the same architecture parameters as training
    model = HotelRecognitionModel(
        n_classes=len(classes),
        model_name=Config.BACKBONE,
        pretrained=False,  # Pretrained weights not needed as we load checkpoint
        embedding_size=Config.EMBEDDING_SIZE,
    )
    model.to(device)
    model.eval()

    # 3. Load Checkpoint
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    else:
        print(f"Error: Model checkpoint not found at {model_path}")
        return

    # 4. Prepare for Inference
    predictions_str = []

    # Pre-compute normalized class centers from the ArcFace head
    # Shape: (NumClasses, EmbeddingSize)
    class_centers = model.get_class_centers().detach()

    print(
        "Running inference with Test-Time Augmentation (Original + Horizontal Flip)..."
    )

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)

            # --- Test-Time Augmentation (TTA) ---
            # 1. Forward pass with original images
            emb_orig = model(images, labels=None)  # (B, EmbedSize)

            # 2. Forward pass with horizontally flipped images
            # Dim 3 is width for (B, C, H, W)
            images_flip = torch.flip(images, dims=[3])
            emb_flip = model(images_flip, labels=None)

            # 3. Average embeddings
            embeddings = (emb_orig + emb_flip) / 2.0

            # 4. Normalize final embeddings
            embeddings = F.normalize(embeddings)

            # --- Similarity & Ranking ---
            # Cosine Similarity: (B, Embed) @ (NumClasses, Embed).T -> (B, NumClasses)
            sims = torch.matmul(embeddings, class_centers.T)

            # Get Top 5 indices
            _, topk_indices = torch.topk(sims, k=5, dim=1)
            topk_indices = topk_indices.cpu().numpy()

            # Map indices back to hotel_ids
            for i in range(len(topk_indices)):
                indices = topk_indices[i]
                # Map index -> hotel_id
                pred_hotel_ids = [str(classes[idx]) for idx in indices]
                predictions_str.append(" ".join(pred_hotel_ids))

    # 5. Generate Submission File
    # The test dataloader preserves order if shuffle=False (default in get_dataloaders)
    test_df = test_loader.dataset.df
    image_ids = test_df["image"].tolist()

    # Ensure lengths match
    if len(image_ids) != len(predictions_str):
        print(
            f"Warning: Mismatch between image IDs ({len(image_ids)}) and predictions ({len(predictions_str)})"
        )

    submission_df = pd.DataFrame({"image": image_ids, "hotel_id": predictions_str})

    # Save to disk
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    submission_df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
    print("Inference completed.")
