import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything
from library.model import EfficientNetArcFace
from library.dataset import get_dataloaders


def inference_fn(dataloader, model, device, label_map):
    """
    Generates predictions for the test set using the trained model.
    Applies Test-Time Augmentation (TTA) and saves the results to a CSV file.

    Args:
        dataloader (DataLoader): The test set dataloader.
        model (nn.Module): The trained EfficientNetArcFace model.
        device (torch.device): The computation device (CPU or GPU).
        label_map (dict): Dictionary mapping hotel_id -> label_idx.
    """
    model.eval()
    test_preds = []

    # Get normalized class centers from the ArcFace head
    # shape: (num_classes, embedding_size)
    # We detach to ensure no gradients are tracked
    centers = F.normalize(model.head.weight, p=2, dim=1).detach()

    print("Starting Inference on Test Set...")

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)

            # 1. Forward Pass (Original)
            # When labels are None, the model returns embeddings
            embeddings = model(images)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # 2. TTA: Horizontal Flip
            if Config.TTA:
                # Flip images horizontally (dim 3 is width)
                images_flip = torch.flip(images, [3])
                embeddings_flip = model(images_flip)
                embeddings_flip = F.normalize(embeddings_flip, p=2, dim=1)

                # Average embeddings
                embeddings = (embeddings + embeddings_flip) / 2.0
                # Re-normalize after averaging to ensure unit length
                embeddings = F.normalize(embeddings, p=2, dim=1)

            # 3. Compute Cosine Similarity
            # (Batch_Size, Emb_Size) @ (Num_Classes, Emb_Size)^T -> (Batch_Size, Num_Classes)
            logits = torch.matmul(embeddings, centers.T)

            # 4. Get Top 5 Predictions
            _, top_indices = logits.topk(Config.TOP_K, dim=1)

            test_preds.extend(top_indices.cpu().numpy())

    # Decode predictions using label_map
    # label_map maps hotel_id -> label_idx
    # We need label_idx -> hotel_id
    idx_to_hotel = {v: k for k, v in label_map.items()}

    final_submission = []

    # Get list of image names from the dataset dataframe
    # Accessing the underlying dataset from the dataloader
    image_names = dataloader.dataset.df["image"].values

    for img_name, pred_indices in zip(image_names, test_preds):
        # Map indices to hotel_ids
        # Ensure hotel_ids are strings as per submission format
        pred_hotel_ids = [str(idx_to_hotel[idx]) for idx in pred_indices]
        prediction_string = " ".join(pred_hotel_ids)
        final_submission.append({"image": img_name, "hotel_id": prediction_string})

    # Create DataFrame
    submission_df = pd.DataFrame(final_submission)

    # Save to CSV
    submission_path = Config.SUBMISSION_FILE
    # Ensure output directory exists
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(submission_df.head())


def run_inference(debug=Config.DEBUG):
    """
    Main entry point for running inference.
    Loads the model, data, and label encoder, then triggers prediction.

    Args:
        debug (bool): Whether to run in debug mode (smaller subset).
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Label Encoder
    # The encoder should have been created during the training phase setup.
    # We use it to determine num_classes and to map predictions back to hotel IDs.
    encoder_path = os.path.join(Config.WORKING_DIR, "label_encoder.parquet")

    if not os.path.exists(encoder_path):
        # If running inference standalone without training first, we might need to regenerate it
        # However, typically inference follows training. We'll assume it exists or regenerate via get_dataloaders logic.
        print(
            "Label encoder not found in cache. Initializing dataloaders to generate it..."
        )
        _, _, _, num_classes = get_dataloaders(load_cached_data=False)
        # Reload to get the map
        label_df = pd.read_parquet(encoder_path)
    else:
        label_df = pd.read_parquet(encoder_path)
        num_classes = len(label_df)

    label_map = dict(zip(label_df["hotel_id"], label_df["label_idx"]))

    # 2. Initialize Model
    print(f"Initializing Model: {Config.MODEL_NAME} with {num_classes} classes...")
    model = EfficientNetArcFace(n_classes=num_classes).to(device)

    # 3. Load Pretrained Weights
    if os.path.exists(Config.MODEL_PATH):
        print(f"Loading weights from {Config.MODEL_PATH}...")
        state_dict = torch.load(Config.MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model checkpoint not found at {Config.MODEL_PATH}. Using random weights (Expected only in testing pipeline without training)."
        )

    # 4. Get Test DataLoader
    # We don't need train/val loaders here, but get_dataloaders returns them.
    print("Loading Test Data...")
    _, _, test_loader, _ = get_dataloaders(load_cached_data=True, debug=debug)

    # 5. Run Inference
    inference_fn(test_loader, model, device, label_map)
