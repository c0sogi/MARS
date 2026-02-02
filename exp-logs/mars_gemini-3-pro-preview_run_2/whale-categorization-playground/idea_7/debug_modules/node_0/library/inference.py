import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from library.config import Config
from library.dataset import get_test_loader, get_label_mapping
from library.model import WhaleModel
from library.loss import ArcFaceLoss


def extract_embeddings_with_tta(loader, model, device):
    """
    Extracts embeddings for the given loader using the model.
    Applies Test-Time Augmentation (Horizontal Flip).

    Args:
        loader: DataLoader for the images.
        model: The model to use for extraction.
        device: Torch device.

    Returns:
        torch.Tensor: Normalized embeddings (N, Embedding_Size).
    """
    model.eval()
    embeddings_list = []

    with torch.no_grad():
        for batch_idx, images in enumerate(loader):
            images = images.to(device)

            # 1. Forward pass original
            emb_orig = model(images)

            # 2. Forward pass flipped (TTA)
            # Assuming images are (B, C, H, W)
            images_flip = torch.flip(images, dims=[3])
            emb_flip = model(images_flip)

            # 3. Average
            emb = (emb_orig + emb_flip) / 2.0

            # 4. Normalize
            # We normalize here to ensure the ensemble averaging is on the hypersphere
            emb = F.normalize(emb, p=2, dim=1)

            embeddings_list.append(emb.cpu())

    return torch.cat(embeddings_list, dim=0)


def generate_submission():
    """
    Main inference function.
    Loads models, computes ensemble similarities, generates predictions, and saves submission.
    """
    device = Config.DEVICE
    print(f"Starting Inference on device: {device}")

    # 1. Setup Data
    # We use the final resolution from the stages config
    final_resolution = 384
    if Config.STAGES:
        final_resolution = Config.STAGES[-1]["resolution"]

    print(f"Using resolution: {final_resolution}x{final_resolution}")

    # Get Test Loader
    test_loader = get_test_loader(
        resolution=final_resolution, batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # Get ID mapping
    # id2idx: Label -> Int, idx2id: Int -> Label
    _, idx2id = get_label_mapping()

    # 2. Ensemble Loop
    # We will accumulate similarity matrices
    # Shape: (N_Test, N_Classes)
    ensemble_sim_matrix = None
    models_processed = 0

    for model_cfg in Config.MODEL_CONFIGS:
        model_name = model_cfg["name"]
        backbone_name = model_cfg["backbone"]
        emb_size = model_cfg["embedding_size"]

        print(f"\nProcessing Model: {model_name} ({backbone_name})")

        # Initialize Model
        model = WhaleModel(backbone_name, pretrained=False, embedding_size=emb_size)
        model = model.to(device)

        # Load Checkpoint
        checkpoint_path = os.path.join(
            Config.WORKING_DIR, f"{model_name}_{final_resolution}_best.pth"
        )

        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint {checkpoint_path} not found. Skipping model.")
            continue

        print(f"Loading weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Extract Test Embeddings
        print("Extracting test embeddings with TTA...")
        test_embeddings = extract_embeddings_with_tta(test_loader, model, device)

        # Extract Gallery Centers (from Loss State)
        # The loss state dict contains 'weight' which are the class centers
        loss_state = checkpoint["loss_state_dict"]
        if "weight" in loss_state:
            centers = loss_state["weight"].cpu()  # (N_Classes, Emb_Size)
        else:
            print("Error: Could not find class centers in checkpoint.")
            continue

        # Normalize Centers
        centers = F.normalize(centers, p=2, dim=1)

        # Compute Cosine Similarity for this model
        # Test: (N_Test, Emb), Centers: (N_Classes, Emb) -> (N_Test, N_Classes)
        print("Computing similarity matrix...")
        sim_matrix = torch.matmul(test_embeddings, centers.T)

        if ensemble_sim_matrix is None:
            ensemble_sim_matrix = sim_matrix
        else:
            ensemble_sim_matrix += sim_matrix

        models_processed += 1

        # Cleanup to save memory
        del model, checkpoint, test_embeddings, centers, sim_matrix
        torch.cuda.empty_cache()

    if ensemble_sim_matrix is None or models_processed == 0:
        print("Error: No models were successfully processed.")
        return

    # Average the similarities
    ensemble_sim_matrix /= models_processed

    # 3. Generate Predictions
    print("\nGenerating predictions...")

    # Load Test Metadata to get Image IDs in order
    df_test = pd.read_csv(Config.TEST_CSV)
    test_filenames = df_test["Image"].tolist()

    final_predictions = []

    # Threshold for new_whale
    threshold = Config.CONFIDENCE_THRESHOLD

    # Iterate over each test image
    # sim_matrix is (N_Test, N_Classes)
    for i in range(len(test_filenames)):
        filename = test_filenames[i]
        sims = ensemble_sim_matrix[i]

        # Get top 5 candidates
        scores, indices = torch.topk(sims, k=5)
        scores = scores.numpy()
        indices = indices.numpy()

        preds = []
        new_whale_added = False

        for score, idx in zip(scores, indices):
            # Check if we should insert new_whale before this candidate
            # If the similarity to the best known whale is low, new_whale is likely
            if not new_whale_added and score < threshold:
                preds.append("new_whale")
                new_whale_added = True

            if len(preds) >= 5:
                break

            # Add known whale
            label = idx2id[idx]
            preds.append(label)

        # Fill remaining slots
        if len(preds) < 5 and not new_whale_added:
            preds.append("new_whale")
            new_whale_added = True

        # Ensure exactly 5 predictions
        preds = preds[:5]

        prediction_str = " ".join(preds)
        final_predictions.append({"Image": filename, "Id": prediction_str})

    # 4. Save Submission
    df_submission = pd.DataFrame(final_predictions)
    save_path = Config.SUBMISSION_PATH
    df_submission.to_csv(save_path, index=False)

    print(f"Submission saved to {save_path}")
    print(df_submission.head())
