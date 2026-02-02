import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_loaders
from library.model import WhaleModel
from library.engine import generate_submission


def inference_pipeline(load_cached_data=True, debug_limit=None):
    """
    Executes the inference pipeline for Whale Identification.

    Implements the Manifold-Aware Split-Pipeline:
    1. Loads Test (Query) and Gallery (Known Training) data.
    2. Loads the trained EfficientNet-B4 ArcFace model.
    3. Delegates to library.engine.generate_submission which performs:
       - Feature Extraction
       - Global Cosine Retrieval (for Open-Set Rejection)
       - k-Reciprocal Re-ranking (for Candidate Sorting)
       - Submission CSV generation

    Args:
        load_cached_data (bool): If True, uses cached numpy arrays for images.
        debug_limit (int, optional): If set, limits the number of test/gallery samples
                                     for rapid debugging.
    """
    # 1. Setup
    seed_everything(Config.seed)
    device = Config.device
    print(f"Inference Device: {device}")

    # 2. Data Loading
    # get_loaders returns: train_loader, val_loader, gallery_loader, test_loader, num_classes, label_encoder
    # We only need gallery (known whales) and test (queries) for inference.
    print("Loading data...")
    _, _, gallery_loader, test_loader, num_classes, label_encoder = get_loaders(
        load_cached_data=load_cached_data
    )

    # 3. Debugging Logic (Limit Dataset Size)
    original_test_csv_path = Config.test_csv_path

    if debug_limit is not None:
        print(f"DEBUG MODE: Limiting Test and Gallery to {debug_limit} samples.")

        # A. Limit Gallery Loader
        gallery_dataset = gallery_loader.dataset
        if len(gallery_dataset) > debug_limit:
            # Slice images and labels
            gallery_dataset.images = gallery_dataset.images[:debug_limit]
            gallery_dataset.labels = gallery_dataset.labels[:debug_limit]

            # Re-create DataLoader
            gallery_loader = DataLoader(
                gallery_dataset,
                batch_size=gallery_loader.batch_size,
                shuffle=False,
                num_workers=gallery_loader.num_workers,
                pin_memory=True,
            )

        # B. Limit Test Loader
        test_dataset = test_loader.dataset
        if len(test_dataset) > debug_limit:
            # Slice images (labels are None for test)
            test_dataset.images = test_dataset.images[:debug_limit]

            # Re-create DataLoader
            test_loader = DataLoader(
                test_dataset,
                batch_size=test_loader.batch_size,
                shuffle=False,
                num_workers=test_loader.num_workers,
                pin_memory=True,
            )

            # C. Handle Metadata Mismatch
            # generate_submission reads Config.test_csv_path to get filenames.
            # We must create a temporary CSV matching the subset to avoid IndexError.
            test_df = pd.read_csv(Config.test_csv_path)
            subset_test_df = test_df.iloc[:debug_limit]

            temp_test_csv = os.path.join(Config.working_dir, "test_debug.csv")
            subset_test_df.to_csv(temp_test_csv, index=False)

            # Monkey-patch Config to point to the temporary file
            Config.test_csv_path = temp_test_csv
            print(f"DEBUG: Temporarily patched Config.test_csv_path to {temp_test_csv}")

    # 4. Model Initialization
    # Initialize the architecture. We use pretrained=False because we will load our own weights.
    print("Initializing model...")
    model = WhaleModel(embedding_size=Config.embedding_size, pretrained=False)

    # 5. Load Weights
    if os.path.exists(Config.model_save_path):
        print(f"Loading model weights from {Config.model_save_path}...")
        state_dict = torch.load(Config.model_save_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"WARNING: Model checkpoint not found at {Config.model_save_path}.")
        print("Inference will proceed with random weights (expect poor performance).")

    model.to(device)
    model.eval()

    # 6. Run Inference Pipeline
    # This function extracts features, calculates re-ranking, applies the open-set threshold,
    # and saves the results to Config.submission_path.
    try:
        generate_submission(test_loader, gallery_loader, model, device, label_encoder)
    finally:
        # Restore Config if it was modified for debugging
        if debug_limit is not None:
            Config.test_csv_path = original_test_csv_path
            # Optional: Clean up temp file
            temp_csv = os.path.join(Config.working_dir, "test_debug.csv")
            if os.path.exists(temp_csv):
                os.remove(temp_csv)

    print("Inference pipeline completed successfully.")
