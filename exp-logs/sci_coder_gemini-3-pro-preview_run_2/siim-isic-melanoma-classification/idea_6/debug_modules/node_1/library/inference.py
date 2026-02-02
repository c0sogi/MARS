import os
import torch
import pandas as pd
import numpy as np
from library.utils import seed_everything
from library.data_loader import get_dataloaders
from library.model import HierarchicalEfficientNet


def generate_submission(
    model_path="./working/idea_6/model_best.pth",
    output_path="./submission/submission.csv",
    batch_size=64,
    device="cuda" if torch.cuda.is_available() else "cpu",
    load_cached_data=True,
):
    """
    Generates submission file using Test Time Augmentation (TTA).

    Args:
        model_path (str): Path to the trained model weights (Model Soup).
        output_path (str): Path where the submission CSV will be saved.
        batch_size (int): Batch size for inference.
        device (str): Device to run inference on.
        load_cached_data (bool): Whether to load cached metadata features.
    """
    seed_everything(42)

    print(f"Inference device: {device}")

    # 1. Prepare Data
    # We rely on get_dataloaders to handle metadata processing and caching.
    # We only need the test_loader and num_diag_classes.
    print("Loading test data...")
    _, _, test_loader, num_diag_classes = get_dataloaders(
        batch_size=batch_size,
        image_size=384,
        num_workers=4,
        load_cached_data=load_cached_data,
    )

    # 2. Initialize Model
    # Determine metadata dimensions from a sample batch to correctly initialize the fusion layer
    try:
        sample_batch = next(iter(test_loader))
    except StopIteration:
        print("Error: Test loader is empty.")
        return

    n_meta_features = sample_batch["meta"].shape[1]

    print(
        f"Initializing model (Meta features: {n_meta_features}, Diag classes: {num_diag_classes})..."
    )
    model = HierarchicalEfficientNet(
        model_name="efficientnet_b3",
        pretrained=False,  # Avoid downloading weights; we load custom weights from disk
        n_meta_features=n_meta_features,
        n_diagnosis_classes=num_diag_classes,
        num_classes=1,
    )

    # 3. Load Weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Inference with TTA
    print("Starting inference with TTA (4 views)...")
    image_names = []
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            meta = batch["meta"].to(device)
            names = batch["image_name"]

            # TTA Strategy: Average probabilities of 4 views
            # View 1: Original
            logits_1, _ = model(images, meta)
            probs_1 = torch.sigmoid(logits_1)

            # View 2: Horizontal Flip (width dim is 3)
            images_h = torch.flip(images, dims=[3])
            logits_2, _ = model(images_h, meta)
            probs_2 = torch.sigmoid(logits_2)

            # View 3: Vertical Flip (height dim is 2)
            images_v = torch.flip(images, dims=[2])
            logits_3, _ = model(images_v, meta)
            probs_3 = torch.sigmoid(logits_3)

            # View 4: Horizontal + Vertical Flip
            images_hv = torch.flip(images, dims=[2, 3])
            logits_4, _ = model(images_hv, meta)
            probs_4 = torch.sigmoid(logits_4)

            # Average Predictions
            avg_probs = (probs_1 + probs_2 + probs_3 + probs_4) / 4.0

            # Store results
            image_names.extend(names)
            predictions.extend(avg_probs.cpu().numpy().flatten())

    # 5. Create Submission DataFrame
    df_submission = pd.DataFrame({"image_name": image_names, "target": predictions})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} with {len(df_submission)} rows.")
