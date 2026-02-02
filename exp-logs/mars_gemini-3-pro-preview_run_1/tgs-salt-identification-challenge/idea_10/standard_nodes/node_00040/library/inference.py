import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from library.utils import set_seed, rle_encode
from library.model import DeepResUNet
from library.dataset import get_test_loader


def predict_and_submit(
    checkpoint_dir="./working/checkpoints",
    output_dir="./submission",
    cache_dir="./working/idea_10",
    batch_size=32,
    num_workers=4,
    device=None,
):
    """
    Performs inference using Snapshot Ensembling and Classification Gating.
    Generates a submission file in RLE format.
    """
    # 1. Setup
    set_seed(42)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)

    # 2. Load Data
    # get_test_loader handles caching internally
    test_loader, test_ids = get_test_loader(
        test_metadata_path="./metadata/test.csv",
        cache_dir=cache_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        load_cached_data=True,
    )

    # 3. Load Models (Snapshot Ensemble)
    # Priority: best_cycle_2.pth + best_cycle_3.pth
    # Fallback: best_model.pth

    cycle_2_path = os.path.join(checkpoint_dir, "best_cycle_2.pth")
    cycle_3_path = os.path.join(checkpoint_dir, "best_cycle_3.pth")
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    model_paths = []
    if os.path.exists(cycle_2_path) and os.path.exists(cycle_3_path):
        print(f"Loading Snapshot Ensemble: {cycle_2_path} and {cycle_3_path}")
        model_paths = [cycle_2_path, cycle_3_path]
    elif os.path.exists(best_model_path):
        print(
            f"Snapshot checkpoints missing. Loading single best model: {best_model_path}"
        )
        model_paths = [best_model_path]
    else:
        print(
            "No checkpoints found. Initializing random model (for debugging/testing only)."
        )
        # This path usually shouldn't be reached in a valid evaluation run
        model = DeepResUNet(in_channels=2, classes=1).to(device)
        models = [model]

    models = []
    for p in model_paths:
        m = DeepResUNet(in_channels=2, classes=1).to(device)
        # Load weights
        state_dict = torch.load(p, map_location=device)
        m.load_state_dict(state_dict)
        m.eval()
        models.append(m)

    print(f"Inference initialized with {len(models)} model(s) on {device}.")

    # 4. Inference Loop
    rle_results = []
    processed_ids = []

    # Padding info for unpadding: 101 -> 128
    # Pad was: Top=13, Bottom=14, Left=13, Right=14
    # Valid region indices: [13 : 114]

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(device)

            # Test-Time Augmentation: Horizontal Flip
            images_flip = torch.flip(images, dims=[3])

            batch_seg_probs = []

            for model in models:
                # Forward pass - Original
                out = model(images)
                # Forward pass - Flipped
                out_flip = model(images_flip)

                # Extract Logits
                logits = out["logits"]
                logits_flip = out_flip["logits"]

                # Sigmoid
                probs = torch.sigmoid(logits)
                probs_flip = torch.sigmoid(logits_flip)

                # Revert Flip for spatial maps
                probs_flip = torch.flip(probs_flip, dims=[3])

                # TTA Averaging for this model
                avg_prob = (probs + probs_flip) / 2.0

                batch_seg_probs.append(avg_prob)

            # Ensemble Averaging
            # Stack: (Num_Models, B, 1, H, W) -> Mean -> (B, 1, H, W)
            final_prob = torch.stack(batch_seg_probs).mean(dim=0)

            # Unpad to 101x101
            # Crop [13:114, 13:114]
            final_prob = final_prob[:, 0, 13:114, 13:114]

            # Binarize
            pred_masks = (final_prob > 0.5).byte().cpu().numpy()

            # Encode
            # Map batch indices to global IDs
            start_idx = i * batch_size
            current_batch_size = pred_masks.shape[0]

            for j in range(current_batch_size):
                mask = pred_masks[j]
                rle = rle_encode(mask)
                rle_results.append(rle)
                processed_ids.append(test_ids[start_idx + j])

    # 5. Save Submission
    submission_df = pd.DataFrame({"id": processed_ids, "rle_mask": rle_results})

    save_path = os.path.join(output_dir, "submission.csv")
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")
