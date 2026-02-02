import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.models import HierarchicalEfficientNet, HierarchicalSwin
from library.dataset import get_test_loader


def predict_with_tta(model, dataloader, device):
    """
    Generates predictions using Test-Time Augmentation (Original, Horizontal Flip, Vertical Flip).

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): Test dataloader.
        device (torch.device): Computation device.

    Returns:
        tuple: (predictions numpy array, image_ids list)
    """
    model.eval()
    all_preds = []
    all_ids = []

    # TTA Strategy: Original + Horizontal Flip + Vertical Flip
    # Transpose is explicitly excluded as per Config/Strategy.

    with torch.no_grad():
        for images, image_ids in dataloader:
            images = images.to(device)

            # 1. Original Prediction
            out_orig = model(images)
            prob_orig = torch.softmax(out_orig, dim=1)

            # 2. Horizontal Flip Prediction
            # Tensor shape: [B, C, H, W], flip on W (dim 3)
            images_h = torch.flip(images, dims=[3])
            out_h = model(images_h)
            prob_h = torch.softmax(out_h, dim=1)

            # 3. Vertical Flip Prediction
            # Tensor shape: [B, C, H, W], flip on H (dim 2)
            images_v = torch.flip(images, dims=[2])
            out_v = model(images_v)
            prob_v = torch.softmax(out_v, dim=1)

            # Average probabilities
            avg_prob = (prob_orig + prob_h + prob_v) / 3.0

            all_preds.append(avg_prob.cpu().numpy())
            all_ids.extend(image_ids)

    return np.concatenate(all_preds), all_ids


def generate_submission():
    """
    Loads all trained models (EfficientNet and Swin across 5 folds),
    generates predictions with TTA, averages them, and saves the submission file.
    """
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    output_path = os.path.join(submission_dir, "submission.csv")

    # Initialize accumulator for ensemble predictions
    # We will use a dictionary to map image_id to accumulated probabilities
    ensemble_preds = {}
    total_models = 0

    # Define model configurations
    model_configs = [
        {
            "type": "effnet",
            "class": HierarchicalEfficientNet,
            "img_size": Config.IMG_SIZE_EFFNET,
        },
        {"type": "swin", "class": HierarchicalSwin, "img_size": Config.IMG_SIZE_SWIN},
    ]

    # Iterate over model types
    for config in model_configs:
        model_type = config["type"]
        ModelClass = config["class"]
        img_size = config["img_size"]

        # Get Test Loader for this resolution
        # We assume batch size can be handled by the GPU
        test_loader = get_test_loader(image_size=img_size, batch_size=Config.BATCH_SIZE)

        # Iterate over folds
        for fold in range(Config.N_FOLDS):
            weight_file = f"{model_type}_fold_{fold}_best.pth"
            weight_path = os.path.join(Config.WORK_DIR, weight_file)

            if not os.path.exists(weight_path):
                print(f"Weights not found: {weight_path}. Skipping.")
                continue

            # Load Model
            # Set pretrained=False to avoid downloading weights, we load from disk immediately
            model = ModelClass(pretrained=False)
            model.load_state_dict(torch.load(weight_path, map_location=Config.DEVICE))
            model.to(Config.DEVICE)

            print(f"Predicting with {model_type} (Fold {fold})...")
            preds, ids = predict_with_tta(model, test_loader, Config.DEVICE)

            # Accumulate
            for i, img_id in enumerate(ids):
                if img_id not in ensemble_preds:
                    ensemble_preds[img_id] = np.zeros(
                        Config.NUM_CLASSES, dtype=np.float32
                    )
                ensemble_preds[img_id] += preds[i]

            total_models += 1

            # Clean up
            del model
            torch.cuda.empty_cache()

    if total_models == 0:
        print("No trained models found. Cannot generate submission.")
        return

    # Prepare final dataframe
    final_data = []
    for img_id, sum_probs in ensemble_preds.items():
        avg_probs = sum_probs / total_models
        row = {"image_id": img_id}
        for idx, cls_name in enumerate(Config.CLASSES):
            row[cls_name] = avg_probs[idx]
        final_data.append(row)

    df_submission = pd.DataFrame(final_data)

    # Ensure column order matches requirements
    cols = ["image_id"] + Config.CLASSES
    df_submission = df_submission[cols]

    # Sort by image_id for consistency
    df_submission = df_submission.sort_values("image_id")

    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
