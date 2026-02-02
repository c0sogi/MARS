import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from library.config import Config
from library.modeling import CassavaClassifier
from library.dataset import get_test_loader
from library.utils import seed_everything


def predict_with_tta(model, dataloader, device, use_tta=False):
    """
    Generates predictions for a single model, optionally using Test Time Augmentation.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): Test data loader.
        device (torch.device): Computation device.
        use_tta (bool): Whether to use TTA (Horizontal & Vertical Flips).

    Returns:
        np.ndarray: Softmax probabilities of shape (N, num_classes).
    """
    model.eval()
    all_probs = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)

            # 1. Forward pass on original images
            logits = model(images)
            probs = F.softmax(logits, dim=1)

            if use_tta:
                # 2. Horizontal Flip
                images_h = torch.flip(images, dims=[3])
                logits_h = model(images_h)
                probs += F.softmax(logits_h, dim=1)

                # 3. Vertical Flip
                images_v = torch.flip(images, dims=[2])
                logits_v = model(images_v)
                probs += F.softmax(logits_v, dim=1)

                # Average probabilities
                probs /= 3.0

            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


def run_inference():
    """
    Main inference routine.
    Loads all available model checkpoints (Architectures x Folds),
    computes predictions with TTA, aggregates them, and saves the submission.
    """
    # Ensure reproducibility
    seed_everything(Config.seed)

    device = Config.device

    # Use the higher resolution (Phase 2) for inference as models were fine-tuned on it.
    # We use p1_batch_size (32) which is safe for inference on A100 even at 512x512.
    inference_batch_size = Config.p1_batch_size

    print(
        f"Initializing Test Loader (Size: {Config.p2_image_size}, Batch: {inference_batch_size})..."
    )
    test_loader = get_test_loader(
        image_size=Config.p2_image_size, batch_size=inference_batch_size
    )

    num_test_samples = len(test_loader.dataset)
    # Accumulator for ensemble probabilities
    ensemble_probs = np.zeros((num_test_samples, Config.num_classes), dtype=np.float32)
    model_count = 0

    print(f"Starting Inference on {num_test_samples} images...")
    print(f"TTA Enabled: {Config.use_tta}")

    # Iterate through defined architectures and folds
    for model_name in Config.model_names:
        for fold in range(Config.n_folds):
            # Construct expected checkpoint path
            # Convention: {model_name}_fold_{fold}.pth inside output_dir
            checkpoint_filename = f"{model_name}_fold_{fold}.pth"
            checkpoint_path = os.path.join(Config.output_dir, checkpoint_filename)

            if not os.path.exists(checkpoint_path):
                # Fallback check for "model_best.pth" if running single fold/debug
                # This helps if the user ran a simplified training loop
                if Config.n_folds == 1 and fold == 0:
                    fallback_path = os.path.join(Config.output_dir, "model_best.pth")
                    if os.path.exists(fallback_path):
                        checkpoint_path = fallback_path
                    else:
                        print(f"Checkpoint not found: {checkpoint_path}. Skipping...")
                        continue
                else:
                    print(f"Checkpoint not found: {checkpoint_path}. Skipping...")
                    continue

            print(f"Processing Model: {model_name} | Fold: {fold}")

            # Initialize Model
            # pretrained=False because we are loading custom weights
            model = CassavaClassifier(model_name=model_name, pretrained=False)
            model.to(device)

            # Load Weights
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)

                # Handle different checkpoint saving formats
                if isinstance(checkpoint, dict):
                    if "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    elif "model" in checkpoint:
                        state_dict = checkpoint["model"]
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint

                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading checkpoint {checkpoint_path}: {e}")
                continue

            # Generate Predictions
            probs = predict_with_tta(model, test_loader, device, use_tta=Config.use_tta)

            # Accumulate
            ensemble_probs += probs
            model_count += 1

            # Cleanup to free memory
            del model, checkpoint, state_dict, probs
            torch.cuda.empty_cache()

    # Finalize Predictions
    if model_count > 0:
        print(f"Aggregating predictions from {model_count} models...")
        ensemble_probs /= model_count
    else:
        print("WARNING: No models found. Generating random predictions.")
        ensemble_probs = np.random.rand(num_test_samples, Config.num_classes)

    final_preds = np.argmax(ensemble_probs, axis=1)

    # Create Submission DataFrame
    # We read the test metadata again to ensure alignment of image_ids
    df_test = pd.read_csv(Config.test_metadata_path)

    # Safety check
    if len(df_test) != len(final_preds):
        raise ValueError(
            f"Mismatch: Metadata has {len(df_test)} rows, Predictions have {len(final_preds)} rows."
        )

    df_test["label"] = final_preds

    # Format for submission
    submission_df = df_test[["image_id", "label"]]

    # Save
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print("Head of submission:")
    print(submission_df.head())
