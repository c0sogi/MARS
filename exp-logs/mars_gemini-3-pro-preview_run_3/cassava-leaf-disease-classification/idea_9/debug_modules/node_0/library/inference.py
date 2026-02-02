import os
import torch
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything
from library.data import CassavaDataset, get_transforms
from library.model import CassavaClassifier


def inference_fn(model, loader, device):
    """
    Performs inference with Test-Time Augmentation (TTA).

    TTA Strategy:
    1. Original Image
    2. Horizontal Flip
    3. Vertical Flip
    4. Transpose (Swap Height and Width)

    Softmax probabilities from all views are averaged.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # 1. Original
            out1 = model(images)
            prob1 = F.softmax(out1, dim=1)

            # 2. Horizontal Flip
            # Image tensor shape: [B, C, H, W] -> Flip on W (dim 3)
            out2 = model(torch.flip(images, dims=[3]))
            prob2 = F.softmax(out2, dim=1)

            # 3. Vertical Flip
            # Flip on H (dim 2)
            out3 = model(torch.flip(images, dims=[2]))
            prob3 = F.softmax(out3, dim=1)

            # 4. Transpose
            # Swap H and W (dims 2 and 3)
            out4 = model(torch.transpose(images, 2, 3))
            prob4 = F.softmax(out4, dim=1)

            # Average probabilities across all TTA views
            avg_prob = (prob1 + prob2 + prob3 + prob4) / 4.0

            # Move to CPU to save GPU memory during accumulation
            preds.append(avg_prob.cpu())

    return torch.cat(preds)


def generate_submission(
    model_a_path=os.path.join(Config.working_dir, "model_a_best.pth"),
    model_b_path=os.path.join(Config.working_dir, "model_b_best.pth"),
    output_path=Config.submission_path,
):
    """
    Generates the submission file using the ensemble of Model A and Model B with TTA.
    """
    seed_everything(Config.seed)
    device = torch.device(Config.device)

    print("Initializing Submission Generation...")

    # 1. Prepare Data
    if not os.path.exists(Config.test_metadata):
        raise FileNotFoundError(f"Test metadata not found at {Config.test_metadata}")

    test_df = pd.read_csv(Config.test_metadata)

    # Initialize Dataset and Loader
    # Note: We use 'test' transforms which only apply Resize and Normalize.
    # Geometric augmentations are handled manually in inference_fn for TTA.
    test_dataset = CassavaDataset(
        metadata_path=Config.test_metadata,
        transform=get_transforms("test", img_size=Config.img_size),
        is_train=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    print(f"Test set size: {len(test_dataset)} images")

    # 2. Model A Inference (ViT)
    print(f"Processing Model A: {Config.model_a_name}")
    if not os.path.exists(model_a_path):
        raise FileNotFoundError(f"Model A weights not found at {model_a_path}")

    # Initialize model structure (pretrained=False as we load custom weights)
    model_a = CassavaClassifier(Config.model_a_name, pretrained=False)
    model_a.load_state_dict(torch.load(model_a_path, map_location=device))
    model_a.to(device)

    # Run Inference
    preds_a = inference_fn(model_a, test_loader, device)

    # Explicitly clean up to free memory for Model B
    del model_a
    torch.cuda.empty_cache()

    # 3. Model B Inference (BEiT)
    print(f"Processing Model B: {Config.model_b_name}")
    if not os.path.exists(model_b_path):
        raise FileNotFoundError(f"Model B weights not found at {model_b_path}")

    model_b = CassavaClassifier(Config.model_b_name, pretrained=False)
    model_b.load_state_dict(torch.load(model_b_path, map_location=device))
    model_b.to(device)

    # Run Inference
    preds_b = inference_fn(model_b, test_loader, device)

    # Clean up
    del model_b
    torch.cuda.empty_cache()

    # 4. Ensemble Averaging
    print("Ensembling predictions...")
    # Average the probabilities from both models
    final_probs = (preds_a + preds_b) / 2.0

    # Get final class labels
    final_labels = torch.argmax(final_probs, dim=1).numpy()

    # 5. Create and Save Submission
    submission = pd.DataFrame({"image_id": test_df["image_id"], "label": final_labels})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission.to_csv(output_path, index=False)
    print(f"Submission saved successfully to {output_path}")
    print("First 5 predictions:")
    print(submission.head())
