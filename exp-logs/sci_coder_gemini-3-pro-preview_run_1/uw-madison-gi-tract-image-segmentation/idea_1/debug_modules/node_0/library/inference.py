import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.utils import set_seed, rle_encode
from library.dataset import UWMadisonDataset
from library.model import UNetResNet18


def run_inference(
    batch_size=32,
    img_size=256,
    checkpoint_path="./working/idea_1/best_model.pth",
    submission_dir="./submission",
):
    """
    Executes the inference pipeline: loads model, predicts on test set,
    resizes masks to original dimensions, RLE encodes, and saves submission.

    Args:
        batch_size (int): Batch size for inference.
        img_size (int): Input image size expected by the model.
        checkpoint_path (str): Path to the trained model weights.
        submission_dir (str): Directory to save the submission file.
    """
    # Ensure reproducibility
    set_seed(42)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference Device: {device}")

    # Ensure submission directory exists
    os.makedirs(submission_dir, exist_ok=True)

    # Initialize Model
    model = UNetResNet18(num_classes=3).to(device)

    if os.path.exists(checkpoint_path):
        # Load weights
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model checkpoint from {checkpoint_path}")
    else:
        print(
            f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights."
        )

    model.eval()

    # Initialize Test Dataset and Loader
    # mode='test' ensures the dataset returns (image, id, original_shape)
    test_dataset = UWMadisonDataset(mode="test", img_size=img_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")

    results = []
    classes = ["large_bowel", "small_bowel", "stomach"]

    print("Starting prediction loop...")

    with torch.no_grad():
        for images, ids, original_shapes in test_loader:
            images = images.to(device)

            # Forward pass
            # Output shape: (Batch, 3, img_size, img_size)
            outputs = model(images)

            # Convert to numpy for post-processing
            # We keep probabilities for resizing, then threshold later
            preds_batch = outputs.cpu().numpy()

            # original_shapes is a list of tensors or a tensor batch from the dataloader
            # Depending on collate_fn, usually it's a stack.
            # dataset returns np.array([h, w]), so loader returns tensor of shape (B, 2)
            original_shapes = original_shapes.numpy()

            # Iterate through the batch
            for i in range(len(ids)):
                case_id = ids[i]
                orig_h, orig_w = original_shapes[i]

                # Current prediction: (3, 256, 256)
                pred_mask = preds_batch[i]

                for class_idx, class_name in enumerate(classes):
                    # Extract single class mask: (256, 256)
                    single_mask = pred_mask[class_idx]

                    # Resize to original dimensions (W, H)
                    # cv2.resize expects (width, height)
                    if (single_mask.shape[0] != orig_h) or (
                        single_mask.shape[1] != orig_w
                    ):
                        resized_mask = cv2.resize(
                            single_mask,
                            (orig_w, orig_h),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    else:
                        resized_mask = single_mask

                    # Thresholding
                    binary_mask = (resized_mask > 0.5).astype(np.uint8)

                    # RLE Encoding
                    rle = rle_encode(binary_mask)

                    # Append result
                    results.append(
                        {"id": case_id, "class": class_name, "predicted": rle}
                    )

    # Create Submission DataFrame
    submission_df = pd.DataFrame(results)

    # Ensure columns are in correct order
    submission_df = submission_df[["id", "class", "predicted"]]

    # Save submission
    save_path = os.path.join(submission_dir, "submission.csv")
    submission_df.to_csv(save_path, index=False)

    print(f"Inference complete. Submission saved to {save_path}")
