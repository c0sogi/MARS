import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import get_label_map, get_transforms, collate_fn
from library.dataset import KuzushijiDataset
from library.model import get_model


def predict_and_submit(config=None):
    """
    Performs inference on the test dataset and generates the submission file.

    Args:
        config (Config, optional): Configuration object.
    """
    if config is None:
        config = Config()

    # 1. Setup Environment
    device = config.DEVICE
    print(f"Using device: {device}")

    # 2. Load Label Map (ID -> Unicode)
    # get_label_map returns (unicode_to_id, id_to_unicode)
    _, id_to_unicode = get_label_map(config)

    # 3. Prepare Test Data
    # We use the 'test' split which loads metadata from ./metadata/test.csv
    test_dataset = KuzushijiDataset(
        split="test",
        config=config,
        transforms=get_transforms(train=False),
        load_cached_data=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # 4. Load Model
    # The get_model function uses config to set:
    # - RPN proposals (2000)
    # - Detections per image (1200)
    # - Score threshold (0.35)
    model = get_model(config.NUM_CLASSES, config)

    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {config.MODEL_PATH}. Train the model first."
        )

    # Load weights
    state_dict = torch.load(config.MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 5. Inference Loop
    results = []
    print(f"Starting inference on {len(test_dataset)} images...")

    with torch.no_grad():
        for images, targets in test_loader:
            # Move images to device
            images = [img.to(device) for img in images]

            # Forward pass
            # outputs is a list of dicts: [{'boxes': ..., 'labels': ..., 'scores': ...}, ...]
            # The model internally handles resizing and rescaling boxes back to original image coordinates.
            outputs = model(images)

            # Process batch results
            for i, output in enumerate(outputs):
                image_id = targets[i]["image_id"]

                pred_boxes = output["boxes"].cpu().numpy()
                pred_labels = output["labels"].cpu().numpy()
                # pred_scores = output["scores"].cpu().numpy() # Already filtered by threshold in model

                label_strings = []

                for box, label_id in zip(pred_boxes, pred_labels):
                    # Box format is [x1, y1, x2, y2]
                    x1, y1, x2, y2 = box

                    # Calculate center point
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # Get Unicode character
                    char = id_to_unicode.get(label_id)
                    if char is None:
                        continue

                    # Append to list "Char X Y"
                    label_strings.append(f"{char} {center_x} {center_y}")

                # Join all predictions for this image
                full_label_str = " ".join(label_strings)
                results.append({"image_id": image_id, "labels": full_label_str})

    # 6. Generate Submission File
    submission_df = pd.DataFrame(results)

    # Ensure correct column order
    submission_df = submission_df[["image_id", "labels"]]

    # Save to CSV
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved successfully to {config.SUBMISSION_PATH}")
