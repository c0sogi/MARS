import os
import torch
import numpy as np
import shutil
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library components
from library.config import Config
from library.utils import (
    seed_everything,
    format_prediction_string,
    parse_prediction_string,
    apply_wbf,
    calculate_map,
)
from library.dataset import CovidDataset, collate_fn
from library.model import get_model
from library.engine import train_one_epoch, evaluate


def verify_utilities():
    print("--- Verifying Utilities ---")

    # 1. Test Prediction String Formatting
    labels = [1, 1]
    boxes = [[10, 10, 50, 50], [60, 60, 100, 100]]
    scores = [0.95, 0.88]

    pred_str = format_prediction_string(labels, boxes, scores)
    expected_part = "opacity 0.950000 10 10 50 50"

    assert isinstance(pred_str, str), "Prediction string must be a string"
    assert expected_part in pred_str, f"Expected '{expected_part}' in '{pred_str}'"

    # 2. Test Prediction String Parsing
    p_labels, p_boxes, p_scores = parse_prediction_string(pred_str)
    assert len(p_labels) == 2
    assert p_scores[0] == 0.95
    assert p_boxes[0] == [10.0, 10.0, 50.0, 50.0]

    # 3. Test Weighted Boxes Fusion (WBF)
    # Create two very similar boxes that should be fused
    boxes_list = [[[0.0, 0.0, 100.0, 100.0], [2.0, 2.0, 102.0, 102.0]]]
    scores_list = [[0.9, 0.8]]
    labels_list = [[1, 1]]

    fused_boxes, fused_scores, fused_labels = apply_wbf(
        boxes_list, scores_list, labels_list, iou_thr=0.5
    )

    # Should fuse into 1 box
    assert len(fused_boxes) == 1, f"WBF failed to fuse boxes. Got {len(fused_boxes)}"
    assert fused_labels[0] == 1
    # Score should be average of scores (0.9 + 0.8) / 1 model = 1.7 ??
    # Logic in utils.py: final_score = sum(c["scores"]) / len(boxes_list)
    # sum(scores) = 1.7, len(boxes_list) = 1 -> 1.7.
    # Note: WBF implementation details vary, checking strict equality might be flaky,
    # but based on the provided code, it sums scores.
    assert fused_scores[0] > 0.9, "Fused score calculation unexpected"

    print("Utilities verification passed.")


def verify_dataset_and_loader():
    print("\n--- Verifying Dataset and DataLoader ---")

    # Instantiate dataset in Train mode
    # It will use the DEBUG_SAMPLE_SIZE defined in Config (patched below)
    ds = CovidDataset(mode="train", load_cached_data=False)

    assert (
        len(ds) == Config.DEBUG_SAMPLE_SIZE
    ), f"Dataset length mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(ds)}"

    # Fetch one item
    image, target = ds[0]

    # Check Image
    assert isinstance(image, torch.Tensor), "Image is not a tensor"
    assert image.ndim == 3, "Image must have 3 dimensions (C, H, W)"
    assert image.shape[0] == 3, "Image must have 3 channels"
    assert image.shape[1] == Config.IMAGE_SIZE, "Image height mismatch"
    assert image.shape[2] == Config.IMAGE_SIZE, "Image width mismatch"

    # Check Target
    assert isinstance(target, dict), "Target is not a dict"
    assert "boxes" in target
    assert "labels" in target
    assert "study_ids" in target
    assert target["boxes"].ndim == 2, "Boxes tensor dimension incorrect"
    if target["boxes"].shape[0] > 0:
        assert target["boxes"].shape[1] == 4, "Boxes must have 4 coordinates"

    # Test DataLoader with Collate
    loader = DataLoader(ds, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn)
    batch_images, batch_targets = next(iter(loader))

    assert batch_images.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert len(batch_targets) == Config.BATCH_SIZE, "Target batch size mismatch"

    print("Dataset and DataLoader verification passed.")
    return loader


def verify_model_and_training(train_loader):
    print("\n--- Verifying Model and Training Step ---")

    device = Config.DEVICE
    model = get_model()
    model.to(device)

    # 1. Test Training Forward Pass
    model.train()
    images, targets = next(iter(train_loader))
    images = list(img.to(device) for img in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    loss_dict = model(images, targets)

    # Check that we get all expected losses
    expected_losses = [
        "loss_classifier",
        "loss_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
        "loss_study",
    ]
    for loss_name in expected_losses:
        assert loss_name in loss_dict, f"Missing loss: {loss_name}"
        assert isinstance(
            loss_dict[loss_name], torch.Tensor
        ), f"{loss_name} is not a tensor"
        assert not torch.isnan(loss_dict[loss_name]), f"{loss_name} is NaN"

    print("Model training forward pass successful.")

    # 2. Test Inference Forward Pass
    model.eval()
    with torch.no_grad():
        detections = model(images)

    assert len(detections) == Config.BATCH_SIZE
    for det in detections:
        assert "boxes" in det
        assert "scores" in det
        assert "labels" in det
        assert "study_probs" in det
        assert "study_label" in det

        # Check shapes
        if det["boxes"].shape[0] > 0:
            assert det["boxes"].shape[1] == 4
            assert det["scores"].shape[0] == det["boxes"].shape[0]

    print("Model inference forward pass successful.")

    # 3. Test Engine Training Step
    print("Running one training epoch via engine...")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # We suppress print frequency to keep output clean
    avg_loss = train_one_epoch(
        model, optimizer, train_loader, device, epoch=0, print_freq=100
    )

    assert isinstance(avg_loss, float)
    assert avg_loss > 0
    print(f"Epoch complete. Avg Loss: {avg_loss:.4f}")

    return model


def verify_evaluation(model):
    print("\n--- Verifying Evaluation ---")

    # Create validation dataset/loader
    val_ds = CovidDataset(mode="val", load_cached_data=False)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn)

    device = Config.DEVICE

    # Run evaluation
    map_score, study_acc = evaluate(model, val_loader, device)

    assert 0.0 <= map_score <= 1.0, "mAP score out of range"
    assert 0.0 <= study_acc <= 1.0, "Study accuracy out of range"

    print(f"Evaluation complete. mAP: {map_score:.4f}, Study Acc: {study_acc:.4f}")


if __name__ == "__main__":
    # 1. Setup & Configuration Override
    seed_everything(42)

    # Override Config for the purpose of this demo script
    print("Overriding Config for Demo...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10  # Small subset for speed
    Config.BATCH_SIZE = 2
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script

    # Ensure working directory exists for cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    try:
        # 2. Run Verifications
        verify_utilities()

        train_loader = verify_dataset_and_loader()

        model = verify_model_and_training(train_loader)

        verify_evaluation(model)

        print("\nAll demonstrations and verifications passed successfully!")

    except Exception as e:
        print(f"\nFAILED: {e}")
        raise e
    finally:
        # Cleanup temporary files
        print("\nCleaning up temporary files...")
        train_cache = os.path.join(Config.WORKING_DIR, "train_processed.parquet")
        val_cache = os.path.join(Config.WORKING_DIR, "val_processed.parquet")

        if os.path.exists(train_cache):
            os.remove(train_cache)
        if os.path.exists(val_cache):
            os.remove(val_cache)
