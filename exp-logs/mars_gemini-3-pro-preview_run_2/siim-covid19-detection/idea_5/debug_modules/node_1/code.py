import os
import torch
import numpy as np
import pandas as pd
import shutil
import sys

# Import library modules
import library.config as config
import library.utils as utils
import library.dataset as dataset
import library.model as model_lib
import library.engine as engine


def run_demonstration():
    print("=== Starting Demonstration ===")

    # 1. Configuration & Monkey Patching for Speed
    # We override constants to ensure the demo runs quickly within the time limit.
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config constants
    config.NUM_EPOCHS = 1
    config.BATCH_SIZE = 4
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 20  # Small sample for demo

    # Override Engine constants (since they are imported into engine's namespace)
    engine.NUM_EPOCHS = 1
    engine.BATCH_SIZE = 4

    # Ensure working directory is clean for this run
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    utils.seed_everything(config.SEED)
    print("Configuration updated for demo mode.")

    # 2. Verify Utility Functions
    print("\n[2] Verifying Utility Functions...")

    # Test format_prediction_string
    dummy_boxes = [[10, 10, 50, 50], [60, 60, 100, 100]]
    dummy_scores = [0.95, 0.88]
    dummy_labels = [1, 1]
    pred_str = utils.format_prediction_string(dummy_boxes, dummy_scores, dummy_labels)
    expected_substr = "opacity 0.950000 10.0 10.0 50.0 50.0"
    assert (
        expected_substr in pred_str
    ), f"Prediction string format incorrect: {pred_str}"
    print("format_prediction_string: OK")

    # Test format_study_prediction_string
    study_str = utils.format_study_prediction_string("Negative for Pneumonia", 0.99)
    assert (
        study_str == "negative 0.990000 0 0 1 1"
    ), f"Study string format incorrect: {study_str}"
    print("format_study_prediction_string: OK")

    # 3. Verify Dataset Loading
    print("\n[3] Verifying Dataset...")

    # Initialize Train Dataset in Debug mode
    train_ds = dataset.CovidDataset(
        mode="train",
        transforms=utils.get_train_transforms(),
        load_cached_data=False,  # Force reload to apply debug slicing correctly on raw data if needed
        debug=True,
    )

    print(f"Train Dataset Size (Debug): {len(train_ds)}")
    assert (
        len(train_ds) == config.DEBUG_SAMPLE_SIZE
    ), "Dataset did not respect debug sample size."

    # Fetch one item
    img, target, img_id = train_ds[0]

    # Verify Image
    assert isinstance(img, torch.Tensor), "Image is not a tensor"
    assert img.shape == (
        3,
        config.IMG_SIZE,
        config.IMG_SIZE,
    ), f"Image shape mismatch: {img.shape}"

    # Verify Target
    assert isinstance(target, dict), "Target is not a dict"
    assert "boxes" in target, "Target missing boxes"
    assert "study_label" in target, "Target missing study_label"
    assert (
        target["boxes"].dim() == 2 and target["boxes"].shape[1] == 4
    ), "Boxes shape incorrect"

    print("Dataset structure verified.")

    # 4. Verify Model Architecture
    print("\n[4] Verifying Model Architecture...")

    device = config.DEVICE
    net = model_lib.get_model()
    net.to(device)

    # Create a dummy batch
    imgs_batch = [img.to(device)]
    targets_batch = [{k: v.to(device) for k, v in target.items()}]

    # Test Training Forward Pass
    net.train()
    loss_dict = net(imgs_batch, targets_batch)

    print("Loss Keys:", loss_dict.keys())
    assert "loss_classifier" in loss_dict, "Missing classifier loss"
    assert "loss_box_reg" in loss_dict, "Missing box regression loss"
    assert "loss_mil" in loss_dict, "Missing MIL loss"

    # Test Inference Forward Pass
    net.eval()
    with torch.no_grad():
        detections = net(imgs_batch)

    assert isinstance(detections, list), "Inference output should be a list"
    assert "boxes" in detections[0], "Detection missing boxes"
    assert (
        "study_logits" in detections[0]
    ), "Detection missing study logits (MIL output)"

    print("Model forward passes verified.")

    # 5. Run Training Loop (Engine)
    print("\n[5] Running Training Loop (Short Demo)...")

    # We call the fit function from the engine.
    # We've already patched engine.NUM_EPOCHS and engine.BATCH_SIZE.
    try:
        engine.fit(load_cached_data=False, debug=True)
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e

    # Check if model was saved
    model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(model_path), "best_model.pth was not created."
    print("Training loop completed and model saved.")

    # 6. Inference and Submission Generation
    print("\n[6] Generating Inference on Test Set...")

    # Load the trained model
    net = model_lib.get_model()
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    # Setup Test Dataset
    test_ds = dataset.CovidDataset(
        mode="test",
        transforms=utils.get_valid_transforms(),
        load_cached_data=False,
        debug=True,  # Small subset for speed
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=utils.collate_fn,
    )

    results = []

    print(f"Running inference on {len(test_ds)} test images...")
    with torch.no_grad():
        for images, targets, image_ids in test_loader:
            images = list(img.to(device) for img in images)

            # Forward pass
            outputs = net(images)

            for i, output in enumerate(outputs):
                img_id = image_ids[i]

                # 1. Process Study Level Prediction (MIL Head)
                # output['study_probs'] is (Num_Study_Classes,)
                study_probs = output["study_probs"].cpu().numpy()

                # We need to output a prediction for each study label class
                # The task requires at least one label. We usually output the argmax or all.
                # Here we demonstrate outputting the max confidence class.
                best_study_idx = np.argmax(study_probs)
                best_study_label = config.STUDY_ID_TO_LABEL[best_study_idx]
                best_study_score = study_probs[best_study_idx]

                study_pred_str = utils.format_study_prediction_string(
                    best_study_label, best_study_score
                )

                # Note: In the real competition, study IDs are separate rows in submission.
                # The provided metadata links image_id to StudyInstanceUID.
                # For this demo, we just print the formatted string.

                # 2. Process Image Level Prediction (Detection Head)
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                # Filter by threshold
                mask = scores > 0.2  # Arbitrary threshold for demo
                boxes = boxes[mask]
                scores = scores[mask]
                labels = labels[mask]

                image_pred_str = utils.format_prediction_string(boxes, scores, labels)

                results.append(
                    {
                        "image_id": img_id,
                        "study_pred": study_pred_str,
                        "image_pred": image_pred_str,
                    }
                )

    # Verify results
    assert len(results) > 0, "No results generated."
    print("Sample Result:")
    print(f"ID: {results[0]['image_id']}")
    print(f"Study Prediction: {results[0]['study_pred']}")
    print(f"Image Prediction: {results[0]['image_pred']}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
