import os
from ultralytics import YOLO
from library.config import (
    IDEA_DIR,
    YOLO_DATASET_DIR,
    SEED,
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
    seed_everything,
)
from library.data_setup import prepare_yolo_data


def train_model(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    img_size=IMG_SIZE,
    debug_sample_size=None,
    load_cached_data=True,
):
    """
    Initializes and trains the YOLOv8 model.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        img_size (int): Input image resolution.
        debug_sample_size (int, optional): Number of samples to use for debugging.
        load_cached_data (bool): Whether to use cached preprocessed data.

    Returns:
        str: Path to the best model weights file.
    """
    # 1. Set Seed for Reproducibility
    seed_everything(SEED)

    # 2. Prepare Data
    # This ensures images are converted and data.yaml is created
    print("Ensuring data is ready for training...")
    prepare_yolo_data(sample_size=debug_sample_size, load_cached_data=load_cached_data)

    yaml_path = os.path.join(YOLO_DATASET_DIR, "data.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Data configuration file not found at {yaml_path}")

    # 3. Initialize Model
    # Load a pretrained YOLOv8s model (Small) for better performance
    print("Initializing YOLOv8s model...")
    model = YOLO("yolov8s.pt")

    # 4. Configure Output Paths
    project_dir = os.path.join(IDEA_DIR, "training")
    run_name = "run"

    # Clean up previous run if exists to ensure clean metrics logging (optional but recommended)
    # Ultralytics handles 'exist_ok', but we want to be sure about the 'best.pt' path.

    # 5. Train Model
    print(f"Starting training for {epochs} epochs...")
    # verbose=False suppresses the progress bar but keeps epoch summary
    results = model.train(
        data=yaml_path,
        project=project_dir,
        name=run_name,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        seed=SEED,
        patience=5,  # Early stopping patience
        device=0,  # Use first GPU
        workers=4,
        exist_ok=True,  # Overwrite existing run directory
        verbose=False,  # Suppress per-batch progress bar
        val=True,  # Run validation
        plots=False,  # Disable plot generation to save time/space
    )

    # 6. Report Metrics
    # The 'results' object usually contains metrics.
    # In recent Ultralytics versions, results.results_dict or similar holds the values.
    print("\n==== Training Completed ====")

    # Attempt to print metrics with full precision
    try:
        # Accessing standard YOLO metrics
        # map50: mAP at IoU=0.5
        # map: mAP at IoU=0.5:0.95
        metrics = model.val(split="val", verbose=False)

        print("Validation Metrics (Full Precision):")
        print(f"mAP@50: {metrics.box.map50}")
        print(f"mAP@50-95: {metrics.box.map}")

        # Detailed class-wise metrics if needed are in metrics.box.maps
    except Exception as e:
        print(f"Could not print detailed metrics object: {e}")

    # 7. Return Best Weights Path
    best_weights_path = os.path.join(project_dir, run_name, "weights", "best.pt")

    if os.path.exists(best_weights_path):
        print(f"Best model weights saved at: {best_weights_path}")
        return best_weights_path
    else:
        # Fallback if 'best.pt' wasn't saved (e.g., if training failed or 0 epochs)
        # Try 'last.pt'
        last_weights_path = os.path.join(project_dir, run_name, "weights", "last.pt")
        if os.path.exists(last_weights_path):
            print(
                f"Warning: 'best.pt' not found. Returning 'last.pt': {last_weights_path}"
            )
            return last_weights_path
        else:
            raise FileNotFoundError("No model weights found after training.")
