import os
import torch
import pandas as pd
import warnings
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config, seed_everything
from library.dataset import KuzushijiDataset
from library.utils import get_transforms, collate_fn
from library.model import get_model
from library.engine import fit
from library.inference import predict_and_submit


def main():
    # 1. Setup Configuration and Seeding
    # Enable debug mode to reduce epochs to 1 and define small sample sizes
    config = Config(debug=True, num_epochs=1)
    seed_everything(config.SEED)

    print(f"Configuration initialized. Debug: {config.DEBUG}, Device: {config.DEVICE}")
    print(f"Directories: Input={config.INPUT_DIR}, Working={config.WORKING_DIR}")

    # 2. Prepare Datasets and Dataloaders
    print("\n--- Preparing Data ---")

    # Initialize Training Dataset
    train_dataset = KuzushijiDataset(
        split="train", config=config, transforms=get_transforms(train=True)
    )

    # Initialize Validation Dataset
    val_dataset = KuzushijiDataset(
        split="val", config=config, transforms=get_transforms(train=False)
    )

    # Optimization: Manually slice the dataframes to the debug sample sizes
    # This ensures the training loop finishes very quickly for demonstration purposes.
    if config.TRAIN_SAMPLE_SIZE:
        original_len = len(train_dataset)
        train_dataset.df = train_dataset.df.iloc[: config.TRAIN_SAMPLE_SIZE]
        print(f"Sliced Train Dataset: {original_len} -> {len(train_dataset)} samples")

    if config.VAL_SAMPLE_SIZE:
        original_len = len(val_dataset)
        val_dataset.df = val_dataset.df.iloc[: config.VAL_SAMPLE_SIZE]
        print(f"Sliced Val Dataset: {original_len} -> {len(val_dataset)} samples")

    # Verify dataset integrity
    img, target = train_dataset[0]
    assert isinstance(img, torch.Tensor), "Dataset should return image as Tensor"
    assert "boxes" in target, "Target dict must contain 'boxes'"
    assert "labels" in target, "Target dict must contain 'labels'"

    # Create Dataloaders
    # num_workers=0 to avoid multiprocessing overhead in this short script
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # 3. Initialize Model
    print("\n--- Initializing Model ---")
    model = get_model(num_classes=config.NUM_CLASSES, config=config)

    # Logic Check: Verify the predictor head has the correct number of classes
    # The cls_score layer should have output features equal to NUM_CLASSES
    predictor_classes = model.roi_heads.box_predictor.cls_score.out_features
    assert (
        predictor_classes == config.NUM_CLASSES
    ), f"Model initialized with {predictor_classes} classes, expected {config.NUM_CLASSES}"

    model.to(config.DEVICE)
    print("Model moved to device successfully.")

    # 4. Train Model
    print("\n--- Starting Training (Demo) ---")
    # The fit function handles the training loop, validation, and saving the best model.
    best_f1 = fit(model, train_loader, val_loader, config)

    print(f"Training complete. Best Validation F1: {best_f1}")

    # Ensure a model file exists for inference.
    # 'fit' only saves if validation F1 improves. In a short debug run, F1 might remain 0.
    if not os.path.exists(config.MODEL_PATH):
        print(
            "Validation F1 did not improve or was 0. Forcing model save for inference demo."
        )
        torch.save(model.state_dict(), config.MODEL_PATH)

    assert os.path.exists(config.MODEL_PATH), "Model file must exist before inference."

    # 5. Inference and Submission
    print("\n--- Starting Inference & Submission ---")
    # predict_and_submit loads the test set (metadata/test.csv) and the saved model,
    # then generates the submission CSV.
    predict_and_submit(config)

    # 6. Verify Submission
    if os.path.exists(config.SUBMISSION_PATH):
        df_sub = pd.read_csv(config.SUBMISSION_PATH)
        print(f"\nSubmission file generated at: {config.SUBMISSION_PATH}")
        print(f"Rows: {len(df_sub)}")
        print("Head:")
        print(df_sub.head())

        # Validation assertions
        assert "image_id" in df_sub.columns, "Submission missing 'image_id' column"
        assert "labels" in df_sub.columns, "Submission missing 'labels' column"
        assert len(df_sub) > 0, "Submission file is empty"
    else:
        raise FileNotFoundError(
            f"Submission file not found at {config.SUBMISSION_PATH}"
        )

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    main()
