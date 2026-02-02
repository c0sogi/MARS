import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library modules
from library import config, utils, dataset, model, train_engine, inference_engine


def main():
    # 1. Setup
    utils.seed_everything(42)
    device = config.DEVICE
    print(f"Device: {device}")

    # 2. Data Loading
    # We load a small subset for training/validation to optimize for speed as requested.
    # We load the full test set to ensure the submission file is generated correctly for all IDs.
    print("Loading datasets...")
    train_imgs, train_lbls, train_ids = dataset.load_data(
        config.TRAIN_METADATA_PATH, "train", sample_size=200
    )
    val_imgs, val_lbls, val_ids = dataset.load_data(
        config.VAL_METADATA_PATH, "val", sample_size=100
    )
    test_imgs, test_lbls, test_ids = dataset.load_data(
        config.TEST_METADATA_PATH, "test"
    )

    # Verify data loading logic
    assert len(train_imgs) == 200, "Training subset size mismatch."
    assert len(val_imgs) == 100, "Validation subset size mismatch."
    assert train_imgs.shape[1:] == (
        32,
        32,
        3,
    ), f"Incorrect image shape: {train_imgs.shape}"
    print(
        f"Data loaded. Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}"
    )

    # 3. Dataset and DataLoader Creation
    train_ds = dataset.CactusDataset(
        train_imgs, train_lbls, train_ids, transform=dataset.get_transforms("train")
    )
    val_ds = dataset.CactusDataset(
        val_imgs, val_lbls, val_ids, transform=dataset.get_transforms("val")
    )
    test_ds = dataset.CactusDataset(
        test_imgs, test_lbls, test_ids, transform=dataset.get_transforms("test")
    )

    # Use a reasonable batch size
    batch_size = 32
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 4. Model Initialization & Logic Check
    print("Initializing model...")
    net = model.MicroConvNeXt(num_classes=1).to(device)

    # Verify model output shape with a dummy input
    dummy_input = torch.randn(2, 3, 32, 32).to(device)
    with torch.no_grad():
        dummy_output = net(dummy_input)

    assert dummy_output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {dummy_output.shape}"
    print("Model logic verification passed.")

    # 5. Training Execution
    print("Starting training demonstration...")
    save_path = os.path.join(config.WORKING_DIR, "demo_best_model.pth")

    # Train for 2 epochs to demonstrate the loop and saving mechanism quickly
    best_auc = train_engine.train_model(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_path=save_path,
        epochs=2,
        lr=1e-3,
        weight_decay=0.01,
        patience=2,
    )

    # Verify that the model file was created (implies validation AUC > 0, which is expected)
    if not os.path.exists(save_path):
        # Fallback: If model wasn't saved (e.g. extremely poor random init), save current state
        # This ensures the rest of the demo script runs.
        torch.save(net.state_dict(), save_path)
        print(
            "Note: Model was not saved by trainer (likely low AUC), saved manually for demo continuity."
        )
    else:
        print(f"Training complete. Best AUC: {best_auc}")

    # 6. Inference with Test Time Augmentation (TTA)
    print("Running inference on test set...")

    # Load the best saved weights
    net.load_state_dict(torch.load(save_path, map_location=device))

    # Generate predictions
    predictions = inference_engine.predict_with_tta(net, test_loader, device)

    # Verify predictions
    assert len(predictions) == len(
        test_ids
    ), "Number of predictions does not match number of test IDs."
    assert np.all(
        (predictions >= 0) & (predictions <= 1)
    ), "Probabilities must be between 0 and 1."
    print("Inference complete.")

    # 7. Submission Generation
    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    inference_engine.save_submission(test_ids, predictions, submission_path)

    # Verify submission file
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    print(f"Submission loaded. Shape: {df_sub.shape}")

    assert list(df_sub.columns) == ["id", "has_cactus"], "Submission columns mismatch."
    assert len(df_sub) == len(test_ids), "Submission row count mismatch."
    assert df_sub["id"].nunique() == len(test_ids), "Duplicate IDs found in submission."

    print("Pipeline demonstration completed successfully.")


if __name__ == "__main__":
    main()
