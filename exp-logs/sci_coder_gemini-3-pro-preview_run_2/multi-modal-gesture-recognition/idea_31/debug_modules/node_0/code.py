import os
import torch
import numpy as np
import shutil
import logging
import sys

# Import from the provided library
from library.config import (
    INPUT_DIM,
    HIDDEN_DIM,
    NUM_CLASSES,
    WORKING_DIR,
    SUBMISSION_DIR,
    BATCH_SIZE,
)
from library.utils import set_seed, setup_logger
from library.data_loader import get_dataloaders
from library.model import MSE_GCN
from library.losses import TotalLoss
from library.train import Trainer
from library.inference import Predictor

# Define a custom logger for this demonstration script
logger = logging.getLogger("DemoScript")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


def main():
    # 1. Setup
    logger.info("Setting up environment...")
    set_seed(42)

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.join(WORKING_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # 2. Data Loading Demonstration
    logger.info("Initializing DataLoaders...")
    # get_dataloaders handles caching. First run might take a moment to process metadata.
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, load_cached_data=True
    )

    logger.info(f"Train Loader Batches: {len(train_loader)}")
    logger.info(f"Val Loader Batches: {len(val_loader)}")
    logger.info(f"Test Loader Batches: {len(test_loader)}")

    # Fetch a single batch to verify structure
    logger.info("Fetching a single training batch...")
    batch = next(iter(train_loader))

    features = batch["features"]
    cls_labels = batch["cls_labels"]
    bnd_labels = batch["bnd_labels"]
    mask = batch["mask"]
    lengths = batch["lengths"]
    ids = batch["sample_ids"]

    # Assertions to verify data shapes
    # Features: (B, T, D) -> D should be 85 (36 pos + 36 vel + 13 audio)
    assert features.dim() == 3, f"Expected 3D features, got {features.shape}"
    assert (
        features.shape[2] == INPUT_DIM
    ), f"Expected feature dim {INPUT_DIM}, got {features.shape[2]}"
    assert mask.shape == (features.shape[0], features.shape[1]), "Mask shape mismatch"
    logger.info(f"Batch shapes verified: Features {features.shape}, Mask {mask.shape}")

    # 3. Model Instantiation and Forward Pass
    logger.info("Instantiating MSE_GCN Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MSE_GCN().to(device)

    # Move batch to device
    features = features.to(device)
    mask = mask.to(device)
    lengths = lengths.to(device)
    cls_labels = cls_labels.to(device)
    bnd_labels = bnd_labels.to(device)

    logger.info("Running forward pass...")
    stage_outputs = model(features, mask, lengths)

    # Verify Output
    assert isinstance(
        stage_outputs, list
    ), "Model output should be a list (multi-stage)"
    assert len(stage_outputs) == 3, f"Expected 3 stages, got {len(stage_outputs)}"

    final_stage = stage_outputs[-1]
    assert "cls" in final_stage and "bnd" in final_stage, "Missing keys in model output"

    # Check shape of classification output: (B, T, NUM_CLASSES)
    cls_out = final_stage["cls"]
    assert cls_out.shape == (
        features.shape[0],
        features.shape[1],
        NUM_CLASSES,
    ), f"Class output shape mismatch: {cls_out.shape}"

    logger.info("Forward pass successful. Output shapes verified.")

    # 4. Loss Calculation
    logger.info("Calculating Loss...")
    criterion = TotalLoss().to(device)

    loss, metrics = criterion(stage_outputs, cls_labels, bnd_labels, mask)

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    logger.info(f"Loss computed: {loss.item():.4f}")
    logger.info(f"Metrics: {metrics}")

    # 5. Training Loop Demonstration (Optimization Step)
    logger.info("Demonstrating Optimization Step...")
    # We use the Trainer class but manually run a few steps to save time
    trainer = Trainer()
    # Ensure trainer uses the same device
    assert str(trainer.device) == str(device)

    trainer.model.train()

    # Run optimization on the fetched batch
    trainer.optimizer.zero_grad()

    # Re-run forward with trainer's model
    out = trainer.model(features, mask, lengths)
    loss_val, _ = trainer.criterion(out, cls_labels, bnd_labels, mask)
    loss_val.backward()
    trainer.optimizer.step()

    logger.info("Optimization step completed successfully.")

    # 6. Save Dummy Checkpoint for Inference
    logger.info("Saving dummy checkpoint for inference testing...")
    checkpoint_path = os.path.join(WORKING_DIR, "checkpoints", "best_model.pth")
    torch.save(trainer.model.state_dict(), checkpoint_path)
    assert os.path.exists(checkpoint_path), "Checkpoint file not created"

    # 7. Inference Demonstration
    logger.info("Initializing Predictor...")
    predictor = Predictor(model_path=checkpoint_path)

    logger.info("Running inference on a subset of test data (1 batch)...")
    # Manually iterate to limit runtime
    test_batch = next(iter(test_loader))

    # Mock a small dataloader or just pass tensors manually?
    # The Predictor.predict method expects a DataLoader.
    # We will create a list containing just one batch to simulate a small loader.
    class MockLoader:
        def __init__(self, batch):
            self.batch = batch

        def __iter__(self):
            yield self.batch

        def __len__(self):
            return 1

    mock_loader = MockLoader(test_batch)

    predictions = predictor.predict(mock_loader)

    logger.info(f"Predictions generated for {len(predictions)} samples.")

    # Verify prediction format
    sample_id = test_batch["sample_ids"][0]
    if sample_id in predictions:
        pred_seq = predictions[sample_id]
        logger.info(f"Sample {sample_id} Predicted Sequence: {pred_seq}")
        assert isinstance(pred_seq, list), "Prediction should be a list"
        # Check elements are integers
        if len(pred_seq) > 0:
            assert isinstance(
                pred_seq[0], int
            ), "Prediction elements should be integers"
    else:
        logger.error(f"Sample ID {sample_id} not found in predictions!")
        raise AssertionError("Prediction logic failed to cover input batch.")

    # 8. Submission File Generation Logic
    logger.info("Demonstrating submission file creation...")
    submission_file = os.path.join(SUBMISSION_DIR, "submission_demo.csv")

    with open(submission_file, "w") as f:
        for sid, seq in predictions.items():
            seq_str = ",".join(map(str, seq))
            f.write(f"{sid},{seq_str}\n")

    assert os.path.exists(submission_file), "Submission file was not written"
    logger.info(f"Demo submission written to {submission_file}")

    logger.info("All demonstrations completed successfully.")


if __name__ == "__main__":
    main()
