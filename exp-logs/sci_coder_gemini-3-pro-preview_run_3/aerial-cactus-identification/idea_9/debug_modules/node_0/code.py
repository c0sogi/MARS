import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import load_data, CactusDataset, get_transforms
from library.model_factory import get_model
from library.trainer import CactusTrainer
from library.stacking import StackingMetaLearner


def run_demo():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[1] Setting up Configuration...")

    # Modify Config for a fast demonstration
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 64  # Small subset for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 16
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Clean working directory for a fresh run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Initialize Logger and Seeds
    logger = setup_logger(
        "DemoLogger", log_file=os.path.join(Config.WORKING_DIR, "demo.log")
    )
    set_seed(Config.SEED)

    logger.info(f"Device: {Config.DEVICE}")
    logger.info("Configuration configured for demo mode.")

    # ==========================================
    # 2. Data Loading & Processing
    # ==========================================
    logger.info("\n[2] Loading Data...")

    # Load Training Data (Subset due to DEBUG=True)
    # Note: load_data handles caching automatically
    train_imgs, train_lbls, train_ids = load_data(
        Config.TRAIN_METADATA_PATH, cache_prefix="train", load_cached_data=False
    )

    # Load Validation Data
    val_imgs, val_lbls, val_ids = load_data(
        Config.VAL_METADATA_PATH, cache_prefix="val", load_cached_data=False
    )

    # Verify Data Shapes
    assert (
        len(train_imgs) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} train samples, got {len(train_imgs)}"
    assert train_imgs.shape[1:] == (32, 32, 3), "Image shape mismatch"
    assert len(train_lbls) == len(train_imgs), "Label count mismatch"

    logger.info(f"Train Data Shape: {train_imgs.shape}")
    logger.info(f"Val Data Shape: {val_imgs.shape}")

    # Create Datasets and DataLoaders
    train_dataset = CactusDataset(
        train_imgs, train_lbls, transform=get_transforms("train")
    )
    val_dataset = CactusDataset(val_imgs, val_lbls, transform=get_transforms("val"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # 0 workers for simple debug
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    logger.info("DataLoaders initialized successfully.")

    # ==========================================
    # 3. Model Initialization & Verification
    # ==========================================
    logger.info("\n[3] Initializing Model...")

    model_name = "resnet34"
    # Using pretrained=False for speed/offline safety in demo
    model = get_model(model_name, pretrained=False)

    # Verify Stem Surgery (Specific to the provided model_factory logic)
    # ResNet34 stem surgery changes conv1 kernel size to 3 and stride to 1
    assert model.conv1.kernel_size == (
        3,
        3,
    ), "Stem surgery failed: Kernel size incorrect"
    assert model.conv1.stride == (1, 1), "Stem surgery failed: Stride incorrect"
    assert isinstance(
        model.maxpool, nn.Identity
    ), "Stem surgery failed: MaxPool not removed"

    # Verify Head Adaptation
    assert (
        model.fc.out_features == Config.NUM_CLASSES
    ), "Classification head output dimension incorrect"

    logger.info(f"Model {model_name} initialized and verified.")

    # ==========================================
    # 4. Training Loop
    # ==========================================
    logger.info("\n[4] Starting Training...")

    trainer = CactusTrainer(model, device=torch.device(Config.DEVICE), logger=logger)

    save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    best_auc = trainer.fit(
        train_loader,
        val_loader,
        epochs=Config.EPOCHS,
        patience=1,  # Strict patience for demo
        save_path=save_path,
    )

    assert os.path.exists(save_path), "Model checkpoint was not saved."
    assert isinstance(best_auc, float), "Best AUC is not a float."

    logger.info(f"Training completed. Best AUC: {best_auc:.4f}")

    # ==========================================
    # 5. Inference & Test-Time Augmentation
    # ==========================================
    logger.info("\n[5] Running Inference with TTA...")

    # We use the validation set as a proxy for test set in this demo
    # In a real scenario, we would load test data using Config.TEST_METADATA_PATH

    # Load test data (using test metadata)
    test_imgs, _, test_ids = load_data(
        Config.TEST_METADATA_PATH,
        cache_prefix="test_full",  # Use a different prefix to avoid debug sampling collision if needed
        load_cached_data=False,
    )

    # Create test dataset (no labels needed, but loader returns dummy)
    test_dataset = CactusDataset(test_imgs, transform=get_transforms("test"))
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Predict
    preds = trainer.predict_with_tta(test_loader)

    assert len(preds) == len(test_ids), "Prediction count does not match ID count."
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions out of probability range [0, 1]."

    # Generate Submission CSV
    trainer.generate_submission(
        test_loader, test_ids, output_path=Config.SUBMISSION_PATH
    )
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."

    logger.info(f"Inference complete. Submission saved to {Config.SUBMISSION_PATH}")

    # ==========================================
    # 6. Stacking Meta-Learner
    # ==========================================
    logger.info("\n[6] Demonstrating Stacking Meta-Learner...")

    # Simulate OOF predictions for 2 models on the validation set
    # In practice, these come from cross-validation
    n_samples = len(val_lbls)

    # Synthetic predictions for Model A (slightly noisy ground truth)
    # Adding noise to ensure it's not perfect
    noise_a = np.random.normal(0, 0.2, n_samples)
    preds_model_a = np.clip(val_lbls + noise_a, 0.1, 0.9)

    # Synthetic predictions for Model B (random guess)
    preds_model_b = np.random.rand(n_samples)

    oof_preds = {"resnet34": preds_model_a, "densenet121": preds_model_b}

    # Initialize and Train Meta-Learner
    meta_learner = StackingMetaLearner(random_state=Config.SEED)
    meta_auc = meta_learner.train(oof_preds, val_lbls)

    logger.info(f"Meta-Learner OOF AUC: {meta_auc:.4f}")

    # Verify Meta-Learner Logic
    # Since Model A is correlated with truth, its coefficient should be positive and higher than B
    coefs = meta_learner.model.coef_[0]
    # Features are sorted alphabetically: ['densenet121', 'resnet34']
    # So index 0 is densenet, index 1 is resnet
    logger.info(f"Coefficients: {coefs}")

    # Save Meta-Learner
    meta_model_path = os.path.join(Config.WORKING_DIR, "meta_learner.pkl")
    # We manually save using pickle as demonstrated in the class method (it calls pickle internally)
    # The class has a .train method that accepts save_path, but let's verify saving explicitly
    # Re-training with save path
    meta_learner.train(oof_preds, val_lbls, save_path=meta_model_path)
    assert os.path.exists(meta_model_path), "Meta-learner model file not found."

    # Load Meta-Learner
    meta_learner_loaded = StackingMetaLearner()
    meta_learner_loaded.load(meta_model_path)

    # Simulate Test Predictions
    test_preds_a = np.random.rand(10)
    test_preds_b = np.random.rand(10)
    test_preds_dict = {"resnet34": test_preds_a, "densenet121": test_preds_b}

    final_preds = meta_learner_loaded.predict(test_preds_dict)
    assert len(final_preds) == 10, "Meta-learner prediction shape incorrect."

    logger.info("Stacking demonstration complete.")
    logger.info("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
