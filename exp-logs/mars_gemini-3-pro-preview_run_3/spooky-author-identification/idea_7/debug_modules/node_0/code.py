import os
import pandas as pd
import numpy as np
import torch
import shutil
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, compute_log_loss
from library.data_loader import load_raw_data, get_stylometric_features, AuthorDataset
from library.statistical_models import StatisticalModel
from library.pretraining import train_mlm
from library.neural_models import CustomTransformer
from library.distillation_engine import DistillationEngine
from library.ensemble_optimizer import EnsembleOptimizer
from transformers import AutoTokenizer


def create_demo_data():
    """
    Creates a tiny subset of the data for demonstration purposes to ensure
    the script runs quickly.
    """
    print("--- Creating Demo Datasets (Subset) ---")

    # Load original metadata
    train_df = pd.read_csv("./metadata/train.csv")
    val_df = pd.read_csv("./metadata/val.csv")
    test_df = pd.read_csv("./metadata/test.csv")

    # Sample 50 rows for speed
    demo_train = train_df.head(50).copy()
    demo_val = val_df.head(50).copy()
    demo_test = test_df.head(50).copy()

    # Define demo paths
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    demo_train_path = os.path.join(Config.WORKING_DIR, "demo_train.csv")
    demo_val_path = os.path.join(Config.WORKING_DIR, "demo_val.csv")
    demo_test_path = os.path.join(Config.WORKING_DIR, "demo_test.csv")

    # Save demo files
    demo_train.to_csv(demo_train_path, index=False)
    demo_val.to_csv(demo_val_path, index=False)
    demo_test.to_csv(demo_test_path, index=False)

    return demo_train_path, demo_val_path, demo_test_path


def configure_environment(train_path, val_path, test_path):
    """
    Overrides Config parameters to use demo data and fast training settings.
    """
    print("--- Configuring Environment for Fast Execution ---")

    # Override Data Paths
    Config.TRAIN_DATA_PATH = train_path
    Config.VAL_DATA_PATH = val_path
    Config.TEST_DATA_PATH = test_path

    # Override Training Hyperparameters for Speed
    Config.MLM_EPOCHS = 1
    Config.FT_EPOCHS = 1
    Config.MLM_BATCH_SIZE = 4
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 8
    Config.DEBUG = True

    # Use a lighter backbone for the demo if possible, but we stick to
    # roberta-base as it is in the config and relatively standard.
    # We will only process one backbone to save time.
    Config.MODEL_BACKBONES = ["roberta-base"]

    # Ensure cache directory is clean-ish or we force reload
    # We will use load_cached_data=False in function calls

    seed_everything(Config.SEED)


def demo_statistical_model():
    """
    Demonstrates the StatisticalModel class (TF-IDF + Logistic Regression/NB).
    """
    print("\n=== Demonstrating Statistical Model ===")

    model = StatisticalModel()

    # Fit the model (this uses the files pointed to by Config)
    print("Fitting statistical model...")
    val_loss = model.fit(load_cached_data=False)

    print(f"Statistical Model Validation Loss: {val_loss:.4f}")

    # Predict on validation set
    print("Predicting on validation set...")
    preds = model.predict_proba(dataset_type="val", load_cached_data=False)

    # Validation
    assert preds.shape == (50, 3), f"Expected shape (50, 3), got {preds.shape}"
    assert np.allclose(preds.sum(axis=1), 1.0), "Probabilities do not sum to 1"
    print("Statistical Model verification passed.")

    return preds


def demo_neural_pipeline():
    """
    Demonstrates MLM Pretraining, Supervised Training, and Distillation.
    """
    print("\n=== Demonstrating Neural Pipeline ===")

    model_name = "roberta-base"

    # 1. MLM Pre-training
    # This uses the data from Config paths
    print(f"Running MLM Pre-training for {model_name}...")
    mlm_output_dir = train_mlm(model_name, load_cached_data=False)

    assert os.path.exists(
        os.path.join(mlm_output_dir, "config.json")
    ), "MLM model config not saved."
    print("MLM Pre-training verification passed.")

    # 2. Setup for Supervised/Distillation
    print("Initializing Neural Model and Engine...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize custom model architecture
    model = CustomTransformer(model_name, num_classes=Config.NUM_CLASSES)
    model.to(Config.DEVICE)

    engine = DistillationEngine(model, Config.DEVICE, tokenizer)

    # Load Dataframes explicitly for Dataset creation
    train_df, val_df, test_df = load_raw_data()

    # Create Datasets
    train_dataset = AuthorDataset(train_df, tokenizer)
    val_dataset = AuthorDataset(val_df, tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=Config.TRAIN_BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=Config.VALID_BATCH_SIZE)

    # 3. Supervised Training
    print("Running Supervised Training...")
    best_loss = engine.train_supervised(train_loader, val_loader, epochs=1)
    print(f"Supervised Training Best Loss: {best_loss:.4f}")

    # 4. Distillation Training
    print("Running Distillation Training...")
    # Prepare inputs
    train_texts = train_df["text"].tolist()
    train_labels = train_df["author"].map(Config.LABEL2ID).values

    test_texts = test_df["text"].tolist()
    # Generate dummy soft targets for test data (simulating teacher predictions)
    # Shape: (n_samples, n_classes)
    dummy_soft_targets = np.random.dirichlet(np.ones(3), size=len(test_texts))

    distill_loss = engine.train_distilled(
        train_texts, train_labels, test_texts, dummy_soft_targets, val_loader, epochs=1
    )
    print(f"Distillation Training Best Loss: {distill_loss:.4f}")

    # Final Evaluation check
    final_loss, final_preds = engine.evaluate(val_loader)
    assert final_preds.shape == (50, 3), "Prediction shape mismatch"
    print("Neural Pipeline verification passed.")


def demo_ensemble_optimizer():
    """
    Demonstrates the EnsembleOptimizer using synthetic OOF predictions.
    """
    print("\n=== Demonstrating Ensemble Optimizer ===")

    # Generate synthetic ground truth
    n_samples = 100
    y_true = np.random.randint(0, 3, size=n_samples)

    # Generate synthetic predictions for 2 models
    # Model A: Random but slightly informed
    probs_a = np.random.rand(n_samples, 3)
    probs_a = probs_a / probs_a.sum(axis=1, keepdims=True)

    # Model B: Another set of random probs
    probs_b = np.random.rand(n_samples, 3)
    probs_b = probs_b / probs_b.sum(axis=1, keepdims=True)

    oof_preds = {"model_a": probs_a, "model_b": probs_b}

    optimizer = EnsembleOptimizer()

    # Optimize
    weights = optimizer.optimize_weights(oof_preds, y_true)

    # Verify weights sum to 1
    weight_sum = sum(weights.values())
    assert abs(weight_sum - 1.0) < 1e-5, f"Weights sum to {weight_sum}, expected 1.0"

    # Blend
    blended = optimizer.blend_predictions(oof_preds)

    assert blended.shape == (n_samples, 3), "Blended shape mismatch"
    print("Ensemble Optimizer verification passed.")


if __name__ == "__main__":
    # 1. Setup Data
    train_path, val_path, test_path = create_demo_data()

    # 2. Configure
    configure_environment(train_path, val_path, test_path)

    # 3. Run Demonstrations
    try:
        demo_statistical_model()
        demo_neural_pipeline()
        demo_ensemble_optimizer()

        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nAn error occurred during demonstration: {e}")
        raise e
    finally:
        # Cleanup temporary files if needed, though usually kept for inspection in 'working'
        pass
