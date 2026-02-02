import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import config
from library.data_utils import build_vocab, load_embeddings
from library.dataset import NQDataset
from library.model_components import (
    CosineInteraction,
    RBFKernelLayer,
    DepthwiseSeparableConv1D,
)
from library.model import KernelPoolingNetwork
from library.loss import MultiTaskLoss
from library.trainer import Trainer
from library.inference import run_inference


def set_reproducibility(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def demo_config_override():
    """
    Override config parameters to ensure the demo runs quickly and uses less memory.
    """
    print("--- 1. Configuring Hyperparameters for Demo ---")
    config.VOCAB_SIZE = 2000  # Small vocab
    config.EMBEDDING_DIM = 64  # Small embeddings
    config.HIDDEN_DIM = 32
    config.CONV_FILTERS = 32
    config.BATCH_SIZE = 4
    config.EPOCHS = 1
    config.NEGATIVE_SAMPLES_RATIO = 1  # Reduce negative sampling

    # Ensure working directory is clean for a fresh run
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    print("Configuration updated for speed.")


def demo_data_utils_and_dataset():
    """
    Demonstrate Vocabulary building and Dataset loading.
    """
    print("\n--- 2. Data Utils and Dataset Verification ---")

    # Build Vocabulary
    vocab = build_vocab(load_cached_data=False)
    assert len(vocab) <= config.VOCAB_SIZE, "Vocab size exceeded limit"
    print(f"Vocabulary built. Size: {len(vocab)}")

    # Initialize Dataset (Train) with limit
    limit = 20
    train_ds = NQDataset(
        split="train", vocab=vocab, load_cached_data=False, limit_size=limit
    )

    assert len(train_ds) > 0, "Dataset is empty"

    # Check a single sample
    sample = train_ds[0]
    print("Sample keys:", sample.keys())

    # Verify shapes
    q_shape = sample["question"].shape
    c_shape = sample["candidate"].shape
    assert q_shape == (
        config.MAX_QUESTION_LENGTH,
    ), f"Question shape mismatch: {q_shape}"
    assert c_shape == (
        config.MAX_CANDIDATE_LENGTH,
    ), f"Candidate shape mismatch: {c_shape}"

    # Verify labels
    assert isinstance(sample["label_long"], torch.Tensor)
    assert isinstance(sample["label_span_start"], torch.Tensor)
    print("Dataset verification successful.")
    return vocab


def demo_model_components():
    """
    Demonstrate and verify individual model layers.
    """
    print("\n--- 3. Model Component Verification ---")

    batch_size = 2
    q_len = 10
    c_len = 50
    emb_dim = config.EMBEDDING_DIM

    # 1. Cosine Interaction
    interaction_layer = CosineInteraction()
    q_emb = torch.randn(batch_size, q_len, emb_dim)
    c_emb = torch.randn(batch_size, c_len, emb_dim)
    interaction_matrix = interaction_layer(q_emb, c_emb)

    assert interaction_matrix.shape == (batch_size, q_len, c_len)
    # Check range [-1, 1] (approximately due to float precision)
    assert interaction_matrix.max() <= 1.0001 and interaction_matrix.min() >= -1.0001
    print("CosineInteraction passed.")

    # 2. RBF Kernel Layer
    rbf_layer = RBFKernelLayer(means=config.KERNEL_MEANS, sigmas=config.KERNEL_SIGMAS)
    log_pooled = rbf_layer(interaction_matrix)

    # Output shape should be [Batch, Q_Len, Num_Kernels]
    assert log_pooled.shape == (batch_size, q_len, config.NUM_KERNELS)
    print("RBFKernelLayer passed.")

    # 3. Depthwise Separable Conv
    in_ch = emb_dim
    out_ch = config.HIDDEN_DIM
    conv_layer = DepthwiseSeparableConv1D(in_ch, out_ch, kernel_size=3, padding=1)

    # Input to conv: [Batch, Channels, Length]
    conv_input = torch.randn(batch_size, in_ch, c_len)
    conv_output = conv_layer(conv_input)

    assert conv_output.shape == (batch_size, out_ch, c_len)
    print("DepthwiseSeparableConv1D passed.")


def demo_full_model_and_loss(vocab):
    """
    Demonstrate full model forward pass and loss calculation.
    """
    print("\n--- 4. Full Model and Loss Verification ---")

    model = KernelPoolingNetwork(vocab)
    criterion = MultiTaskLoss()

    # Create dummy batch
    batch_size = config.BATCH_SIZE
    q_input = torch.randint(0, len(vocab), (batch_size, config.MAX_QUESTION_LENGTH))
    c_input = torch.randint(0, len(vocab), (batch_size, config.MAX_CANDIDATE_LENGTH))

    # Forward pass
    outputs = model(q_input, c_input)

    # Verify outputs
    assert outputs["long_score"].shape == (batch_size, 1)
    assert outputs["start_logits"].shape == (batch_size, config.MAX_CANDIDATE_LENGTH)
    assert outputs["yesno_logits"].shape == (batch_size, config.NUM_YES_NO_CLASSES)
    print("Model forward pass successful.")

    # Create dummy targets
    targets = {
        "label_long": torch.zeros(batch_size).float(),
        "label_span_start": torch.zeros(batch_size).long(),
        "label_span_end": torch.zeros(batch_size).long(),
        "label_yesno": torch.zeros(batch_size).long(),
    }

    # Loss calculation
    loss, metrics = criterion(outputs, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0
    print(f"Loss calculation successful. Total Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")


def demo_training_loop():
    """
    Demonstrate the Trainer class running a short training epoch.
    """
    print("\n--- 5. Training Loop Demonstration ---")

    # Initialize Trainer with limited data size
    # limit_size=100 ensures we only process a tiny fraction of data
    trainer = Trainer(load_cached_data=True, limit_size=100)

    # Run training (Config is set to 1 epoch)
    trainer.train()

    # Check if model checkpoint was created
    if os.path.exists(config.MODEL_CHECKPOINT_PATH):
        print(f"Training completed. Checkpoint saved at {config.MODEL_CHECKPOINT_PATH}")
    else:
        # It might not save if validation loss doesn't improve or logic skips,
        # but for 1 epoch it usually saves if it's the best so far (init infinity).
        # However, Trainer logic saves if val_loss < best_val_loss.
        print("Training completed (no checkpoint saved or logic skipped).")


def demo_inference():
    """
    Demonstrate the inference pipeline.
    """
    print("\n--- 6. Inference Demonstration ---")

    # Run inference with limited size
    run_inference(load_cached_data=True, limit_size=50, batch_size=4)

    # Verify submission file
    if os.path.exists(config.SUBMISSION_PATH):
        df = pd.read_csv(config.SUBMISSION_PATH)
        print(f"Submission file generated. Shape: {df.shape}")
        print("First 5 rows:")
        print(df.head())
    else:
        raise FileNotFoundError("Submission file was not generated.")


if __name__ == "__main__":
    set_reproducibility()

    # 1. Override Config for Speed
    demo_config_override()

    # 2. Verify Data Utils & Dataset
    vocab = demo_data_utils_and_dataset()

    # 3. Verify Model Components
    demo_model_components()

    # 4. Verify Full Model & Loss
    demo_full_model_and_loss(vocab)

    # 5. Run Training Demo
    demo_training_loop()

    # 6. Run Inference Demo
    demo_inference()

    print("\n=== All Demonstrations Completed Successfully ===")
