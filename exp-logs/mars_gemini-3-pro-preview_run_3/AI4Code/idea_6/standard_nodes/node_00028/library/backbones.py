import os
import math
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from library.config import Config
from library.utils import set_seed
from library.dataset import create_relaxed_proximity_pairs


class BackboneFineTuner:
    """
    Manages the fine-tuning of sentence-transformer models (backbones)
    using Contrastive Learning (Multiple Negatives Ranking Loss).
    """

    def __init__(self, model_name, output_path):
        """
        Initialize the fine-tuner.

        Args:
            model_name (str): The HuggingFace model identifier (e.g., 'microsoft/codebert-base').
            output_path (str): Directory where the fine-tuned model will be saved.
        """
        self.model_name = model_name
        self.output_path = output_path
        self.device = Config.DEVICE
        set_seed(Config.SEED)

    def _load_model(self):
        """
        Loads the model. If it's a raw transformer like CodeBERT, ensures
        it is wrapped correctly with a pooling layer for SentenceTransformer compatibility.
        """
        print(f"Initializing backbone: {self.model_name}")

        # Check if it's a pre-existing SentenceTransformer model or a raw HF model
        # sentence-transformers handles most HF models automatically, but for CodeBERT
        # we ensure a dense pooling strategy if needed.
        try:
            model = SentenceTransformer(self.model_name, device=self.device)
        except Exception:
            # Fallback for models that might need explicit module definition
            word_embedding_model = models.Transformer(
                self.model_name, max_seq_length=Config.MAX_LEN
            )
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension()
            )
            model = SentenceTransformer(
                modules=[word_embedding_model, pooling_model], device=self.device
            )

        return model

    def train(self, debug=False, load_cached_data=True, force_retrain=False):
        """
        Executes the fine-tuning pipeline.

        Args:
            debug (bool): If True, uses a smaller subset of data for quick testing.
            load_cached_data (bool): Whether to use cached training pairs.
            force_retrain (bool): If True, ignores existing saved model and retrains.
        """
        # 1. Caching Check: If model exists and not forcing retrain, skip.
        if os.path.exists(self.output_path) and not force_retrain:
            # Check if essential files exist to confirm it's a valid save
            if os.path.exists(os.path.join(self.output_path, "config.json")):
                print(
                    f"Fine-tuned model found at {self.output_path}. Skipping training."
                )
                return

        print(f"Starting fine-tuning for {self.model_name}...")

        # 2. Load Data
        # We use the 'train' split for fine-tuning
        df_pairs = create_relaxed_proximity_pairs(
            Config.TRAIN_PATH,
            mode="train",
            debug=debug,
            load_cached_data=load_cached_data,
        )

        if len(df_pairs) == 0:
            print("No training pairs found. Aborting training.")
            return

        # 3. Prepare InputExamples
        # Each example is a pair: (Markdown Text, Code Text)
        train_examples = []
        for _, row in df_pairs.iterrows():
            train_examples.append(
                InputExample(texts=[str(row["markdown_text"]), str(row["code_text"])])
            )

        # 4. Prepare DataLoader
        train_dataloader = DataLoader(
            train_examples, shuffle=True, batch_size=Config.BATCH_SIZE
        )

        # 5. Initialize Model
        model = self._load_model()

        # 6. Define Loss
        # MultipleNegativesRankingLoss is effective for (anchor, positive) pairs
        # It treats other samples in the batch as negatives.
        train_loss = losses.MultipleNegativesRankingLoss(model)

        # 7. Training Configuration
        # Calculate warmup steps
        num_steps_per_epoch = len(train_dataloader)
        total_steps = num_steps_per_epoch * Config.NUM_EPOCHS
        warmup_steps = min(Config.WARMUP_STEPS, int(0.1 * total_steps))

        print(
            f"Training on {len(train_examples)} pairs for {Config.NUM_EPOCHS} epochs."
        )
        print(
            f"Batch size: {Config.BATCH_SIZE}, Total steps: {total_steps}, Warmup steps: {warmup_steps}"
        )

        # 8. Run Training
        # We disable the progress bar to comply with requirements
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=Config.NUM_EPOCHS,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": Config.LEARNING_RATE},
            weight_decay=Config.WEIGHT_DECAY,
            output_path=self.output_path,
            show_progress_bar=False,
            use_amp=True,  # Use Automatic Mixed Precision if available
        )

        print(f"Training complete. Model saved to {self.output_path}")
