import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample, evaluation
from library.config import config
from library.utils import set_seed
from library.dataset import load_contrastive_pairs


class FineTuner:
    """
    Handles the Domain-Adaptive Contrastive Fine-Tuning of the semantic backbone.
    Uses Siamese Networks with Multiple Negatives Ranking Loss to align
    markdown and code cells in the vector space.
    """

    def __init__(self):
        """
        Initializes the FineTuner with the base SentenceTransformer model.
        """
        self.model_name = config.BASE_MODEL_NAME
        self.model = SentenceTransformer(self.model_name)
        self.model.max_seq_length = config.MAX_LENGTH
        self.device = config.DEVICE
        self.model.to(self.device)

    def train(self, load_cached_data=True):
        """
        Executes the fine-tuning pipeline.

        Args:
            load_cached_data (bool): Whether to load training pairs from cache.
        """
        set_seed(config.SEED)

        print(f"Loading contrastive pairs using base model: {self.model_name}")
        # Load (Markdown, Code) pairs using the library function
        # This handles caching and debug sampling
        df_pairs = load_contrastive_pairs(
            config.TRAIN_METADATA_PATH, "train_pairs", load_cached_data=load_cached_data
        )

        if len(df_pairs) == 0:
            print("No pairs found. Skipping training.")
            return

        # Split into training and validation sets
        train_df, val_df = train_test_split(
            df_pairs, test_size=0.1, random_state=config.SEED, shuffle=True
        )

        print(f"Training on {len(train_df)} pairs, Validating on {len(val_df)} pairs.")

        # Prepare Training Data
        train_examples = [
            InputExample(texts=[row["markdown"], row["code"]])
            for _, row in train_df.iterrows()
        ]
        train_dataloader = DataLoader(
            train_examples, shuffle=True, batch_size=config.TRAIN_BATCH_SIZE
        )

        # Define Loss: MultipleNegativesRankingLoss
        # This loss expects (anchor, positive) pairs and uses in-batch negatives.
        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        # Prepare Validation Evaluator
        # We use InformationRetrievalEvaluator to check if the correct code is retrieved for a markdown query.
        # To keep evaluation fast, we sample the validation set if it is too large.
        eval_sample_size = min(len(val_df), 1000)
        val_sample = val_df.sample(n=eval_sample_size, random_state=config.SEED)

        queries = {}
        corpus = {}
        relevant_docs = {}

        # Construct Query/Corpus maps using index as ID
        for idx, row in val_sample.iterrows():
            q_id = str(idx)
            c_id = str(idx)
            queries[q_id] = row["markdown"]
            corpus[c_id] = row["code"]
            relevant_docs[q_id] = {c_id}

        evaluator = evaluation.InformationRetrievalEvaluator(
            queries,
            corpus,
            relevant_docs,
            name="val_evaluator",
            show_progress_bar=False,
            main_score_function="cos_sim",
            score_functions={"cos_sim": True},  # Focus on cosine similarity
        )

        # Calculate evaluation steps
        # Evaluate 5 times per epoch or at least once
        steps_per_epoch = len(train_dataloader)
        eval_steps = max(1, steps_per_epoch // 5)

        warmup_steps = int(steps_per_epoch * config.NUM_EPOCHS * 0.1)

        print(f"Starting training for {config.NUM_EPOCHS} epochs...")
        print(
            f"Steps per epoch: {steps_per_epoch}, Evaluation every {eval_steps} steps."
        )

        # Train the model
        # fit() handles the training loop, optimizer, and saving the best model
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=config.NUM_EPOCHS,
            evaluation_steps=eval_steps,
            warmup_steps=warmup_steps,
            output_path=config.FINE_TUNED_MODEL_PATH,
            save_best_model=True,
            optimizer_params={"lr": config.LEARNING_RATE},
            weight_decay=config.WEIGHT_DECAY,
            use_amp=(self.device == "cuda"),
            show_progress_bar=False,
        )

        print(
            f"Fine-tuning complete. Best model saved to {config.FINE_TUNED_MODEL_PATH}"
        )

        # Reload the best model to ensure self.model is the optimized version
        self.model = SentenceTransformer(config.FINE_TUNED_MODEL_PATH)
        self.model.to(self.device)

    def save_model(self):
        """
        Saves the current model to the configured path.
        """
        print(f"Saving model to {config.FINE_TUNED_MODEL_PATH}...")
        self.model.save(config.FINE_TUNED_MODEL_PATH)
