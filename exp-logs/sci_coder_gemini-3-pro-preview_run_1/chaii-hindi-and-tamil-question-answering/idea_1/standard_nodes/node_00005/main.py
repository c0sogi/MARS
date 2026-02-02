import pandas as pd
import torch
import numpy as np
import os
from library.config import Config
from library.dataset import get_dataloaders
from library.model import load_model
from library.trainer import Trainer
from library.inference import generate_predictions
from library.utils import jaccard, find_best_substring


def perform_failure_analysis(model, val_loader, tokenizer, device):
    """
    Performs failure analysis on the validation set.
    Calculates correlations between error magnitude and input features.
    """
    print("\nRunning Failure Analysis on Validation Set...")
    model.eval()

    analysis_data = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            contexts = batch["context"]
            questions = batch["question"]
            ground_truths = batch["answer_text"]

            # Generate predictions
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=Config.MAX_TARGET_LENGTH,
            )

            decoded_preds = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            for i, pred_text in enumerate(decoded_preds):
                context = contexts[i]
                question = questions[i]
                gt_text = ground_truths[i]

                # Post-processing
                final_pred = find_best_substring(context, pred_text)

                # Metric calculation
                score = jaccard(gt_text, final_pred)
                error_magnitude = 1.0 - score

                # Feature extraction
                # Simple whitespace splitting for rough word count
                context_len = len(context.split())
                question_len = len(question.split())

                analysis_data.append(
                    {
                        "context_len": context_len,
                        "question_len": question_len,
                        "jaccard": score,
                        "error": error_magnitude,
                    }
                )

    # Create DataFrame
    df_analysis = pd.DataFrame(analysis_data)

    # Calculate correlations
    if not df_analysis.empty:
        correlations = df_analysis[["error", "context_len", "question_len"]].corr()[
            "error"
        ]

        print("-" * 40)
        print("Correlation with Error Magnitude (1 - Jaccard):")
        print(f"Context Length (words):  {correlations['context_len']:.4f}")
        print(f"Question Length (words): {correlations['question_len']:.4f}")
        print("-" * 40)
    else:
        print("No analysis data collected.")


def main():
    # 1. Setup
    Config.setup()

    print("Configuration set. Starting pipeline...")

    # 2. Data Loading
    # Load cached data if available to save time
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(
        load_cached_data=True
    )

    # 3. Model Initialization
    model = load_model()

    # 4. Training
    trainer = Trainer(model, tokenizer)
    trainer.fit(train_loader, val_loader)

    # 5. Validation Assessment
    # Reload the best model saved during training for accurate evaluation
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print("Reloading best model for final validation...")
        model = load_model(Config.MODEL_SAVE_PATH)
        trainer.model = model  # Update trainer's model reference

    # Compute final metric
    val_score = trainer.evaluate(val_loader)
    print(f"Final Validation Metric: {val_score}")

    # 6. Failure Analysis
    perform_failure_analysis(model, val_loader, tokenizer, Config.DEVICE)

    # 7. Submission Generation
    # This function handles loading the best model and saving to CSV
    generate_predictions(load_cached_data=True)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
