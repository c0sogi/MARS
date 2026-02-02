import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from sklearn.metrics import f1_score
from library.config import SUBMISSION_DIR, LEARNING_RATE, NUM_EPOCHS, WORKING_DIR
from library.model import PlantClassifier
from library.utils import get_device


class SupervisedTrainer:
    """
    Trainer for supervised fine-tuning of the plant classifier.
    """

    def __init__(self):
        self.device = get_device()
        self.model = PlantClassifier().to(self.device)
        self.best_score = 0.0
        self.best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    def fit(self, train_loader, val_loader):
        """
        Trains the model using supervised learning.
        """
        print(f"Starting training for {NUM_EPOCHS} epochs...")

        optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(NUM_EPOCHS):
            self.model.train()
            train_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

            # Validation step
            val_score = self.evaluate(val_loader, verbose=False)
            print(
                f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {train_loss/len(train_loader):.4f} - Val F1: {val_score:.4f}"
            )

            if val_score > self.best_score:
                self.best_score = val_score
                torch.save(self.model.state_dict(), self.best_model_path)

        # Load best model for final usage
        if os.path.exists(self.best_model_path):
            print(f"Loading best model with F1: {self.best_score:.4f}")
            self.model.load_state_dict(torch.load(self.best_model_path))

    def predict(self, loader, is_test=False):
        self.model.eval()
        all_preds = []
        all_aux = []

        with torch.no_grad():
            for batch in loader:
                images = batch[0].to(self.device)
                aux = batch[1]

                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().tolist())

                if torch.is_tensor(aux):
                    all_aux.extend(aux.tolist())
                else:
                    all_aux.extend(aux)

        return all_preds, all_aux

    def evaluate(self, val_loader, verbose=True):
        if verbose:
            print("Evaluating on validation set...")
        preds, labels = self.predict(val_loader, is_test=False)
        score = f1_score(labels, preds, average="macro")
        if verbose:
            print(f"Validation Macro F1 Score: {score}")
        return score

    def generate_submission(self, test_loader):
        print("Generating submission...")
        preds, image_ids = self.predict(test_loader, is_test=True)

        submission_df = pd.DataFrame({"Id": image_ids, "Predicted": preds})
        submission_df["Id"] = submission_df["Id"].astype(int)

        output_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
