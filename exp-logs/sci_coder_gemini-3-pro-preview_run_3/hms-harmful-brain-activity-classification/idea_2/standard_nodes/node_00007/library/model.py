import torch
import torch.nn as nn
from library import config


class EEGWaveNet(nn.Module):
    """
    1D-CNN + Bi-GRU architecture for raw EEG signal classification.
    Cite solution_lesson_node_00001: Preceding RNN with 1D-CNN to extract features.
    Cite solution_lesson_node_00006: Using raw signals to preserve spatial integrity.
    """

    def __init__(self, pretrained=False):
        super(EEGWaveNet, self).__init__()

        # Feature Extractor (1D CNN)
        self.features = nn.Sequential(
            nn.Conv1d(config.IN_CHANNELS, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.MaxPool1d(2),
        )

        # Bi-Directional GRU
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        # Classifier Head
        self.fc = nn.Linear(256 * 2, config.NUM_CLASSES)

    def forward(self, x):
        # x shape: (Batch, 19, 2500)
        x = self.features(x)  # -> (Batch, 512, ~156)

        # Permute for GRU: (Batch, Seq, Channels)
        x = x.permute(0, 2, 1)

        # GRU
        out, _ = self.gru(x)  # -> (Batch, Seq, Hidden*2)

        # Global Average Pooling over time dimension
        out = torch.mean(out, dim=1)

        return self.fc(out)
