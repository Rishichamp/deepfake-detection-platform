# -*- coding: utf-8 -*-
"""CNN-LSTM temporal model for video-level deepfake detection.

Architecture Overview
---------------------
Many deepfake artifacts are *temporally* inconsistent – flickering edges,
jittery landmarks, unnatural blinking.  A single-frame classifier cannot
capture these cues, so we pair a frozen (or fine-tunable) CNN backbone with
a bidirectional LSTM that reasons over a sequence of frames.

Pipeline:
    1. **Frame feature extraction** – A ResNet-18 backbone (via ``timm``)
       maps each frame to a 512-d feature vector.
    2. **Temporal modelling** – A 2-layer bidirectional LSTM processes the
       sequence, producing hidden states at every timestep.
    3. **Temporal attention** – A lightweight attention mechanism learns
       *which frames* are most discriminative, producing a single
       context vector.
    4. **Classification** – A linear head maps the context vector to
       binary logits.

Input:  (B, T, 3, 224, 224)  –  B video clips of T RGB frames each
Output: (B, num_classes)      –  logits (default num_classes=2: real / fake)

References
----------
* Güera & Delp, "Deepfake Video Detection Using Recurrent Neural Networks",
  AVSS 2018.
* Sabir et al., "Recurrent Convolutional Strategies for Face Manipulation
  Detection in Videos", 2019.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CNNLSTMDetector", "create_cnn_lstm"]


class TemporalAttention(nn.Module):
    """Additive (Bahdanau-style) attention over LSTM hidden states.

    Given a sequence of hidden vectors ``H = (h_1, …, h_T)`` the module
    computes:

        e_t   = v^T · tanh(W · h_t + b)
        α     = softmax(e)
        ctx   = Σ α_t · h_t

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of each hidden vector.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attention_fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-weighted context vector.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Shape ``(B, T, hidden_dim)`` – LSTM outputs at each timestep.

        Returns
        -------
        context : torch.Tensor
            Attention-weighted summary, shape ``(B, hidden_dim)``.
        weights : torch.Tensor
            Normalised attention weights, shape ``(B, T)``.
        """
        # (B, T, hidden_dim) → (B, T, hidden_dim) → (B, T, 1)
        energy = torch.tanh(self.attention_fc(hidden_states))
        scores = self.context_vector(energy).squeeze(-1)   # (B, T)
        weights = F.softmax(scores, dim=-1)                # (B, T)

        # Weighted sum: (B, 1, T) × (B, T, hidden_dim) → (B, hidden_dim)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, weights


class CNNLSTMDetector(nn.Module):
    """CNN + Bidirectional LSTM with temporal attention for deepfake detection.

    Parameters
    ----------
    pretrained : bool
        Load ImageNet-pretrained weights for the CNN backbone.
    num_classes : int
        Number of output classes (default ``2``).
    seq_length : int
        Expected number of frames per clip (for documentation; the model
        handles variable lengths at runtime).
    lstm_hidden : int
        Hidden size of each LSTM direction (total = ``2 × lstm_hidden``).
    lstm_layers : int
        Number of stacked LSTM layers.
    dropout_rate : float
        Dropout applied between LSTM layers and before the classifier.
    freeze_backbone : bool
        If ``True``, CNN backbone weights are frozen (faster training when
        the backbone is already well-suited to face features).
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 2,
        seq_length: int = 16,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.seq_length = seq_length

        # ------------------------------------------------------------------
        # 1. CNN backbone – ResNet-18 feature extractor
        # ------------------------------------------------------------------
        self.cnn = timm.create_model(
            "resnet18",
            pretrained=pretrained,
            num_classes=0,      # strip classifier → returns pooled features
            global_pool="avg",
        )
        self.cnn_feature_dim: int = 512  # ResNet-18 final pool output

        if freeze_backbone:
            for param in self.cnn.parameters():
                param.requires_grad = False

        # ------------------------------------------------------------------
        # 2. Temporal LSTM
        # ------------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=self.cnn_feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if lstm_layers > 1 else 0.0,
        )
        self.lstm_output_dim: int = lstm_hidden * 2  # bidirectional

        # ------------------------------------------------------------------
        # 3. Temporal attention
        # ------------------------------------------------------------------
        self.attention = TemporalAttention(self.lstm_output_dim)

        # ------------------------------------------------------------------
        # 4. Classifier
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.lstm_output_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.lstm_output_dim, num_classes),
        )

        # Store the last attention weights for visualisation
        self.last_attention_weights: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass on a batch of video clips.

        Parameters
        ----------
        x : torch.Tensor
            Batch of clips, shape ``(B, T, 3, 224, 224)`` where ``T`` is the
            number of frames.

        Returns
        -------
        torch.Tensor
            Class logits, shape ``(B, num_classes)``.
        """
        B, T, C, H, W = x.shape

        # Flatten batch and time dims so CNN processes all frames at once
        frames = x.reshape(B * T, C, H, W)             # (B*T, 3, 224, 224)
        frame_features = self.cnn(frames)               # (B*T, 512)
        frame_features = frame_features.reshape(B, T, -1)  # (B, T, 512)

        # LSTM over the temporal sequence
        lstm_out, _ = self.lstm(frame_features)         # (B, T, 512)

        # Attention pooling over timesteps
        context, attn_weights = self.attention(lstm_out)  # (B, 512), (B, T)
        self.last_attention_weights = attn_weights.detach()

        # Classification
        logits = self.classifier(context)               # (B, num_classes)
        return logits

    def extract_temporal_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the attention-pooled temporal feature vector.

        Parameters
        ----------
        x : torch.Tensor
            Batch of clips, shape ``(B, T, 3, 224, 224)``.

        Returns
        -------
        torch.Tensor
            Temporal context vector, shape ``(B, lstm_output_dim)``.
        """
        B, T, C, H, W = x.shape
        frames = x.reshape(B * T, C, H, W)
        frame_features = self.cnn(frames).reshape(B, T, -1)
        lstm_out, _ = self.lstm(frame_features)
        context, _ = self.attention(lstm_out)
        return context


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------
def create_cnn_lstm(
    pretrained: bool = True,
    num_classes: int = 2,
    seq_length: int = 16,
    lstm_hidden: int = 256,
    lstm_layers: int = 2,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = False,
) -> CNNLSTMDetector:
    """Instantiate a CNN-LSTM temporal deepfake detector.

    Parameters
    ----------
    pretrained : bool
        Load ImageNet-pretrained backbone weights.
    num_classes : int
        Number of output classes (``2`` for real / fake).
    seq_length : int
        Default number of frames per clip (the model handles variable
        lengths at runtime).
    lstm_hidden : int
        Hidden size per LSTM direction.
    lstm_layers : int
        Number of stacked LSTM layers.
    dropout_rate : float
        Dropout probability.
    freeze_backbone : bool
        Freeze CNN backbone weights.

    Returns
    -------
    CNNLSTMDetector
        Ready-to-use model.

    Example
    -------
    >>> model = create_cnn_lstm(pretrained=True, num_classes=2, seq_length=16)
    >>> dummy = torch.randn(2, 16, 3, 224, 224)
    >>> logits = model(dummy)
    >>> logits.shape
    torch.Size([2, 2])
    """
    return CNNLSTMDetector(
        pretrained=pretrained,
        num_classes=num_classes,
        seq_length=seq_length,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
    )
