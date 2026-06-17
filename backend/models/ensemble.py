# -*- coding: utf-8 -*-
"""Weighted ensemble fusion for deepfake detection.

Architecture Overview
---------------------
The ensemble module fuses predictions from three heterogeneous detectors:

    1. **EfficientNet-B4** – strong single-frame spatial features.
    2. **ViT-Base/16** – global self-attention over spatial patches.
    3. **CNN-LSTM** – temporal consistency across video frames.

Each model produces class logits independently.  This module learns a
*softmax-normalised* weighting over the three probability distributions,
producing a calibrated final prediction.

**Confidence calibration** is handled by a lightweight ``TemperatureScaler``
that adjusts the sharpness of the fused distribution.  The temperature
parameter is learnt on a held-out calibration set *after* the individual
models have been trained (see Guo et al., 2017).

Input:  Three tensors of logits, each ``(B, num_classes)``
Output: ``(B, num_classes)`` – calibrated fused probabilities (or logits)

References
----------
* Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "WeightedEnsemble",
    "TemperatureScaler",
    "create_ensemble",
]


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling for confidence calibration.

    A single learnable scalar ``T`` divides the logits before softmax,
    controlling the peakiness of the output distribution without
    changing the predicted class.

    Parameters
    ----------
    init_temperature : float
        Initial temperature value (``1.0`` = no scaling).
    """

    def __init__(self, init_temperature: float = 1.0) -> None:
        super().__init__()
        # Store as log so the effective temperature is always positive
        self.log_temperature = nn.Parameter(
            torch.tensor(init_temperature).log()
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Current (positive) temperature value."""
        return self.log_temperature.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by the learned temperature.

        Parameters
        ----------
        logits : torch.Tensor
            Raw logits, shape ``(B, num_classes)``.

        Returns
        -------
        torch.Tensor
            Temperature-scaled logits, same shape.
        """
        return logits / self.temperature


class WeightedEnsemble(nn.Module):
    """Learnable weighted fusion of multiple model predictions.

    The ensemble converts each model's logits to probabilities via
    softmax, computes a weighted average (with softmax-normalised
    fusion weights), and optionally applies temperature scaling for
    calibration.

    Parameters
    ----------
    num_models : int
        Number of constituent models to fuse (default ``3``).
    num_classes : int
        Number of output classes (default ``2``).
    calibrate : bool
        If ``True``, apply learnable temperature scaling to the fused logits.

    Attributes
    ----------
    raw_weights : nn.Parameter
        Unnormalised fusion weights, shape ``(num_models,)``.
    temperature_scaler : TemperatureScaler or nn.Identity
        Post-hoc calibration module.
    """

    def __init__(
        self,
        num_models: int = 3,
        num_classes: int = 2,
        calibrate: bool = True,
    ) -> None:
        super().__init__()
        self.num_models = num_models
        self.num_classes = num_classes

        # Learnable fusion weights (initialised to equal weighting)
        self.raw_weights = nn.Parameter(torch.ones(num_models))

        # Confidence calibration
        self.temperature_scaler: nn.Module = (
            TemperatureScaler() if calibrate else nn.Identity()
        )

    @property
    def fusion_weights(self) -> torch.Tensor:
        """Softmax-normalised fusion weights (always sum to 1)."""
        return F.softmax(self.raw_weights, dim=0)

    def forward(
        self,
        logits_efficientnet: torch.Tensor,
        logits_vit: torch.Tensor,
        logits_cnn_lstm: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Fuse predictions from all three models.

        Parameters
        ----------
        logits_efficientnet : torch.Tensor
            EfficientNet logits, shape ``(B, num_classes)``.
        logits_vit : torch.Tensor
            ViT logits, shape ``(B, num_classes)``.
        logits_cnn_lstm : torch.Tensor
            CNN-LSTM logits, shape ``(B, num_classes)``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``"logits"``
                Fused logits (temperature-scaled), shape ``(B, num_classes)``.
            ``"probabilities"``
                Fused probabilities, shape ``(B, num_classes)``.
            ``"confidence"``
                Maximum class probability per sample, shape ``(B,)``.
            ``"predicted_class"``
                Argmax class index per sample, shape ``(B,)``.
            ``"fusion_weights"``
                Current normalised fusion weights, shape ``(num_models,)``.
            ``"individual_probs"``
                Stacked per-model probabilities, shape ``(B, num_models, num_classes)``.
        """
        # Stack logits: (B, num_models, num_classes)
        all_logits = torch.stack(
            [logits_efficientnet, logits_vit, logits_cnn_lstm], dim=1
        )

        # Convert each model's logits to probabilities independently
        all_probs = F.softmax(all_logits, dim=-1)  # (B, num_models, num_classes)

        # Normalised fusion weights: (num_models,) → (1, num_models, 1)
        weights = self.fusion_weights.view(1, self.num_models, 1)

        # Weighted average of probabilities
        fused_probs = (all_probs * weights).sum(dim=1)  # (B, num_classes)

        # Convert back to logit space for temperature scaling
        # (clamp to avoid log(0))
        fused_logits = torch.log(fused_probs.clamp(min=1e-7))
        fused_logits = self.temperature_scaler(fused_logits)

        # Final calibrated probabilities
        calibrated_probs = F.softmax(fused_logits, dim=-1)

        confidence, predicted_class = calibrated_probs.max(dim=-1)

        return {
            "logits": fused_logits,
            "probabilities": calibrated_probs,
            "confidence": confidence,
            "predicted_class": predicted_class,
            "fusion_weights": self.fusion_weights.detach(),
            "individual_probs": all_probs.detach(),
        }

    def extra_repr(self) -> str:
        w = self.fusion_weights.detach().cpu().tolist()
        labels = ["EfficientNet", "ViT", "CNN-LSTM"]
        parts = [f"{l}: {v:.3f}" for l, v in zip(labels, w)]
        return "fusion_weights=(" + ", ".join(parts) + ")"


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------
def create_ensemble(
    num_models: int = 3,
    num_classes: int = 2,
    calibrate: bool = True,
) -> WeightedEnsemble:
    """Instantiate a weighted ensemble fusion module.

    Parameters
    ----------
    num_models : int
        Number of base models whose predictions will be fused.
    num_classes : int
        Number of output classes.
    calibrate : bool
        Enable learnable temperature scaling.

    Returns
    -------
    WeightedEnsemble
        Ready-to-use fusion module.

    Example
    -------
    >>> ensemble = create_ensemble()
    >>> eff_logits = torch.randn(4, 2)
    >>> vit_logits = torch.randn(4, 2)
    >>> lstm_logits = torch.randn(4, 2)
    >>> out = ensemble(eff_logits, vit_logits, lstm_logits)
    >>> out["probabilities"].shape
    torch.Size([4, 2])
    >>> out["confidence"].shape
    torch.Size([4,])
    """
    return WeightedEnsemble(
        num_models=num_models,
        num_classes=num_classes,
        calibrate=calibrate,
    )
