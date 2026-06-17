# -*- coding: utf-8 -*-
"""Model zoo for the Deepfake Detection Platform.

This package exposes three detection architectures and a fusion module:

====================  ============================================================
Module                Description
====================  ============================================================
``efficientnet``      EfficientNet-B4 single-frame detector with GradCAM support.
``vit``               ViT-Base/16 single-frame detector with attention rollout.
``cnn_lstm``          CNN-LSTM temporal detector (ResNet-18 + BiLSTM + Attention).
``ensemble``          Weighted ensemble fusion with temperature calibration.
====================  ============================================================

Quick Start
-----------
>>> from backend.models import (
...     create_efficientnet,
...     create_vit,
...     create_cnn_lstm,
...     create_ensemble,
... )
>>> eff_model = create_efficientnet(pretrained=True)
>>> vit_model = create_vit(pretrained=True)
>>> lstm_model = create_cnn_lstm(pretrained=True, seq_length=16)
>>> ensemble = create_ensemble()
"""

from __future__ import annotations

# -- EfficientNet ---------------------------------------------------------
from .efficientnet import EfficientNetDetector, create_efficientnet

# -- Vision Transformer ---------------------------------------------------
from .vit import ViTDetector, create_vit

# -- CNN-LSTM (temporal) ---------------------------------------------------
from .cnn_lstm import CNNLSTMDetector, create_cnn_lstm

# -- Ensemble fusion -------------------------------------------------------
from .ensemble import TemperatureScaler, WeightedEnsemble, create_ensemble

__all__ = [
    # Classes
    "EfficientNetDetector",
    "ViTDetector",
    "CNNLSTMDetector",
    "WeightedEnsemble",
    "TemperatureScaler",
    # Factory functions
    "create_efficientnet",
    "create_vit",
    "create_cnn_lstm",
    "create_ensemble",
]
