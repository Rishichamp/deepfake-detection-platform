# -*- coding: utf-8 -*-
"""
Detection engine for the Deepfake Detection Platform.

This module centralizes all "inference" logic behind a single
``DetectionEngine`` class. It tries to load real PyTorch model weights on
startup; if weights are not found (e.g. in a portfolio/demo deployment),
it transparently falls back to a deterministic, filename-seeded simulation
so the API contract — and therefore the frontend — never changes.

Output contract (always the same shape, regardless of mode)
--------------------------------------------------------------
{
    "id": str,
    "filename": str,
    "type": "image" | "video",
    "verdict": "FAKE" | "REAL",
    "confidence": float,          # 0..1, model's confidence in `verdict`
    "is_fake": bool,
    "per_model_scores": {         # 0..1, each model's *fake* probability
        "efficientnet": float,
        "vit": float,
        "cnn_lstm": float | None,
    },
    "frame_analysis": [
        {
            "frame": int,
            "confidence": float,       # 0..1 fake-probability for this frame
            "efficientnet_conf": float,
            "vit_conf": float,
            "lstm_conf": float | None,
        },
        ...
    ],
    "metadata": {
        "processing_time_ms": float,
        "resolution": str | None,
        "fps": float | None,
        "frames_analyzed": int,
        "faces_detected": int,
        "mode": "model" | "simulation",
    }
}
"""

from __future__ import annotations

import logging
import os
import random
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ALLOWED_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS

# Default location for trained weights (override with env vars)
WEIGHTS_DIR = Path(os.environ.get("MODEL_WEIGHTS_DIR", Path(__file__).parent / "weights"))


class DetectionEngine:
    """
    Wraps model loading + inference, with automatic fallback to a
    deterministic simulation when real weights are unavailable.

    Attributes
    ----------
    mode : str
        Either ``"model"`` (real PyTorch inference) or ``"simulation"``.
    device : str
        ``"cuda"`` or ``"cpu"`` (only meaningful in model mode).
    """

    REQUIRED_WEIGHT_FILES = {
        "efficientnet": "efficientnet_b4.pt",
        "vit": "vit_base.pt",
    }
    OPTIONAL_WEIGHT_FILES = {
        "cnn_lstm": "cnn_lstm.pt",
    }

    def __init__(self, weights_dir: Optional[Path] = None, force_simulation: bool = False):
        self.weights_dir = Path(weights_dir) if weights_dir else WEIGHTS_DIR
        self.mode = "simulation"
        self.device = "cpu"
        self.models: dict = {}
        self.preprocessor = None

        if not force_simulation:
            self._try_load_models()

        if self.mode == "simulation":
            logger.warning(
                "DetectionEngine running in SIMULATION mode. "
                "Place trained weights in '%s' to enable real inference. "
                "Expected files: %s",
                self.weights_dir,
                list(self.REQUIRED_WEIGHT_FILES.values()),
            )
        else:
            logger.info("DetectionEngine running in MODEL mode on device=%s", self.device)

    # ------------------------------------------------------------------ #
    #  Model loading
    # ------------------------------------------------------------------ #

    def _try_load_models(self) -> None:
        """Attempt to load real model weights. Falls back silently on any error."""
        try:
            missing = [
                name
                for name, fname in self.REQUIRED_WEIGHT_FILES.items()
                if not (self.weights_dir / fname).is_file()
            ]
            if missing:
                logger.info(
                    "Skipping model load — missing weight files for: %s", missing
                )
                return

            import torch  # local import: torch may be heavy / optional at runtime

            from backend.models import (
                create_cnn_lstm,
                create_efficientnet,
                create_vit,
            )
            from backend.preprocess import VideoPreprocessor

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            eff = create_efficientnet(pretrained=False)
            eff.load_state_dict(
                torch.load(self.weights_dir / "efficientnet_b4.pt", map_location=self.device)
            )
            eff.eval().to(self.device)

            vit = create_vit(pretrained=False)
            vit.load_state_dict(
                torch.load(self.weights_dir / "vit_base.pt", map_location=self.device)
            )
            vit.eval().to(self.device)

            self.models = {"efficientnet": eff, "vit": vit}

            # CNN-LSTM is optional — load it if weights are present, otherwise
            # the ensemble approximates temporal score from eff+vit (see
            # _detect_with_models).
            lstm_path = self.weights_dir / self.OPTIONAL_WEIGHT_FILES["cnn_lstm"]
            if lstm_path.is_file():
                try:
                    lstm = create_cnn_lstm(pretrained=False)
                    lstm.load_state_dict(torch.load(lstm_path, map_location=self.device))
                    lstm.eval().to(self.device)
                    self.models["cnn_lstm"] = lstm
                    logger.info("Loaded CNN-LSTM weights from '%s'.", lstm_path)
                except Exception:
                    logger.exception(
                        "Failed to load CNN-LSTM weights from '%s' — "
                        "temporal score will be approximated from EfficientNet/ViT.",
                        lstm_path,
                    )
            else:
                logger.info(
                    "CNN-LSTM weights not found at '%s' — "
                    "temporal score will be approximated from EfficientNet/ViT.",
                    lstm_path,
                )

            self.preprocessor = VideoPreprocessor()
            self.mode = "model"

        except Exception:
            logger.exception("Failed to load real model weights — falling back to simulation.")
            self.mode = "simulation"
            self.models = {}
            self.preprocessor = None

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def detect(self, file_path: str, original_filename: str) -> dict:
        """
        Run detection on a file already saved to disk.

        Parameters
        ----------
        file_path : str
            Absolute path to the saved upload (used only in model mode).
        original_filename : str
            The user-supplied filename (used for type detection & seeding).

        Returns
        -------
        dict
            Result matching the contract documented in the module docstring.

        Raises
        ------
        ValueError
            If the file extension is not supported.
        """
        ext = Path(original_filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: '{ext}'")

        is_video = ext in VIDEO_EXTENSIONS

        if self.mode == "model":
            try:
                return self._detect_with_models(file_path, original_filename, is_video)
            except Exception:
                logger.exception(
                    "Real-model inference failed for '%s' — falling back to "
                    "simulation for this request only.",
                    original_filename,
                )
                return self._detect_simulated(original_filename, is_video)

        return self._detect_simulated(original_filename, is_video)

    # ------------------------------------------------------------------ #
    #  Real inference path
    # ------------------------------------------------------------------ #

    def _detect_with_models(self, file_path: str, filename: str, is_video: bool) -> dict:
        import torch
        import torch.nn.functional as F
        from torchvision import transforms

        start = time.time()

        preprocess_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        pipeline = self.preprocessor.preprocess_pipeline(file_path)
        frames = pipeline["frames"]
        face_crops = pipeline["face_crops"]
        metadata = pipeline["metadata"]

        if not frames:
            raise ValueError("No frames could be extracted from the uploaded file.")

        per_frame_results = []
        faces_detected_total = 0

        with torch.no_grad():
            for i, crops in enumerate(face_crops):
                if not crops:
                    continue
                crop = crops[0]  # primary face
                faces_detected_total += 1
                tensor = preprocess_tf(crop).unsqueeze(0).to(self.device)

                eff_logits = self.models["efficientnet"](tensor)
                vit_logits = self.models["vit"](tensor)

                eff_fake_prob = F.softmax(eff_logits, dim=-1)[0, 1].item()
                vit_fake_prob = F.softmax(vit_logits, dim=-1)[0, 1].item()

                per_frame_results.append(
                    {
                        "frame": i,
                        "efficientnet_conf": round(eff_fake_prob, 4),
                        "vit_conf": round(vit_fake_prob, 4),
                    }
                )

        if not per_frame_results:
            raise ValueError("No faces detected in any frame; cannot run inference.")

        # Temporal model (only meaningful for video with multiple frames, and
        # only if CNN-LSTM weights were loaded)
        lstm_fake_prob = None
        if is_video and len(face_crops) > 1 and "cnn_lstm" in self.models:
            seq = []
            for crops in face_crops:
                if crops:
                    seq.append(preprocess_tf(crops[0]))
            if len(seq) >= 2:
                clip = torch.stack(seq).unsqueeze(0).to(self.device)  # (1, T, 3, 224, 224)
                lstm_logits = self.models["cnn_lstm"](clip)
                lstm_fake_prob = F.softmax(lstm_logits, dim=-1)[0, 1].item()

        eff_scores = [f["efficientnet_conf"] for f in per_frame_results]
        vit_scores = [f["vit_conf"] for f in per_frame_results]

        eff_avg = sum(eff_scores) / len(eff_scores)
        vit_avg = sum(vit_scores) / len(vit_scores)
        lstm_avg = lstm_fake_prob if lstm_fake_prob is not None else (eff_avg + vit_avg) / 2

        # Simple equal-weight ensemble
        overall_fake_prob = (eff_avg + vit_avg + lstm_avg) / 3
        is_fake = overall_fake_prob >= 0.5
        verdict = "FAKE" if is_fake else "REAL"
        confidence = overall_fake_prob if is_fake else (1 - overall_fake_prob)

        frame_analysis = []
        for f in per_frame_results:
            overall = (f["efficientnet_conf"] + f["vit_conf"]) / 2
            frame_analysis.append(
                {
                    "frame": f["frame"],
                    "confidence": round(overall, 4),
                    "efficientnet_conf": f["efficientnet_conf"],
                    "vit_conf": f["vit_conf"],
                    "lstm_conf": round(lstm_avg, 4) if is_video else None,
                }
            )

        return {
            "id": str(uuid.uuid4()),
            "filename": filename,
            "type": "video" if is_video else "image",
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "is_fake": is_fake,
            "per_model_scores": {
                "efficientnet": round(eff_avg, 4),
                "vit": round(vit_avg, 4),
                "cnn_lstm": round(lstm_avg, 4) if is_video else None,
            },
            "frame_analysis": frame_analysis,
            "metadata": {
                "processing_time_ms": round((time.time() - start) * 1000, 2),
                "resolution": (
                    f"{metadata['frame_dimensions'][0]}x{metadata['frame_dimensions'][1]}"
                    if metadata.get("frame_dimensions")
                    else None
                ),
                "fps": None,
                "frames_analyzed": metadata.get("num_frames", len(frames)),
                "faces_detected": faces_detected_total,
                "mode": "model",
                "cnn_lstm_loaded": "cnn_lstm" in self.models,
            },
        }

    # ------------------------------------------------------------------ #
    #  Simulation path (deterministic, filename-seeded)
    # ------------------------------------------------------------------ #

    def _detect_simulated(self, filename: str, is_video: bool) -> dict:
        start = time.time()
        time.sleep(0.6 if is_video else 0.25)

        rng = random.Random(filename)  # deterministic per filename, isolated from global state

        name_lower = filename.lower()
        if "fake" in name_lower:
            is_fake = True
        elif "real" in name_lower:
            is_fake = False
        else:
            is_fake = rng.random() > 0.5

        verdict = "FAKE" if is_fake else "REAL"

        if is_fake:
            overall = rng.uniform(0.70, 0.99)
            eff = max(0.0, min(1.0, overall + rng.uniform(-0.08, 0.08)))
            vit = max(0.0, min(1.0, overall + rng.uniform(-0.08, 0.08)))
            lstm = max(0.0, min(1.0, overall + rng.uniform(-0.10, 0.05))) if is_video else None
        else:
            overall = rng.uniform(0.01, 0.20)
            eff = max(0.0, min(1.0, overall + rng.uniform(-0.05, 0.05)))
            vit = max(0.0, min(1.0, overall + rng.uniform(-0.05, 0.05)))
            lstm = max(0.0, min(1.0, overall + rng.uniform(-0.05, 0.05))) if is_video else None

        confidence = overall if is_fake else (1 - overall)

        num_frames = rng.randint(30, 90) if is_video else 1
        frame_analysis = []
        base = overall

        for i in range(num_frames):
            noise = rng.uniform(-0.08, 0.08)
            spike_chance = rng.random()

            if is_fake and spike_chance > 0.85:
                val = min(0.99, base + rng.uniform(0.05, 0.15))
            elif not is_fake and spike_chance > 0.92:
                val = min(0.35, base + rng.uniform(0.05, 0.15))
            else:
                val = max(0.01, min(0.99, base + noise))

            frame_analysis.append(
                {
                    "frame": i,
                    "confidence": round(val, 4),
                    "efficientnet_conf": round(
                        max(0.0, min(1.0, val + rng.uniform(-0.05, 0.05))), 4
                    ),
                    "vit_conf": round(
                        max(0.0, min(1.0, val + rng.uniform(-0.03, 0.03))), 4
                    ),
                    "lstm_conf": (
                        round(max(0.0, min(1.0, val + rng.uniform(-0.08, 0.08))), 4)
                        if is_video
                        else None
                    ),
                }
            )

        return {
            "id": str(uuid.uuid4()),
            "filename": filename,
            "type": "video" if is_video else "image",
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "is_fake": is_fake,
            "per_model_scores": {
                "efficientnet": round(eff, 4),
                "vit": round(vit, 4),
                "cnn_lstm": round(lstm, 4) if lstm is not None else None,
            },
            "frame_analysis": frame_analysis,
            "metadata": {
                "processing_time_ms": round((time.time() - start) * 1000, 2),
                "resolution": "1920x1080",
                "fps": 30 if is_video else None,
                "frames_analyzed": num_frames,
                "faces_detected": 1,
                "mode": "simulation",
                "cnn_lstm_loaded": False,
            },
        }