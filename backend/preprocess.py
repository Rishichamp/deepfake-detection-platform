"""
Preprocessing pipeline for deepfake detection.

Handles frame extraction from videos, face detection using OpenCV Haar cascades,
and face crop resizing for model input. Supports both image and video file inputs.
"""

import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Supported file extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


@dataclass
class BoundingBox:
    """Face bounding box with coordinates."""

    x: int
    y: int
    w: int
    h: int

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def area(self) -> int:
        return self.w * self.h

    def expand(self, factor: float, max_w: int, max_h: int) -> "BoundingBox":
        """Expand bounding box by a factor while staying within image bounds."""
        cx, cy = self.x + self.w // 2, self.y + self.h // 2
        new_w = int(self.w * factor)
        new_h = int(self.h * factor)
        new_x = max(0, cx - new_w // 2)
        new_y = max(0, cy - new_h // 2)
        new_w = min(new_w, max_w - new_x)
        new_h = min(new_h, max_h - new_y)
        return BoundingBox(x=new_x, y=new_y, w=new_w, h=new_h)


@dataclass
class PreprocessResult:
    """Result from the preprocessing pipeline."""

    frames: list[Image.Image] = field(default_factory=list)
    faces: list[list[BoundingBox]] = field(default_factory=list)
    face_crops: list[list[Image.Image]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class VideoPreprocessor:
    """
    Preprocessing pipeline for deepfake detection.

    Extracts frames from video/images, detects faces using OpenCV Haar cascades,
    and produces 224x224 face crops suitable for model input.
    """

    # Default Haar cascade shipped with OpenCV
    DEFAULT_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    def __init__(
        self,
        target_size: int = 224,
        num_frames: int = 16,
        cascade_path: Optional[str] = None,
        face_expand_factor: float = 1.3,
        min_face_size: tuple[int, int] = (30, 30),
    ):
        self.target_size = target_size
        self.num_frames = num_frames
        self.face_expand_factor = face_expand_factor
        self.min_face_size = min_face_size

        # Load the Haar cascade classifier
        cascade_file = cascade_path or self.DEFAULT_CASCADE
        if not os.path.isfile(cascade_file):
            raise FileNotFoundError(
                f"Haar cascade file not found: {cascade_file}. "
                "Ensure opencv-python or opencv-python-headless is installed."
            )
        self.face_cascade = cv2.CascadeClassifier(cascade_file)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_file}")
        logger.info("Loaded Haar cascade from %s", cascade_file)

    # ------------------------------------------------------------------ #
    #  Frame extraction
    # ------------------------------------------------------------------ #

    def extract_frames(
        self, video_path: str, num_frames: Optional[int] = None
    ) -> list[Image.Image]:
        """
        Extract evenly-spaced frames from a video file.

        Args:
            video_path: Path to the video file.
            num_frames:  Number of frames to extract (default: self.num_frames).

        Returns:
            List of PIL Images (RGB).

        Raises:
            FileNotFoundError: If the video file does not exist.
            ValueError: If the video cannot be opened or contains no frames.
        """
        num_frames = num_frames or self.num_frames
        video_path = str(video_path)

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(
                "Video: %d frames, %.1f fps, %dx%d",
                total_frames,
                fps,
                width,
                height,
            )

            if total_frames <= 0:
                raise ValueError(f"Video has no readable frames: {video_path}")

            # Compute evenly-spaced frame indices
            actual_num = min(num_frames, total_frames)
            if actual_num == total_frames:
                indices = list(range(total_frames))
            else:
                indices = [
                    int(round(i * (total_frames - 1) / (actual_num - 1)))
                    for i in range(actual_num)
                ]

            frames: list[Image.Image] = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Could not read frame %d, skipping", idx)
                    continue
                # BGR -> RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))

            if not frames:
                raise ValueError("No frames could be extracted from the video.")

            logger.info("Extracted %d frames from video", len(frames))
            return frames

        finally:
            cap.release()

    def load_image(self, image_path: str) -> list[Image.Image]:
        """
        Load a single image file as a list containing one PIL Image.

        Args:
            image_path: Path to the image file.

        Returns:
            Single-element list of PIL Images (RGB).
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            img = Image.open(image_path).convert("RGB")
            logger.info("Loaded image: %dx%d", img.width, img.height)
            return [img]
        except Exception as e:
            raise ValueError(f"Failed to open image {image_path}: {e}") from e

    # ------------------------------------------------------------------ #
    #  Face detection
    # ------------------------------------------------------------------ #

    def detect_faces(
        self,
        frame: Image.Image,
        cascade_path: Optional[str] = None,
    ) -> list[BoundingBox]:
        """
        Detect faces in a single frame using Haar cascades.

        Args:
            frame: PIL Image (RGB).
            cascade_path: Optional override cascade XML path.

        Returns:
            List of BoundingBox instances for detected faces.
        """
        # Use a separate cascade if requested, else fall back to the instance one
        if cascade_path:
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                logger.error("Failed to load cascade from %s, using default", cascade_path)
                cascade = self.face_cascade
        else:
            cascade = self.face_cascade

        # Convert PIL -> grayscale numpy array
        frame_np = np.array(frame)
        gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)

        # Histogram equalization to handle varying lighting
        gray = cv2.equalizeHist(gray)

        detections = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.min_face_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(detections) == 0:
            return []

        bboxes = [
            BoundingBox(x=int(x), y=int(y), w=int(w), h=int(h))
            for (x, y, w, h) in detections
        ]

        # Sort by area (largest first) — primary face is likely the biggest
        bboxes.sort(key=lambda b: b.area(), reverse=True)
        logger.debug("Detected %d face(s) in frame", len(bboxes))
        return bboxes

    # ------------------------------------------------------------------ #
    #  Crop & resize
    # ------------------------------------------------------------------ #

    def crop_and_resize(
        self,
        frame: Image.Image,
        bbox: BoundingBox,
        target_size: Optional[int] = None,
    ) -> Image.Image:
        """
        Crop a face region from a frame and resize to target dimensions.

        The bounding box is expanded slightly to include more context around
        the face (forehead, chin, etc.), which helps detection models.

        Args:
            frame: PIL Image (RGB).
            bbox:  Bounding box of the detected face.
            target_size: Output square dimension (default: self.target_size).

        Returns:
            PIL Image resized to (target_size, target_size).
        """
        target_size = target_size or self.target_size

        # Expand the bounding box for context
        expanded = bbox.expand(
            self.face_expand_factor, max_w=frame.width, max_h=frame.height
        )

        # Crop and resize using high-quality Lanczos resampling
        crop = frame.crop(
            (expanded.x, expanded.y, expanded.x + expanded.w, expanded.y + expanded.h)
        )
        resized = crop.resize((target_size, target_size), Image.LANCZOS)
        return resized

    # ------------------------------------------------------------------ #
    #  Full pipeline
    # ------------------------------------------------------------------ #

    def preprocess_pipeline(self, file_path: str) -> dict:
        """
        Run the full preprocessing pipeline on a file.

        1. Determine if the file is an image or video.
        2. Extract frames (or load the single image).
        3. Detect faces in every frame.
        4. Crop and resize detected faces to model input size.

        Args:
            file_path: Path to the input image or video file.

        Returns:
            Dictionary with keys:
                - frames:     list[PIL.Image] — raw extracted frames
                - faces:      list[list[BoundingBox]] — per-frame face boxes
                - face_crops: list[list[PIL.Image]] — per-frame resized crops
                - metadata:   dict with file info and statistics
        """
        file_path = str(file_path)
        ext = Path(file_path).suffix.lower()
        filename = Path(file_path).name

        logger.info("Preprocessing pipeline started for: %s", filename)

        # Determine input type
        if ext in VIDEO_EXTENSIONS:
            input_type = "video"
            frames = self.extract_frames(file_path)
        elif ext in IMAGE_EXTENSIONS:
            input_type = "image"
            frames = self.load_image(file_path)
        else:
            raise ValueError(
                f"Unsupported file extension '{ext}'. "
                f"Supported: {VIDEO_EXTENSIONS | IMAGE_EXTENSIONS}"
            )

        # Per-frame face detection and cropping
        all_faces: list[list[BoundingBox]] = []
        all_crops: list[list[Image.Image]] = []
        total_faces_found = 0

        for i, frame in enumerate(frames):
            bboxes = self.detect_faces(frame)
            all_faces.append(bboxes)

            crops = [self.crop_and_resize(frame, bb) for bb in bboxes]
            all_crops.append(crops)
            total_faces_found += len(bboxes)

            if not bboxes:
                logger.warning("No face detected in frame %d", i)

        # If no faces found in any frame, create a centre-crop fallback
        # so the model still has something to work with
        if total_faces_found == 0:
            logger.warning(
                "No faces detected in any frame. Using centre-crop fallback."
            )
            for i, frame in enumerate(frames):
                fallback = self._centre_crop_fallback(frame)
                all_faces[i] = [
                    BoundingBox(
                        x=(frame.width - min(frame.width, frame.height)) // 2,
                        y=(frame.height - min(frame.width, frame.height)) // 2,
                        w=min(frame.width, frame.height),
                        h=min(frame.width, frame.height),
                    )
                ]
                all_crops[i] = [fallback]

        metadata = {
            "filename": filename,
            "input_type": input_type,
            "num_frames": len(frames),
            "total_faces_detected": total_faces_found,
            "frame_dimensions": (
                (frames[0].width, frames[0].height) if frames else None
            ),
            "target_crop_size": self.target_size,
        }

        logger.info(
            "Pipeline complete: %d frames, %d total faces",
            len(frames),
            total_faces_found,
        )

        return {
            "frames": frames,
            "faces": all_faces,
            "face_crops": all_crops,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _centre_crop_fallback(self, frame: Image.Image) -> Image.Image:
        """Centre-crop the frame to a square and resize to target_size."""
        w, h = frame.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        crop = frame.crop((left, top, left + side, top + side))
        return crop.resize((self.target_size, self.target_size), Image.LANCZOS)
