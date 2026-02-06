"""
Detection-only processor for parallel multi-model detection.

This module provides detection without blurring, designed to be used
in parallel detection tasks where results are merged before blurring.

Unlike the Anonymize class which performs detection and blurring together,
DetectionOnlyProcessor extracts and returns structured detection data that
can be cached, merged with detections from other models, and blurred later.
"""

import os
import cv2
import torch
import numpy as np
import logging
from typing import List, Dict, Optional

from ultralytics import YOLO

from .media_utils import is_image

logger = logging.getLogger(__name__)

# Class name aliases for flexible matching
# Maps normalized class names to their common aliases
CLASS_ALIASES = {
    'plate': ['license_plate', 'license plate', 'licenseplate', 'plate', 'number_plate'],
    'license_plate': ['plate', 'license plate', 'licenseplate', 'number_plate'],
    'face': ['face', 'faces'],
}


class DetectionOnlyProcessor:
    """
    Runs YOLO detection without applying blur.
    Returns structured detection results for later merging.

    This class is designed for parallel detection scenarios where multiple
    models process the same media, and their results are combined before
    applying blur in a single pass.

    Args:
        model_path: Path to YOLO model file
        device: Device to run on ('cuda', 'cpu', or 'auto')

    Example:
        processor = DetectionOnlyProcessor('/path/to/yolo11n.pt')
        results = processor.detect_media('/path/to/video.mp4', ['person', 'car'])
    """

    def __init__(self, model_path: str, device: str = 'auto'):
        self.model_path = model_path
        self.device = device if device != 'auto' else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model = None
        self.class_list = []
        self._is_segmentation = False

    def load_model(self):
        """Load the YOLO model."""
        logger.info(f"[DetectionOnly] Loading model: {self.model_path}")
        self.model = YOLO(self.model_path)

        # Get class names - handle both PyTorch and ONNX models
        # For ONNX, model.model is a string (path), so use model.names directly
        if hasattr(self.model, 'names') and self.model.names:
            self.class_list = list(self.model.names.values())
        elif hasattr(self.model.model, 'names'):
            self.class_list = list(self.model.model.names.values())
        else:
            logger.warning(f"[DetectionOnly] Could not extract class names from model")
            self.class_list = []

        # Normalize class names to lowercase for consistent matching
        self.class_list = [c.lower() for c in self.class_list]

        self._is_segmentation = self._detect_segmentation_model()
        logger.info(f"[DetectionOnly] Model loaded. Segmentation: {self._is_segmentation}, "
                    f"Classes: {self.class_list[:10]}{'...' if len(self.class_list) > 10 else ''}")

    def _detect_segmentation_model(self) -> bool:
        """
        Check if model supports segmentation.

        Returns:
            True if model is a segmentation model
        """
        # Check model path
        if 'seg' in self.model_path.lower():
            return True
        if '/segment/' in self.model_path or '\\segment\\' in self.model_path:
            return True

        # Check model task attribute
        try:
            if hasattr(self.model, 'task') and self.model.task == 'segment':
                return True
        except Exception:
            pass

        return False

    def detect_media(self, media_path: str, classes_to_detect: List[str],
                     detection_threshold: float = 0.25,
                     use_tracking: bool = True) -> Dict:
        """
        Run detection on media (image or video).

        Args:
            media_path: Path to media file
            classes_to_detect: List of class names to detect
            detection_threshold: Confidence threshold
            use_tracking: Use tracking for video (enables track IDs for interpolation)

        Returns:
            DetectionResult dict with structure:
            {
                'media_path': str,
                'model_id': str,
                'model_type': 'detect' | 'segment',
                'frame_detections': {frame_idx: [detection, ...]},
                'total_frames': int,
                'error': str | None,
            }
        """
        if not self.model:
            self.load_model()

        results = {
            'media_path': media_path,
            'model_id': self.model_path,
            'model_type': 'segment' if self._is_segmentation else 'detect',
            'frame_detections': {},
            'total_frames': 0,
            'error': None,
        }

        # Get class indices for filtering (with alias support)
        classes_lower = [c.lower() for c in classes_to_detect]
        class_indices = []

        for i, model_class in enumerate(self.class_list):
            model_class_lower = model_class.lower()

            # Check direct match
            if model_class_lower in classes_lower:
                class_indices.append(i)
                continue

            # Check aliases
            for requested_class in classes_lower:
                # Get aliases for the requested class
                aliases = CLASS_ALIASES.get(requested_class, [requested_class])
                # Also check if model class has aliases that include requested class
                model_aliases = CLASS_ALIASES.get(model_class_lower, [model_class_lower])

                if model_class_lower in aliases or requested_class in model_aliases:
                    class_indices.append(i)
                    logger.debug(f"[DetectionOnly] Matched '{requested_class}' to model class '{model_class}' via alias")
                    break

        if not class_indices:
            error_msg = (f"No matching classes found in model. "
                         f"Requested: {classes_to_detect}, "
                         f"Available: {self.class_list[:10]}...")
            logger.warning(f"[DetectionOnly] {error_msg}")
            results['error'] = error_msg
            return results

        logger.info(f"[DetectionOnly] Detecting classes {classes_to_detect} "
                    f"(indices: {class_indices}) in {media_path}")

        if is_image(media_path):
            return self._detect_image(media_path, class_indices,
                                      detection_threshold, results)
        else:
            return self._detect_video(media_path, class_indices,
                                      detection_threshold, use_tracking, results)

    def _detect_image(self, image_path: str, class_indices: List[int],
                      threshold: float, results: Dict) -> Dict:
        """
        Detect objects in a single image.

        Args:
            image_path: Path to image file
            class_indices: YOLO class indices to detect
            threshold: Confidence threshold
            results: Results dict to populate

        Returns:
            Updated results dict
        """
        img = cv2.imread(image_path)
        if img is None:
            results['error'] = f"Could not load image: {image_path}"
            return results

        task = 'segment' if self._is_segmentation else 'detect'

        try:
            preds = self.model.predict(
                source=img,
                task=task,
                device=self.device,
                retina_masks=self._is_segmentation,
                imgsz=max(img.shape[:2]),
                conf=threshold,
                classes=class_indices,
                verbose=False
            )

            results['total_frames'] = 1
            results['frame_detections'][0] = self._extract_detections(preds[0])

            det_count = len(results['frame_detections'][0])
            logger.info(f"[DetectionOnly] Image: {det_count} detections")

        except Exception as e:
            results['error'] = str(e)
            logger.error(f"[DetectionOnly] Image detection error: {e}")

        return results

    def _detect_video(self, video_path: str, class_indices: List[int],
                      threshold: float, use_tracking: bool, results: Dict) -> Dict:
        """
        Detect objects in video frames.

        Uses tracking mode for videos to enable track IDs, which allows
        for detection interpolation in the blurring phase.

        Args:
            video_path: Path to video file
            class_indices: YOLO class indices to detect
            threshold: Confidence threshold
            use_tracking: Whether to use tracking mode
            results: Results dict to populate

        Returns:
            Updated results dict
        """
        # Get video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            results['error'] = f"Could not open video: {video_path}"
            return results

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        logger.info(f"[DetectionOnly] Video: {width}x{height}, {fps}fps, {total_frames} frames")

        task = 'segment' if self._is_segmentation else 'detect'
        imgsz = max(width, height)

        try:
            # Use track for videos (enables tracking IDs for interpolation)
            if use_tracking:
                preds = self.model.track(
                    source=video_path,
                    task=task,
                    device=self.device,
                    retina_masks=self._is_segmentation,
                    imgsz=imgsz,
                    conf=threshold,
                    classes=class_indices,
                    verbose=False
                )
            else:
                preds = self.model.predict(
                    source=video_path,
                    task=task,
                    device=self.device,
                    retina_masks=self._is_segmentation,
                    imgsz=imgsz,
                    conf=threshold,
                    classes=class_indices,
                    verbose=False,
                    stream=True
                )

            results['total_frames'] = total_frames
            total_detections = 0

            for frame_idx, pred in enumerate(preds):
                detections = self._extract_detections(pred)
                if detections:
                    results['frame_detections'][frame_idx] = detections
                    total_detections += len(detections)

            logger.info(f"[DetectionOnly] Video: {total_detections} total detections "
                        f"across {len(results['frame_detections'])} frames")

        except Exception as e:
            error_str = str(e).lower()
            # Check if it's a CUDA error and we were using GPU
            if 'cuda' in error_str and self.device == 'cuda':
                logger.warning(f"[DetectionOnly] CUDA error detected, retrying on CPU: {e}")
                try:
                    # Clear GPU state
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()

                    # Retry on CPU
                    self.device = 'cpu'
                    if use_tracking:
                        preds = self.model.track(
                            source=video_path,
                            task=task,
                            device='cpu',
                            retina_masks=self._is_segmentation,
                            imgsz=imgsz,
                            conf=threshold,
                            classes=class_indices,
                            verbose=False
                        )
                    else:
                        preds = self.model.predict(
                            source=video_path,
                            task=task,
                            device='cpu',
                            retina_masks=self._is_segmentation,
                            imgsz=imgsz,
                            conf=threshold,
                            classes=class_indices,
                            verbose=False,
                            stream=True
                        )

                    results['total_frames'] = total_frames
                    total_detections = 0

                    for frame_idx, pred in enumerate(preds):
                        detections = self._extract_detections(pred)
                        if detections:
                            results['frame_detections'][frame_idx] = detections
                            total_detections += len(detections)

                    logger.info(f"[DetectionOnly] Video (CPU fallback): {total_detections} total detections "
                                f"across {len(results['frame_detections'])} frames")

                except Exception as cpu_error:
                    results['error'] = f"GPU error: {e}, CPU fallback also failed: {cpu_error}"
                    logger.error(f"[DetectionOnly] CPU fallback also failed: {cpu_error}")
            else:
                results['error'] = str(e)
                logger.error(f"[DetectionOnly] Video detection error: {e}")

        return results

    def _extract_detections(self, prediction) -> List[Dict]:
        """
        Extract detection data from YOLO prediction.

        Converts YOLO prediction objects into plain dict format suitable
        for serialization and caching.

        Args:
            prediction: YOLO prediction result for one frame

        Returns:
            List of detection dicts with bbox, class, confidence, etc.
        """
        detections = []

        if not prediction.boxes or len(prediction.boxes) == 0:
            return detections

        has_masks = (self._is_segmentation and
                     hasattr(prediction, 'masks') and
                     prediction.masks is not None)

        for i, box in enumerate(prediction.boxes):
            det = {
                'bbox': box.xyxy[0].cpu().numpy().tolist(),
                'class': prediction.names[int(box.cls)],
                'confidence': float(box.conf),
                'track_id': int(box.id) if hasattr(box, 'id') and box.id is not None else None,
                'mask': None,
            }

            # Extract segmentation mask if available
            if has_masks and i < len(prediction.masks.data):
                mask = prediction.masks.data[i].cpu().numpy()
                det['mask'] = (mask * 255).astype(np.uint8)

            detections.append(det)

        return detections

    def unload(self):
        """Release model from memory with thorough GPU cleanup."""
        if self.model is not None:
            del self.model
            self.model = None
            self.class_list = []

            # Force garbage collection for GPU memory
            import gc
            gc.collect()

            if torch.cuda.is_available():
                try:
                    # Synchronize all CUDA streams before cleanup
                    torch.cuda.synchronize()
                    # Clear CUDA cache
                    torch.cuda.empty_cache()
                    # Reset peak memory stats
                    torch.cuda.reset_peak_memory_stats()
                    # Additional synchronization after cleanup
                    torch.cuda.synchronize()
                except Exception as e:
                    logger.warning(f"[DetectionOnly] CUDA cleanup warning: {e}")

            logger.info("[DetectionOnly] Model unloaded")
