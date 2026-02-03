"""
SAM3 Processor for WAMA Anonymizer
Handles image and video segmentation using SAM3 (Segment Anything Model 3)
for prompt-based object blurring.

This module provides the SAM3Processor class which is parallel to the YOLO-based
Anonymize class, enabling text prompt-driven segmentation and blurring.
"""

import os
import gc
import cv2
import torch
import numpy as np
import logging
import sys

from tqdm import tqdm
from pathlib import Path
from PIL import Image

from .blur_utils import blur_segmentation, normalize_blur_ratio
from .media_utils import is_image
from .ffmpeg_utils import copy_audio_to_video

# Add wama to path for settings import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from wama.settings import MEDIA_INPUT_ROOT, MEDIA_OUTPUT_ROOT, MODEL_PATHS, AI_MODELS_DIR

logger = logging.getLogger(__name__)


def setup_sam3_hf_environment():
    """
    Setup HuggingFace environment variables for SAM3 model loading.
    Reads the token from the SAM3 models directory and configures HF cache.
    """
    # Get SAM3 directory from centralized config
    sam3_dir = MODEL_PATHS.get('vision', {}).get('sam')
    if sam3_dir:
        sam3_dir = Path(sam3_dir) / 'sam3'
    else:
        sam3_dir = AI_MODELS_DIR / 'models' / 'vision' / 'sam' / 'sam3'

    # Check for token file in SAM3 directory
    token_file = sam3_dir / 'token'
    if token_file.exists():
        try:
            token = token_file.read_text().strip()
            if token:
                # Set HuggingFace token environment variable
                os.environ['HF_TOKEN'] = token
                os.environ['HUGGING_FACE_HUB_TOKEN'] = token
                logger.info(f"[SAM3] HuggingFace token loaded from {token_file}")

                # Also try to save it to the standard location for huggingface_hub
                try:
                    from huggingface_hub import HfFolder
                    HfFolder.save_token(token)
                    logger.debug("[SAM3] Token saved to HuggingFace standard location")
                except Exception as e:
                    logger.debug(f"[SAM3] Could not save token to standard location: {e}")

        except Exception as e:
            logger.warning(f"[SAM3] Could not read token from {token_file}: {e}")

    # Set HuggingFace cache to SAM3 directory
    if sam3_dir.exists():
        # Configure HF to use SAM3 directory as cache
        os.environ['HF_HOME'] = str(sam3_dir.parent)
        os.environ['HF_HUB_CACHE'] = str(sam3_dir.parent)
        logger.info(f"[SAM3] HuggingFace cache set to: {sam3_dir.parent}")


class SAM3Processor:
    """
    SAM3-based processor for prompt-driven object segmentation and blurring.

    This class provides similar functionality to the YOLO-based Anonymize class,
    but uses SAM3 (Segment Anything Model 3) for text prompt-based segmentation.

    Args:
        source_dir: Custom input directory (defaults to MEDIA_INPUT_ROOT)
        destination_dir: Custom output directory (defaults to MEDIA_OUTPUT_ROOT)

    Usage:
        processor = SAM3Processor(source_dir='/path/to/input', destination_dir='/path/to/output')
        processor.process(
            media_path='/path/to/video.mp4',
            sam3_prompt='blur all faces and license plates',
            blur_ratio=25,
            progressive_blur=15
        )
    """

    def __init__(self, source_dir=None, destination_dir=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"[SAM3] Using device: {self.device}")
        print(f"[SAM3] Using device: {self.device}")

        # Path settings - use custom paths or fall back to Django settings
        self.source = str(source_dir) if source_dir else str(MEDIA_INPUT_ROOT)
        self.destination = str(destination_dir) if destination_dir else str(MEDIA_OUTPUT_ROOT)
        os.makedirs(self.source, exist_ok=True)
        os.makedirs(self.destination, exist_ok=True)

        self.input_path = None
        self.output_path = None
        self.temp_video_path = None

        # Model instances (lazy loaded)
        self.image_model = None
        self.image_processor = None
        self.video_predictor = None
        self._model_loaded = False

        # Processing settings
        self.text_prompt = ""
        self.blur_ratio = 25
        self.progressive_blur = 15
        self.confidence_threshold = 0.3

        # Video processing
        self.meta_data = None
        self.vid_writer = None

    def load_model(self, model_type='auto', **kwargs):
        """
        Load SAM3 model for image or video processing.

        Args:
            model_type: 'image', 'video', or 'auto' (detects based on input)
            **kwargs: Additional arguments passed to model builder

        Raises:
            ImportError: If SAM3 is not installed
            RuntimeError: If model loading fails
        """
        if self._model_loaded:
            logger.info("[SAM3] Model already loaded")
            return

        # Setup HuggingFace environment BEFORE importing SAM3 modules
        setup_sam3_hf_environment()

        try:
            # Import SAM3 modules
            # Note: We only import the image model as video predictor requires 'triton' (Linux only)
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor as Sam3ImageProcessor

            if model_type in ['image', 'auto']:
                logger.info("[SAM3] Loading image model...")
                print("[SAM3] Loading image model...")
                self.image_model = build_sam3_image_model()
                self.image_processor = Sam3ImageProcessor(self.image_model)
                logger.info("[SAM3] Image model loaded successfully")
                print("[SAM3] Image model loaded successfully")

            # Note: Video predictor requires 'triton' which is not available on Windows.
            # We use the image processor for frame-by-frame video processing instead.
            if model_type == 'video':
                logger.warning("[SAM3] Video predictor not available on Windows (requires triton)")
                logger.warning("[SAM3] Using image processor for frame-by-frame video processing")
                print("[SAM3] Video predictor not available, using image processor for videos")
                # Ensure image model is loaded for video processing
                if self.image_model is None:
                    self.load_model('image')

            self._model_loaded = True

        except ImportError as e:
            error_msg = f"SAM3 not installed. Install with: pip install sam3\nError: {e}"
            logger.error(error_msg)
            raise ImportError(error_msg)
        except Exception as e:
            error_msg = f"Failed to load SAM3 model. Check HuggingFace authentication.\nError: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _ensure_image_model(self):
        """Load image model if not already loaded."""
        if self.image_model is None or self.image_processor is None:
            self.load_model('image')

    def _ensure_video_model(self):
        """Load model for video processing.
        Since video predictor requires triton (not available on Windows),
        we use the image model for frame-by-frame processing.
        """
        if self.image_model is None or self.image_processor is None:
            self.load_model('image')

    def process(self, **kwargs):
        """
        Main processing entry point.
        Routes to image or video processing based on input file type.

        Args:
            **kwargs: Processing parameters including:
                - media_path: Path to input file
                - sam3_prompt: Text prompt for segmentation
                - blur_ratio: Blur kernel size (default: 25)
                - progressive_blur: Progressive blur strength (default: 15)
                - output_path: Optional custom output path

        Raises:
            ValueError: If no prompt is provided
            FileNotFoundError: If input file doesn't exist
        """
        # Get parameters
        self.text_prompt = kwargs.get('sam3_prompt', self.text_prompt)
        if not self.text_prompt or not self.text_prompt.strip():
            raise ValueError("SAM3 requires a text prompt. Please provide 'sam3_prompt' parameter.")

        self.input_path = kwargs.get('media_path', self.input_path)
        if not self.input_path or not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        self.blur_ratio = normalize_blur_ratio(kwargs.get('blur_ratio', self.blur_ratio))
        self.progressive_blur = int(kwargs.get('progressive_blur', self.progressive_blur))

        logger.info(f"[SAM3] Processing with prompt: {self.text_prompt[:50]}...")
        print(f"[SAM3] Processing with prompt: {self.text_prompt[:50]}...")

        # Route to appropriate processing method
        if is_image(self.input_path):
            self._ensure_image_model()
            self.process_image(**kwargs)
        else:
            self._ensure_video_model()
            self.process_video(**kwargs)

    def process_image(self, **kwargs):
        """
        Process a single image using SAM3.

        Args:
            **kwargs: Processing parameters
        """
        logger.info(f"[SAM3] Processing image: {self.input_path}")
        print(f"[SAM3] Processing image: {self.input_path}")

        # Load image
        pil_image = Image.open(self.input_path).convert('RGB')
        cv_image = cv2.imread(self.input_path)

        if cv_image is None:
            raise RuntimeError(f"Could not load image: {self.input_path}")

        # Get output path
        self.output_path = kwargs.get('output_path', self._get_output_path(self.input_path))

        try:
            # Set image in processor
            inference_state = self.image_processor.set_image(pil_image)

            # Get segmentation from text prompt
            output = self.image_processor.set_text_prompt(
                state=inference_state,
                prompt=self.text_prompt
            )

            masks = output.get("masks", [])
            scores = output.get("scores", [])

            logger.info(f"[SAM3] Found {len(masks)} masks for prompt")
            print(f"[SAM3] Found {len(masks)} masks for prompt")

            # Apply blur to detected regions
            blurred_image = cv_image.copy()
            for i, mask in enumerate(masks):
                score = scores[i] if i < len(scores) else 1.0
                if score >= self.confidence_threshold:
                    # Convert mask to numpy array
                    mask_np = self._convert_mask_to_numpy(mask, cv_image.shape[:2])

                    # Apply blur
                    blurred_image = blur_segmentation(
                        blurred_image, mask_np,
                        self.blur_ratio,
                        self.progressive_blur
                    )

            # Save output
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            cv2.imwrite(self.output_path, blurred_image)

            logger.info(f"[SAM3] Image processed successfully: {self.output_path}")
            print(f"[SAM3] Image processed successfully: {self.output_path}")

        except Exception as e:
            logger.error(f"[SAM3] Error processing image: {e}")
            raise
        finally:
            # Clean up
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def process_video(self, **kwargs):
        """
        Process a video using SAM3 image processor frame-by-frame.

        Since SAM3's video predictor requires 'triton' which is not available on Windows,
        we process each frame independently using the image processor with text prompts.

        Args:
            **kwargs: Processing parameters
        """
        logger.info(f"[SAM3] Processing video: {self.input_path}")
        print(f"[SAM3] Processing video: {self.input_path}")

        # Ensure image model is loaded (we use it for frame-by-frame processing)
        self._ensure_image_model()

        # Get output path
        self.output_path = kwargs.get('output_path', self._get_output_path(self.input_path))

        # Progress callback
        progress_callback = kwargs.get('progress_callback', None)

        cap = None

        try:
            # Get video properties
            cap = cv2.VideoCapture(self.input_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {self.input_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.meta_data = {'fps': fps, 'size': (width, height), 'total_frames': total_frames}

            # Setup video writer
            self._setup_video_writer()

            logger.info(f"[SAM3] Video: {width}x{height}, {fps}fps, {total_frames} frames")
            print(f"[SAM3] Video: {width}x{height}, {fps}fps, {total_frames} frames")
            logger.info(f"[SAM3] Using frame-by-frame processing with prompt: {self.text_prompt}")
            print(f"[SAM3] Using frame-by-frame processing with prompt: {self.text_prompt}")

            # Process each frame using image processor
            for frame_idx in tqdm(range(total_frames), desc='[SAM3] Processing video'):
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR (OpenCV) to RGB (PIL)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)

                try:
                    # Set image in processor
                    inference_state = self.image_processor.set_image(pil_frame)

                    # Get segmentation from text prompt
                    output = self.image_processor.set_text_prompt(
                        state=inference_state,
                        prompt=self.text_prompt
                    )

                    masks = output.get("masks", [])
                    scores = output.get("scores", [])

                    # Apply blur to detected regions
                    blurred_frame = frame.copy()
                    for i, mask in enumerate(masks):
                        score = scores[i] if i < len(scores) else 1.0
                        if score >= self.confidence_threshold:
                            mask_np = self._convert_mask_to_numpy(mask, frame.shape[:2])
                            blurred_frame = blur_segmentation(
                                blurred_frame, mask_np,
                                self.blur_ratio,
                                self.progressive_blur
                            )

                    self.vid_writer.write(blurred_frame)

                except Exception as frame_error:
                    # If frame processing fails, write original frame
                    logger.warning(f"[SAM3] Frame {frame_idx} error: {frame_error}, using original")
                    self.vid_writer.write(frame)

                # Report progress
                if progress_callback:
                    progress = int((frame_idx + 1) / total_frames * 100)
                    progress_callback(progress)

                # Periodic memory cleanup (every 100 frames)
                if frame_idx % 100 == 0 and frame_idx > 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            cap.release()
            self.vid_writer.release()

            # Copy audio from original video
            logger.info("[SAM3] Merging audio...")
            print("[SAM3] Merging audio...")

            final_output = self.output_path
            if not final_output.endswith('.mp4'):
                final_output = os.path.splitext(final_output)[0] + '.mp4'

            copy_audio_to_video(self.input_path, self.temp_video_path, final_output)

            self.output_path = final_output
            logger.info(f"[SAM3] Video processed successfully: {final_output}")
            print(f"[SAM3] Video processed successfully: {final_output}")

        except Exception as e:
            logger.error(f"[SAM3] Error processing video: {e}")
            raise
        finally:
            # Release resources
            if cap is not None:
                cap.release()
            if self.vid_writer is not None:
                self.vid_writer.release()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _convert_mask_to_numpy(self, mask, target_shape):
        """
        Convert a SAM3 mask to numpy array suitable for blurring.

        Args:
            mask: SAM3 output mask (torch tensor or numpy array)
            target_shape: Target (height, width) tuple

        Returns:
            numpy array with shape matching target, values 0-255
        """
        # Convert tensor to numpy
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()

        # Ensure mask is 2D
        if mask.ndim == 3:
            mask = mask.squeeze()
        if mask.ndim == 4:
            mask = mask.squeeze(0).squeeze(0)

        # Convert to uint8 (0-255 range)
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        # Resize to target shape if needed
        if mask.shape[:2] != target_shape:
            mask = cv2.resize(mask, (target_shape[1], target_shape[0]),
                            interpolation=cv2.INTER_LINEAR)

        return mask

    def _setup_video_writer(self):
        """Setup video writer for output."""
        if self.meta_data is None:
            raise RuntimeError("Video metadata not set. Call process_video first.")

        fps = self.meta_data['fps']
        width, height = self.meta_data['size']

        # Use AVI format with MJPG codec (same as Anonymize class)
        suffix = '.avi'
        fourcc = 'MJPG'
        self.temp_video_path = str(Path(self.output_path).with_suffix(suffix))

        self.vid_writer = cv2.VideoWriter(
            self.temp_video_path,
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            (width, height),
            True
        )

        if not self.vid_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for: {self.temp_video_path}")

    def _get_output_path(self, input_path):
        """
        Generate output path from input path.

        Args:
            input_path: Path to input file

        Returns:
            Output path in destination directory with model suffix
        """
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        return os.path.join(self.destination, f"{name}_blurred_sam3{ext}")

    def cleanup(self):
        """Release all resources and models."""
        logger.info("[SAM3] Cleaning up resources...")

        if self.vid_writer is not None:
            self.vid_writer.release()
            self.vid_writer = None

        # Clear model references
        self.image_model = None
        self.image_processor = None
        self.video_predictor = None
        self._model_loaded = False

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("[SAM3] Cleanup complete")


def process_with_sam3(**kwargs):
    """
    Convenience function to process a file with SAM3.

    Args:
        **kwargs: Processing parameters (see SAM3Processor.process)

    Returns:
        str: Path to output file
    """
    processor = SAM3Processor()
    try:
        processor.load_model('auto')
        processor.process(**kwargs)
        return processor.output_path
    finally:
        processor.cleanup()
