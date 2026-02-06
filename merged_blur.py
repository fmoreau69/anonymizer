"""
Merged Blur Processor for WAMA Anonymizer.

Applies blur to media using pre-computed merged detection results from
multiple models. This processor does not perform detection - it only
handles the blurring phase using previously gathered detection data.

This enables multi-model parallel detection where different specialized
models (e.g., face detector + general object detector) can be run
simultaneously, their results merged, and blurred in a single pass.
"""

import os
import cv2
import gc
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

from .blur_utils import blur_detection, blur_segmentation, normalize_blur_ratio
from .media_utils import is_image
from .ffmpeg_utils import copy_audio_to_video

logger = logging.getLogger(__name__)


class MergedBlurProcessor:
    """
    Apply blur to media using pre-merged detection results.

    Unlike the Anonymize class which performs detection and blurring together,
    this processor only handles blurring based on detection data provided
    from the parallel detection pipeline.

    Args:
        source_dir: Input directory (optional, used for path resolution)
        destination_dir: Output directory for processed files

    Example:
        processor = MergedBlurProcessor(destination_dir='/output')
        processor.process_with_detections(
            media_path='/path/to/video.mp4',
            merged_detections=merged_results,
            blur_ratio=25,
            progressive_blur=15
        )
    """

    def __init__(self, source_dir: Optional[str] = None,
                 destination_dir: Optional[str] = None):
        # Import settings for defaults
        from wama.settings import MEDIA_INPUT_ROOT, MEDIA_OUTPUT_ROOT

        self.source = str(source_dir) if source_dir else str(MEDIA_INPUT_ROOT)
        self.destination = str(destination_dir) if destination_dir else str(MEDIA_OUTPUT_ROOT)

        os.makedirs(self.source, exist_ok=True)
        os.makedirs(self.destination, exist_ok=True)

        self.vid_writer = None
        self.temp_video_path = None
        self.meta_data = None
        self.output_path = None

    def process_with_detections(self, media_path: str, merged_detections: Dict, **kwargs):
        """
        Process media by applying blur to merged detections.

        Routes to image or video processing based on file type.

        Args:
            media_path: Path to input media file
            merged_detections: Merged detection results from parallel tasks
                Structure: {
                    'media_id': int,
                    'models_used': [str, ...],
                    'frame_detections': {frame_idx: [detection, ...]},
                    'total_frames': int,
                }
            **kwargs: Blur settings:
                - blur_ratio: Blur kernel size (default: 25)
                - progressive_blur: Progressive blur strength (default: 15)
                - roi_enlargement: Bounding box enlargement factor (default: 1.05)
                - rounded_edges: Corner rounding for blur mask (default: 5)
        """
        self.output_path = self._get_output_path(media_path, merged_detections.get('models_used', []))

        logger.info(f"[MergedBlur] Processing {media_path}")
        logger.info(f"[MergedBlur] Output: {self.output_path}")
        logger.info(f"[MergedBlur] Models used: {merged_detections.get('models_used', [])}")

        total_dets = sum(len(dets) for dets in merged_detections.get('frame_detections', {}).values())
        logger.info(f"[MergedBlur] Total detections to blur: {total_dets}")

        if is_image(media_path):
            self._process_image(media_path, self.output_path, merged_detections, **kwargs)
        else:
            self._process_video(media_path, self.output_path, merged_detections, **kwargs)

    def _get_output_path(self, input_path: str, models_used: List[str]) -> str:
        """
        Generate output path from input path with model suffix.

        Args:
            input_path: Path to input file
            models_used: List of model identifiers used

        Returns:
            Output path in destination directory
        """
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)

        # Create suffix based on models used
        if len(models_used) == 0:
            suffix = 'blurred'
        elif len(models_used) == 1:
            # Extract model name from path
            model_name = Path(models_used[0]).stem
            suffix = f'blurred_{model_name}'
        else:
            suffix = 'blurred_multi-model'

        return os.path.join(self.destination, f"{name}_{suffix}{ext}")

    def _process_image(self, input_path: str, output_path: str,
                       merged_detections: Dict, **kwargs):
        """
        Apply blur to a single image.

        Args:
            input_path: Path to input image
            output_path: Path for output image
            merged_detections: Merged detection results
            **kwargs: Blur settings
        """
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Could not load image: {input_path}")

        blur_ratio = normalize_blur_ratio(kwargs.get('blur_ratio', 25))
        progressive_blur = int(kwargs.get('progressive_blur', 15))
        roi_enlargement = kwargs.get('roi_enlargement', 1.05)
        rounded_edges = int(kwargs.get('rounded_edges', 5))

        # Get detections for frame 0 (images are single-frame)
        frame_dets = merged_detections.get('frame_detections', {})
        detections = frame_dets.get(0, []) or frame_dets.get('0', [])

        logger.info(f"[MergedBlur] Image: {len(detections)} detections to blur")

        for det in detections:
            if det.get('mask') is not None:
                img = blur_segmentation(img, det['mask'], blur_ratio, progressive_blur)
            else:
                img = blur_detection(
                    img, det['bbox'], det.get('class', 'object'),
                    blur_ratio, rounded_edges, progressive_blur, roi_enlargement
                )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save with appropriate quality settings
        ext = os.path.splitext(output_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        elif ext == '.png':
            cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        else:
            cv2.imwrite(output_path, img)

        logger.info(f"[MergedBlur] Image saved: {output_path}")

    def _process_video(self, input_path: str, output_path: str,
                       merged_detections: Dict, **kwargs):
        """
        Apply blur to video frames.

        Processes frame by frame, applying blur to all detections for each frame.
        Preserves audio by copying from original video.

        Args:
            input_path: Path to input video
            output_path: Path for output video
            merged_detections: Merged detection results
            **kwargs: Blur settings
        """
        # Setup video capture
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.meta_data = {'fps': fps, 'size': (width, height), 'total_frames': total_frames}

        logger.info(f"[MergedBlur] Video: {width}x{height}, {fps}fps, {total_frames} frames")

        # Setup video writer (intermediate AVI format for better quality)
        suffix = '.avi'
        fourcc = 'MJPG'
        self.temp_video_path = str(Path(output_path).with_suffix(suffix))

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        self.vid_writer = cv2.VideoWriter(
            self.temp_video_path,
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            (width, height),
            True
        )

        if not self.vid_writer.isOpened():
            # Fallback to mp4v if MJPEG fails
            logger.warning("[MergedBlur] MJPEG not available, using mp4v")
            suffix = '.mp4'
            fourcc = 'mp4v'
            self.temp_video_path = str(Path(output_path).with_suffix(suffix))
            self.vid_writer = cv2.VideoWriter(
                self.temp_video_path,
                cv2.VideoWriter_fourcc(*fourcc),
                fps,
                (width, height)
            )

        # Get blur settings
        blur_ratio = normalize_blur_ratio(kwargs.get('blur_ratio', 25))
        progressive_blur = int(kwargs.get('progressive_blur', 15))
        roi_enlargement = kwargs.get('roi_enlargement', 1.05)
        rounded_edges = int(kwargs.get('rounded_edges', 5))

        frame_detections = merged_detections.get('frame_detections', {})

        # Debug: Log detection statistics by type
        mask_count = 0
        bbox_count = 0
        for frame_dets in frame_detections.values():
            for det in frame_dets:
                if det.get('mask') is not None:
                    mask_count += 1
                else:
                    bbox_count += 1
        logger.info(f"[MergedBlur] Detection types: {mask_count} with masks, {bbox_count} bbox-only")

        # Process each frame
        frames_with_dets = 0
        total_blurs = 0
        mask_blurs = 0
        bbox_blurs = 0

        for frame_idx in tqdm(range(total_frames), desc='[MergedBlur] Blurring', unit='frames'):
            ret, frame = cap.read()
            if not ret:
                break

            # Get detections for this frame (handle both int and str keys)
            detections = frame_detections.get(frame_idx, [])
            if not detections:
                detections = frame_detections.get(str(frame_idx), [])

            if detections:
                frames_with_dets += 1
                total_blurs += len(detections)

            # Apply blur to each detection
            for det in detections:
                try:
                    if det.get('mask') is not None:
                        mask = det['mask']
                        # Debug first mask detection - save sample to disk for inspection
                        is_first_mask = (mask_blurs == 0)
                        if is_first_mask:
                            logger.info(f"[MergedBlur] First mask: shape={mask.shape}, dtype={mask.dtype}, "
                                       f"min={mask.min()}, max={mask.max()}, class={det.get('class')}")
                            # Save debug images to disk
                            debug_dir = os.path.join(self.destination, '_debug')
                            os.makedirs(debug_dir, exist_ok=True)
                            # Save the mask
                            cv2.imwrite(os.path.join(debug_dir, f'mask_frame{frame_idx}.png'), mask)
                            # Save the original frame (before blur)
                            cv2.imwrite(os.path.join(debug_dir, f'original_frame{frame_idx}.jpg'), frame.copy())
                            logger.info(f"[MergedBlur] Debug images saved to: {debug_dir}")

                        frame = blur_segmentation(frame, mask, blur_ratio, progressive_blur)

                        # Debug: save blurred frame for first mask
                        if is_first_mask:
                            cv2.imwrite(os.path.join(debug_dir, f'blurred_frame{frame_idx}.jpg'), frame)
                        mask_blurs += 1
                    else:
                        frame = blur_detection(
                            frame, det['bbox'], det.get('class', 'object'),
                            blur_ratio, rounded_edges, progressive_blur, roi_enlargement
                        )
                        bbox_blurs += 1
                except Exception as e:
                    logger.warning(f"[MergedBlur] Blur error at frame {frame_idx}: {e}")
                    continue

            self.vid_writer.write(frame)

        cap.release()
        self.vid_writer.release()

        logger.info(f"[MergedBlur] Processed {total_frames} frames, "
                    f"{frames_with_dets} with detections, {total_blurs} total blurs "
                    f"({mask_blurs} masks, {bbox_blurs} bboxes)")

        # Copy audio from original video to final output
        final_output = os.path.splitext(output_path)[0] + '.mp4'
        logger.info(f"[MergedBlur] Merging audio to: {final_output}")

        try:
            copy_audio_to_video(input_path, self.temp_video_path, final_output)
            self.output_path = final_output
            logger.info(f"[MergedBlur] Video complete: {final_output}")
        except Exception as e:
            logger.error(f"[MergedBlur] Audio merge failed: {e}")
            # If audio merge fails, just rename temp to output
            import shutil
            shutil.move(self.temp_video_path, final_output)
            self.output_path = final_output

        # Clean up temp file if it still exists
        if os.path.exists(self.temp_video_path) and self.temp_video_path != final_output:
            try:
                os.remove(self.temp_video_path)
            except Exception:
                pass

    def cleanup(self):
        """Release resources."""
        if self.vid_writer is not None:
            self.vid_writer.release()
            self.vid_writer = None

        gc.collect()
        logger.debug("[MergedBlur] Cleanup complete")
