import os
import gc
import cv2
import torch
import numpy as np
import sys

from tqdm import tqdm
from pathlib import Path

from .blur_utils import blur_detection, blur_segmentation, normalize_blur_ratio
from .media_utils import is_image
from .ffmpeg_utils import copy_audio_to_video
from ultralytics import YOLO, settings
from ultralytics.utils import MACOS, WINDOWS

# Add wama to path for settings import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from wama.settings import MEDIA_INPUT_ROOT, MEDIA_OUTPUT_ROOT


class Anonymize:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Path settings - use paths from Django settings
        self.source, self.destination = str(MEDIA_INPUT_ROOT), str(MEDIA_OUTPUT_ROOT)
        os.makedirs(self.source, exist_ok=True), os.makedirs(self.destination, exist_ok=True)
        self.input_path, self.output_path = None, None
        self.save_path, self.models_dir = './runs', './models'
        settings.update({'runs_dir': self.save_path, 'weights_dir': self.models_dir})

        # Model settings
        self.class_list = []
        self.classes2blur = ['face', 'plate']  # ['person', 'car', 'truck', 'bus']
        self.model_name = None
        self.model_path = None
        self.model = None
        self.device = None
        self.tracker = None
        self.usage = (('predict', 'track'), ('detect', 'segment'))
        self.mode = self.usage[0][1]
        self.task = self.usage[1][0]
        self.meta_data = None
        self.ret_mask = False
        self.vid_writer = None
        self.results = None
        self.plotted_img = None

        # Option settings
        self.blur_ratio = 25
        self.rounded_edges = 5
        self.progressive_blur = 15
        self.ROI_enlargement = 1.05
        self.conf = 0.25
        self.blur = True
        self.show = True
        self.line_width = None
        self.boxes = True
        self.show_labels = True
        self.show_conf = True
        self.save = False
        self.save_txt = False

        # Interpolation settings
        self.interpolate_detections = True
        self.max_interpolation_frames = 15  # Will be capped at 0.5s based on FPS
        self.detection_buffer = {}  # {track_id: [(frame_idx, bbox, label), ...]}

    def load_model(self, **kwargs):
        self.model_name = 'yolov8n-seg.pt' if self.task == 'segment' else "yolov8n.pt"
        if any([classe in self.classes2blur for classe in ['face', 'plate']]):
            self.model_name = "yolov8m_faces&plates_720p.pt"  # "yolov8m_faces&plates_1080p.pt"
        self.model_path = kwargs.get('model_path', os.path.join(self.models_dir, self.model_name))
        print(f'Model used: {self.model_path}')
        self.model = YOLO(self.model_path)
        self.class_list = list(self.model.model.names.values())

        # Detect if model supports segmentation
        self._is_segmentation_model = self._detect_segmentation_model()

    def _detect_segmentation_model(self):
        """
        Detect if the loaded model is a segmentation model.

        Returns:
            bool: True if model supports segmentation, False otherwise
        """
        if not self.model:
            return False

        # Check if model path contains 'seg' or is in segment directory
        if 'seg' in self.model_path.lower() or '/segment/' in self.model_path or '\\segment\\' in self.model_path:
            return True

        # Check model task
        try:
            if hasattr(self.model, 'task') and self.model.task == 'segment':
                return True
        except:
            pass

        return False

    def process(self, **kwargs):
        if not self.model:
            print('❌ No model is loaded')
            return

        self.input_path = kwargs.get('media_path', self.input_path or self.source)

        # Folder
        if os.path.isdir(self.input_path):
            for media in os.listdir(self.input_path):
                media_path = os.path.join(self.input_path, media)
                self.input_path = media_path
                self.output_path = os.path.join(
                    self.destination, media[:-4] + '_blurred' + media[-4:]
                )

                if is_image(media_path):
                    self.process_image(media_path, self.output_path, **kwargs)
                else:
                    self.setup_source(**kwargs)
                    self.apply_process(**kwargs)
        # TODO: File list
        # File
        else:
            self.output_path = self.input_path[:-4].replace('input', 'output') + '_blurred' + self.input_path[-4:]

            if is_image(self.input_path):
                self.process_image(self.input_path, self.output_path, **kwargs)
            else:
                self.setup_source(**kwargs)
                self.apply_process(**kwargs)


    def process_image(self, input_path, output_path, **kwargs):
        img = cv2.imread(input_path)
        if img is None:
            print(f"❌ Could not load image: {input_path}")
            return

        self.classes2blur = kwargs.get('classes2blur', self.classes2blur)
        classes2blur_by_index = [
            i for i, name in enumerate(self.class_list) if name in self.classes2blur
        ]

        results = self.model.predict(
            source=img,
            task=self.task,
            device=self.device,
            retina_masks=self.ret_mask,
            imgsz=max(img.shape[:2]),
            conf=kwargs.get('detection_threshold', self.conf),
            classes=classes2blur_by_index,
            verbose=False
        )

        if results:
            self.plotted_img = results[0].plot(boxes=False, conf=False, labels=False)
            self.results = results
            self.blur_results(**kwargs)
        else:
            print("No detections found.")
            cv2.imwrite(output_path, img)


    def setup_source(self, **kwargs):
        print(f'Setting up media: {self.input_path}')
        cap = cv2.VideoCapture(self.input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Use a lossless/high-quality intermediate codec to preserve quality
        # We'll re-encode with FFmpeg later to match input codec
        suffix = '.avi'
        # Use FFV1 (lossless) or MJPEG (high quality) for intermediate file
        # FFV1 requires ffmpeg, so we use MJPEG which is widely supported
        fourcc = 'MJPG'  # Motion JPEG - high quality, widely supported

        save_path = str(Path(self.output_path).with_suffix(suffix))
        self.temp_video_path = save_path  # Store temp video path for later use
        self.meta_data = {'fps': fps, 'size': (width, height)}

        # Try to create video writer with high quality settings
        self.vid_writer = cv2.VideoWriter(
            save_path,
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            (width, height),
            True  # isColor
        )

        if not self.vid_writer.isOpened():
            # Fallback to mp4v if MJPEG fails
            print("Warning: MJPEG codec not available, using mp4v")
            suffix = '.mp4'
            fourcc = 'mp4v'
            save_path = str(Path(self.output_path).with_suffix(suffix))
            self.temp_video_path = save_path  # Update temp path
            self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))

    def apply_process(self, **kwargs):
        self.classes2blur = kwargs.get('classes2blur', self.classes2blur)
        classes2blur_by_index = [i for i, name in enumerate(self.class_list) if name in self.classes2blur]

        self.results = self.model.track(
            source=kwargs.get('media_path', self.input_path),
            # stream=True,
            task=self.task,
            mode=self.mode,
            device=self.device,
            retina_masks=self.ret_mask,
            imgsz=self.meta_data['size'][0] if 'size' in self.meta_data else self.meta_data['shape'][0],
            save=self.save,
            save_txt=self.save_txt,
            classes=classes2blur_by_index,
            conf=kwargs.get('detection_threshold', self.conf),
            show=kwargs.get('show_preview', self.show),
            boxes=kwargs.get('show_boxes', self.boxes),
            show_labels=kwargs.get('show_labels', self.show_labels),
            show_conf=kwargs.get('show_conf', self.show_conf)
        )
        # Blur detections
        if self.classes2blur:
            self.blur_results(**kwargs)
            if self.vid_writer:
                self.vid_writer.release()
                # Use the temp video path (e.g., .avi) that was actually created
                self.copy_audio(self.temp_video_path)
            print(f'✅ Process complete for media: {self.input_path}')
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # cv2.waitKey(0)
        # print(results[0].boxes.data)

    def validate_bbox(self, bbox, img_shape):
        """
        Validate and clamp bounding box to image boundaries.

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            img_shape: Image shape (height, width, channels)

        Returns:
            Valid bbox [x1, y1, x2, y2] or None if invalid
        """
        if bbox is None or len(bbox) < 4:
            return None

        height, width = img_shape[:2]

        # Clamp coordinates to image boundaries
        x1 = max(0, min(bbox[0], width))
        y1 = max(0, min(bbox[1], height))
        x2 = max(0, min(bbox[2], width))
        y2 = max(0, min(bbox[3], height))

        # Ensure x2 > x1 and y2 > y1 (positive dimensions)
        if x2 <= x1 or y2 <= y1:
            return None

        # Ensure minimum size (at least 5x5 pixels)
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            return None

        return [x1, y1, x2, y2]

    def interpolate_bbox(self, bbox1, bbox2, ratio):
        """
        Linear interpolation between two bounding boxes.

        Args:
            bbox1: First bbox [x1, y1, x2, y2]
            bbox2: Second bbox [x1, y1, x2, y2]
            ratio: Interpolation ratio (0 = bbox1, 1 = bbox2)

        Returns:
            Interpolated bbox [x1, y1, x2, y2]
        """
        return [
            bbox1[0] + (bbox2[0] - bbox1[0]) * ratio,
            bbox1[1] + (bbox2[1] - bbox1[1]) * ratio,
            bbox1[2] + (bbox2[2] - bbox1[2]) * ratio,
            bbox1[3] + (bbox2[3] - bbox1[3]) * ratio,
        ]

    def get_interpolated_detections(self, frame_idx, track_id):
        """
        Get interpolated or extrapolated detection for a missing frame.

        Args:
            frame_idx: Current frame index
            track_id: ID of the tracked object

        Returns:
            (bbox, label) if interpolation is possible, None otherwise
        """
        if track_id not in self.detection_buffer:
            return None

        detections = self.detection_buffer[track_id]
        if len(detections) < 2:
            return None

        # Find the two nearest detections (before and after current frame)
        before = None
        after = None

        for det_frame, bbox, label in detections:
            if det_frame < frame_idx:
                if before is None or det_frame > before[0]:
                    before = (det_frame, bbox, label)
            elif det_frame > frame_idx:
                if after is None or det_frame < after[0]:
                    after = (det_frame, bbox, label)

        # Interpolation: we have detections before AND after
        if before and after:
            before_frame, before_bbox, before_label = before
            after_frame, after_bbox, after_label = after

            # Check if gap is not too large
            gap = after_frame - before_frame
            if gap > self.max_interpolation_frames * 2:
                return None

            # Linear interpolation
            ratio = (frame_idx - before_frame) / gap
            interpolated_bbox = self.interpolate_bbox(before_bbox, after_bbox, ratio)
            return (interpolated_bbox, before_label)

        # Extrapolation: we only have detections before (forward extrapolation)
        elif before and not after:
            before_frame, before_bbox, before_label = before
            gap = frame_idx - before_frame

            # Only extrapolate for a limited number of frames
            if gap > self.max_interpolation_frames:
                return None

            # If we have at least 2 previous detections, use velocity estimation
            if len(detections) >= 2:
                # Get the two most recent detections
                sorted_dets = sorted(detections, key=lambda x: x[0], reverse=True)
                latest = sorted_dets[0]
                previous = sorted_dets[1]

                latest_frame, latest_bbox, _ = latest
                prev_frame, prev_bbox, _ = previous

                # Estimate velocity
                frame_diff = latest_frame - prev_frame
                if frame_diff > 0 and frame_diff <= self.max_interpolation_frames:
                    velocity = [
                        (latest_bbox[i] - prev_bbox[i]) / frame_diff
                        for i in range(4)
                    ]

                    # Extrapolate using constant velocity
                    extrapolated_bbox = [
                        before_bbox[i] + velocity[i] * gap
                        for i in range(4)
                    ]
                    return (extrapolated_bbox, before_label)

            # Fallback: use last known position
            return (before_bbox, before_label)

        return None

    def collect_all_detections(self, classes2blur, detection_threshold, use_segmentation):
        """
        PASS 1: Collect all detections from all frames.

        Returns:
            dict: {track_id: [(frame_idx, bbox, label, mask), ...]}
        """
        detection_buffer = {}
        frame_idx = 0

        print("[Interpolation] Pass 1/2: Collecting all detections...")

        for result in self.results:
            if not classes2blur or not result.boxes:
                frame_idx += 1
                continue

            # Process detections in this frame
            if use_segmentation and hasattr(result, 'masks') and result.masks is not None:
                for i, d in enumerate(result.boxes):
                    label = result.names[int(d.cls)]
                    if label not in classes2blur or float(d.conf) < detection_threshold:
                        continue

                    track_id = int(d.id) if hasattr(d, 'id') and d.id is not None else f"det_{i}"
                    bbox = d.xyxy[0].cpu().numpy().tolist()

                    # Get mask if available
                    mask = None
                    if i < len(result.masks.data):
                        mask = result.masks.data[i].cpu().numpy()
                        mask = (mask * 255).astype(np.uint8)

                    if track_id not in detection_buffer:
                        detection_buffer[track_id] = []
                    detection_buffer[track_id].append((frame_idx, bbox, label, mask))
            else:
                for i, d in enumerate(result.boxes):
                    label = result.names[int(d.cls)]
                    if label not in classes2blur or float(d.conf) < detection_threshold:
                        continue

                    track_id = int(d.id) if hasattr(d, 'id') and d.id is not None else f"det_{i}"
                    bbox = d.xyxy[0].cpu().numpy().tolist()

                    if track_id not in detection_buffer:
                        detection_buffer[track_id] = []
                    detection_buffer[track_id].append((frame_idx, bbox, label, None))

            frame_idx += 1

        return detection_buffer

    def fill_detection_gaps(self, detection_buffer, max_gap):
        """
        Identify gaps in detections and fill them with interpolated positions.
        Only interpolates BETWEEN two known detections, never extrapolates.

        Args:
            detection_buffer: {track_id: [(frame_idx, bbox, label, mask), ...]}
            max_gap: Maximum gap size to interpolate (in frames)

        Returns:
            dict: {frame_idx: {track_id: (bbox, label)}}
        """
        interpolated_detections = {}

        print(f"[Interpolation] Pass 2/2: Filling detection gaps (max gap: {max_gap} frames)...")

        for track_id, detections in detection_buffer.items():
            if len(detections) < 2:
                # Need at least 2 detections to interpolate
                continue

            # Sort by frame index
            detections.sort(key=lambda x: x[0])

            # Check for gaps between consecutive detections
            for i in range(len(detections) - 1):
                frame_start, bbox_start, label_start, _ = detections[i]
                frame_end, bbox_end, label_end, _ = detections[i + 1]

                gap = frame_end - frame_start - 1

                if gap > 0 and gap <= max_gap:
                    # Interpolate between these two detections
                    print(f"[Interpolation] Track {track_id}: filling {gap} frames between frame {frame_start} and {frame_end}")

                    for frame_idx in range(frame_start + 1, frame_end):
                        # Linear interpolation
                        ratio = (frame_idx - frame_start) / (frame_end - frame_start)
                        interpolated_bbox = self.interpolate_bbox(bbox_start, bbox_end, ratio)

                        # Store interpolated detection
                        if frame_idx not in interpolated_detections:
                            interpolated_detections[frame_idx] = {}
                        interpolated_detections[frame_idx][track_id] = (interpolated_bbox, label_start)

        return interpolated_detections

    def blur_results(self, **kwargs):

        # Settings
        plot_args = {'line_width': None, 'boxes': False, 'conf': False, 'labels': False}
        classes2blur = kwargs.get('classes2blur', self.classes2blur)
        blur_ratio = normalize_blur_ratio(kwargs.get('blur_ratio', self.blur_ratio))
        rounded_edges = int(kwargs.get('rounded_edges', self.rounded_edges))  # Rounding corners
        progressive_blur = int(kwargs.get('progressive_blur', self.progressive_blur))  # Progressive contours
        roi_enlargement = kwargs.get('ROI_enlargement', self.ROI_enlargement)  # Enlarging the blurred area
        detection_threshold = kwargs.get('detection_threshold', self.conf)  # Object detection threshold
        interpolate_detections = kwargs.get('interpolate_detections', self.interpolate_detections)

        # Calculate max interpolation frames based on FPS (0.5 seconds max)
        fps = self.meta_data.get('fps', 30) if isinstance(self.meta_data, dict) else 30
        max_interpolation_time = 0.5  # seconds
        calculated_max_frames = int(fps * max_interpolation_time)

        # Use the smaller of: user setting or calculated limit (0.5s)
        user_max_frames = kwargs.get('max_interpolation_frames', self.max_interpolation_frames)
        max_gap = min(user_max_frames, calculated_max_frames)

        print(f"[Interpolation] FPS: {fps}, Max gap to fill: {max_gap} frames ({max_gap/fps:.2f}s)")

        # Check if we're using segmentation
        use_segmentation = self._is_segmentation_model if hasattr(self, '_is_segmentation_model') else False

        # TWO-PASS APPROACH for interpolation
        interpolated_detections = {}
        if interpolate_detections:
            # PASS 1: Collect all detections
            detection_buffer = self.collect_all_detections(classes2blur, detection_threshold, use_segmentation)

            # PASS 2: Fill gaps with interpolation
            interpolated_detections = self.fill_detection_gaps(detection_buffer, max_gap)

            print(f"[Interpolation] Generated {sum(len(v) for v in interpolated_detections.values())} interpolated detections across {len(interpolated_detections)} frames")

        # MAIN BLURRING LOOP
        frame_idx = 0
        for result in tqdm(self.results, desc='Blurring media', unit='frames', dynamic_ncols=True):  # Loop on images
            im0 = result.plot(**plot_args)

            if classes2blur and result.boxes:
                # Use segmentation masks if available
                if use_segmentation and hasattr(result, 'masks') and result.masks is not None:
                    for i, d in enumerate(result.boxes):
                        label = result.names[int(d.cls)]
                        if label not in classes2blur or float(d.conf) < detection_threshold:
                            continue

                        # Get segmentation mask for this detection
                        if i < len(result.masks.data):
                            mask = result.masks.data[i].cpu().numpy()
                            # Convert mask to uint8 (0-255)
                            mask = (mask * 255).astype(np.uint8)

                            # Apply segmentation-based blur
                            im0 = blur_segmentation(im0, mask, blur_ratio, progressive_blur)
                else:
                    # Use bounding box-based blur (original method)
                    for d in result.boxes:
                        label = result.names[int(d.cls)]
                        if label not in classes2blur or float(d.conf) < detection_threshold:
                            continue

                        # Blur this detection using the utility function
                        im0 = blur_detection(
                            im0,
                            d.xyxy[0],
                            label,
                            blur_ratio,
                            rounded_edges,
                            progressive_blur,
                            roi_enlargement
                        )

            # Apply ONLY pre-calculated interpolated detections for this frame
            if interpolate_detections and frame_idx in interpolated_detections:
                for track_id, (bbox, label) in interpolated_detections[frame_idx].items():
                    # Validate bbox before blurring
                    validated_bbox = self.validate_bbox(bbox, im0.shape)
                    if validated_bbox is None:
                        continue  # Skip invalid bbox

                    # Blur the interpolated detection
                    try:
                        im0 = blur_detection(
                            im0,
                            validated_bbox,
                            label,
                            blur_ratio,
                            rounded_edges,
                            progressive_blur,
                            roi_enlargement
                        )
                    except Exception as e:
                        print(f"[Interpolation] Error blurring interpolated bbox at frame {frame_idx}: {e}")
                        continue

            self.plotted_img = im0
            self.write_media()
            frame_idx += 1

    def write_media(self):
        if not isinstance(self.meta_data, dict):
            self.meta_data = {}

        if 'fps' not in self.meta_data:
            self.meta_data['fps'] = 1
            # Save image with high quality
            # For JPEG: quality 95 (default is 95, max is 100)
            # For PNG: compression level 3 (default is 3, 0=no compression, 9=max compression)
            ext = os.path.splitext(self.output_path)[1].lower()
            if ext in ['.jpg', '.jpeg']:
                cv2.imwrite(self.output_path, self.plotted_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif ext == '.png':
                cv2.imwrite(self.output_path, self.plotted_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            else:
                cv2.imwrite(self.output_path, self.plotted_img)
        else:
            self.vid_writer.write(self.plotted_img)

    def copy_audio(self, temp_video_path):
        """
        Copy audio from original video to processed video.
        Converts intermediate format (.avi) to final .mp4 format.
        """
        print(f"[copy_audio] Input video: {self.input_path}")
        print(f"[copy_audio] Temp video (intermediate): {temp_video_path}")
        print(f"[copy_audio] Temp video exists: {os.path.exists(temp_video_path)}")

        # Final output should always be .mp4
        final_output_path = os.path.splitext(self.output_path)[0] + '.mp4'
        print(f"[copy_audio] Final output path: {final_output_path}")

        copy_audio_to_video(self.input_path, temp_video_path, final_output_path)


def stop_process():
    print('Process stopped')
    exit()


if __name__ == '__main__':
    print('CUDA available:', torch.cuda.is_available())
    torch.cuda.empty_cache()
    gc.collect()
    model = Anonymize()
    model.load_model()
    model.process()
