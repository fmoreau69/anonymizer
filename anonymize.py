import os
import gc
import cv2
import torch
import shutil
import mimetypes
import numpy as np
import subprocess as sp

from tqdm import tqdm
from shutil import which
from pathlib import Path

from .bounds import Bounds
from ultralytics import YOLO, settings
from ultralytics.utils import MACOS, WINDOWS

mimetypes.init()


class Anonymize:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Path settings
        self.source, self.destination = './media/input_media', './media/output_media'
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

    def load_model(self, **kwargs):
        self.model_name = 'yolov8n-seg.pt' if self.task == 'segment' else "yolov8n.pt"
        if any([classe in self.classes2blur for classe in ['face', 'plate']]):
            self.model_name = "yolov8m_faces&plates_720p.pt"  # "yolov8m_faces&plates_1080p.pt"
        self.model_path = kwargs.get('model_path', os.path.join(self.models_dir, self.model_name))
        print(f'Model used: {self.model_path}')
        self.model = YOLO(self.model_path)
        self.class_list = list(self.model.model.names.values())

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

                if self.is_image(media_path):
                    self.process_image(media_path, self.output_path, **kwargs)
                else:
                    self.setup_source(**kwargs)
                    self.apply_process(**kwargs)
        # TODO: File list
        # File
        else:
            self.output_path = self.input_path[:-4].replace('input', 'output') + '_blurred' + self.input_path[-4:]

            if self.is_image(self.input_path):
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


    def is_image(self, path):
        mime_type, _ = mimetypes.guess_type(path)
        return mime_type and mime_type.startswith("image")


    def setup_source(self, **kwargs):
        print(f'Setting up media: {self.input_path}')
        cap = cv2.VideoCapture(self.input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        suffix, fourcc = ('.mp4', 'avc1') if MACOS else ('.mp4', 'mp4v')
        save_path = str(Path(self.output_path).with_suffix(suffix))
        self.meta_data = {'fps': fps, 'size': (width, height)}
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
                self.copy_audio(self.output_path)
            print(f'✅ Process complete for media: {self.input_path}')
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # cv2.waitKey(0)
        # print(results[0].boxes.data)

    def blur_results(self, **kwargs):

        # Settings
        plot_args = {'line_width': None, 'boxes': False, 'conf': False, 'labels': False}
        classes2blur = kwargs.get('classes2blur', self.classes2blur)
        blur_ratio = int(kwargs.get('blur_ratio', self.blur_ratio))  # Amount of blur
        if blur_ratio <= 0:
            blur_ratio = 1
        if blur_ratio % 2 == 0:
            blur_ratio += 1
        rounded_edges = int(kwargs.get('rounded_edges', self.rounded_edges))  # Rounding corners
        progressive_blur = int(kwargs.get('progressive_blur', self.progressive_blur))  # Progressive contours
        roi_enlargement = kwargs.get('ROI_enlargement', self.ROI_enlargement)  # Enlarging the blurred area
        detection_threshold = kwargs.get('detection_threshold', self.conf)  # Object detection threshold

        for result in tqdm(self.results, desc='Blurring media', unit='frames', dynamic_ncols=True):  # Loop on images
            im0 = result.plot(**plot_args)

            if not classes2blur or not result.boxes:
                self.plotted_img = im0
                self.write_media()
                continue

            for d in result.boxes:
                label = result.names[int(d.cls)]
                if label not in classes2blur or float(d.conf) < detection_threshold:
                    continue

                x, y = int(d.xyxy[0][0]), int(d.xyxy[0][1])
                w, h = int(d.xyxy[0][2]) - x, int(d.xyxy[0][3]) - y
                bounds = Bounds(x, y, x+w, y+h).scale(im0.shape, roi_enlargement).expand(im0.shape, rounded_edges)
                x, y, w, h = bounds.x_min, bounds.y_min, bounds.x_max - bounds.x_min, bounds.y_max - bounds.y_min

                # Area to be blurred
                blur_area = im0[y:y + h, x:x + w]

                if label in ['face', 'person'] and progressive_blur > 0:
                    # Progressive mask (blurred gradient at center)
                    mask = np.zeros((h, w), dtype=np.uint8)
                    center, axes = (w // 2, h // 2), (w // 2, h // 2)
                    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

                    # Apply an additional blur to the mask to make it progressive
                    blur_strength = max(1, progressive_blur)
                    if blur_strength % 2 == 0:
                        blur_strength += 1
                    smooth_mask = cv2.GaussianBlur(mask, (blur_strength, blur_strength), 0)

                    # Convert smoothed mask into 3 channels
                    smooth_mask_3ch = cv2.merge([smooth_mask] * 3)

                    # Blur the image in the relevant area
                    blurred = cv2.GaussianBlur(blur_area, (blur_ratio, blur_ratio), 0)

                    # Apply the progressive mask
                    blended = (blurred * (smooth_mask_3ch / 255.0) + blur_area * (1 - smooth_mask_3ch / 255.0)).astype(
                        np.uint8)
                    im0[y:y + h, x:x + w] = blended

            self.plotted_img = im0
            self.write_media()

    def write_media(self):
        if not isinstance(self.meta_data, dict):
            self.meta_data = {}

        if 'fps' not in self.meta_data:
            self.meta_data['fps'] = 1
            cv2.imwrite(self.output_path, self.plotted_img)
        else:
            self.vid_writer.write(self.plotted_img)

    def copy_audio(self, temp_video_path):
        ffmpeg_exe = which("ffmpeg") or os.getenv("FFMPEG_BINARY")
        if not ffmpeg_exe:
            print("❌ FFMPEG not found. Skipping audio copy.")
            return

        # Ensure the existence of a discharge file
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        if not os.path.isfile(temp_video_path):
            print(f"❌ Temp video file not found: {temp_video_path}")
            return

        if not os.path.isfile(self.input_path):
            print(f"❌ Original input file not found: {self.input_path}")
            return

        # Temporary output file
        final_output = self.output_path
        temp_output_with_audio = final_output.replace(".mp4", "_with_audio.mp4")

        command = [
            ffmpeg_exe, "-y",
            "-i", temp_video_path,  # Edited video without audio
            "-i", self.input_path,  # Original video with audio
            "-map", "0:v",  # Video of the blurred version
            "-map", "1:a?",  # Audio of original video (optional, ? avoids error if there is no audio)
            "-c:v", "copy",
            "-c:a", "copy",
            "-shortest",
            temp_output_with_audio
        ]

        result = sp.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            print("❌ Error during audio copy:")
            print(result.stderr)
            return

        # Replace old output with audio output
        shutil.move(temp_output_with_audio, final_output)
        print(f"✅ Audio copied and merged into: {final_output}")


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
