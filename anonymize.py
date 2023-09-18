import os
import gc
import cv2
import torch
import imageio
import mimetypes
import numpy as np
import subprocess as sp

from tqdm import tqdm
from shutil import which
from pathlib import Path

from bounds import Bounds
from ultralytics import YOLO, settings
from ultralytics.utils import MACOS, WINDOWS


class Anonymize:
    def __init__(self):
        # CUDA settings
        if torch.cuda.is_available():
            self.device = torch.cuda.device_count()
            print(f"Using {torch.cuda.get_device_name(torch.cuda.current_device())}.")
        else:
            self.device = "cpu"
            print("Using CPU.")

        # Path settings
        self.source = './media/input_media'
        os.mkdir(self.source) if not os.path.exists(self.source) else self.source + 'already exists'
        self.destination = './media/output_media'
        os.mkdir(self.destination) if not os.path.exists(self.destination) else self.destination + 'already exists'
        self.input_path = None
        self.output_path = None
        self.save_path = './runs'
        self.models_dir = './models'
        settings.update({'runs_dir': self.save_path, 'weights_dir': self.models_dir})

        # Model settings
        self.classes2blur = ['face', 'plate']
        # self.classes2blur = ['person', 'car', 'truck', 'bus']
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
        self.blur_ratio = 20
        self.rounded_edges = 5
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
        self.model_path = kwargs['model_path'] if 'model_path' in kwargs else os.path.join(self.models_dir,
                                                                                           self.model_name)
        print('Model used: ' + str(self.model_path))
        self.model = YOLO(self.model_path)

    def process(self, **kwargs):
        if self.model is not None:
            if os.path.isdir(kwargs['media_path'] if 'media_path' in kwargs else self.source):
                for media in os.listdir(self.source):
                    self.input_path = os.path.join(self.source, media)
                    self.output_path = os.path.join(self.destination, media[:-4] + '_blurred' + media[-4:])
                    Anonymize.setup_source(self)
                    Anonymize.apply_process(self, **kwargs)
            else:
                self.input_path = kwargs['media_path'] if 'media_path' in kwargs else self.source
                self.output_path = self.destination
                Anonymize.setup_source(self)
                Anonymize.apply_process(self, **kwargs)
        else:
            print('No model is loaded')

    def setup_source(self):
        print('Setting up media: ' + str(self.input_path))
        with imageio.get_reader(self.input_path) as reader:
            self.meta_data = reader.get_meta_data()
            if 'fps' in self.meta_data:
                suffix, fourcc = ('.mp4', 'avc1') if MACOS else ('.avi', 'WMV2') if WINDOWS else ('.avi', 'MJPG')
                save_path = str(Path(self.output_path).with_suffix(suffix))
                self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), self.meta_data['fps'],
                                                  (self.meta_data['size'][0], self.meta_data['size'][1]))
                # self.vid_writer = dict.fromkeys(range(int(self.meta_data['duration'] * self.meta_data['fps'])))

    def apply_process(self, **kwargs):
        # Apply model
        self.results = self.model.track(
            source=(kwargs['media_path'] if 'media_path' in kwargs else self.input_path),
            # stream=True,
            task=self.task,
            mode=self.mode,
            device=self.device,
            retina_masks=self.ret_mask,
            imgsz=self.meta_data['size'][0] if 'size' in self.meta_data else self.meta_data['shape'][0],
            save=self.save,
            save_txt=self.save_txt,
            # classes=self.classes2blur,
            conf=(kwargs['detection_threshold'] if 'detection_threshold' in kwargs else self.conf),
            show=(kwargs['show_preview'] if 'show_preview' in kwargs else self.show),
            boxes=(kwargs['show_boxes'] if 'show_boxes' in kwargs else self.boxes),
            show_labels=(kwargs['show_labels'] if 'show_labels' in kwargs else self.show_labels),
            show_conf=(kwargs['show_conf'] if 'show_conf' in kwargs else self.show_conf)
        )
        # Blur detections
        if self.classes2blur:
            Anonymize.blur_results(self, **kwargs)
            if 'fps' in self.meta_data:
                self.vid_writer.release()
            print('Process complete for media: ' + str(self.input_path))
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # cv2.waitKey(0)
        # print(results[0].boxes.data)

    def blur_results(self, **kwargs):
        plot_args = {'line_width': None, 'boxes': False, 'conf': False, 'labels': False}
        classes2blur = (kwargs['classes2blur'] if 'classes2blur' in kwargs else self.classes2blur)
        blur_ratio = int(kwargs['blur_ratio'] * 100 if 'blur_ratio' in kwargs else self.blur_ratio)
        rounded_edges = int(kwargs['rounded_edges'] if 'rounded_edges' in kwargs else self.rounded_edges)
        roi_enlargement = (kwargs['ROI_enlargement'] if 'ROI_enlargement' in kwargs else self.ROI_enlargement)
        detection_threshold = (kwargs['detection_threshold'] if 'detection_threshold' in kwargs else self.conf)
        for result in tqdm(self.results, desc='Blurring media', unit='frames', dynamic_ncols=True):  # Loop on images
            im0 = result.plot(**plot_args)
            if len(classes2blur) and len(result.boxes):
                for d in result.boxes:  # Loop on boxes
                    label = result.names[int(d.cls)]
                    if label in classes2blur and float(d.conf) >= detection_threshold:
                        bounds = (Bounds(int(d.xyxy[0][0]), int(d.xyxy[0][1]), int(d.xyxy[0][2]), int(d.xyxy[0][3])).
                                  scale(im0.shape, roi_enlargement).expand(im0.shape, rounded_edges))
                        crop_obj = im0[bounds.y_min:bounds.y_max, bounds.x_min:bounds.x_max]
                        blur_area = np.full((bounds.y_max - bounds.y_min, bounds.x_max - bounds.x_min, 3), 0,
                                            dtype=np.float64)
                        # blur_area = crop_obj
                        if label == 'face' or label == 'person':
                            center, axes = bounds.ellipse_coordinates()
                            blur_area = cv2.ellipse(blur_area, center, axes, 0, 0, 360, color=[1, 1, 1], thickness=-1)
                        else:
                            blur_area = cv2.rectangle(blur_area, bounds.pt1(), bounds.pt2(), color=[1, 1, 1],
                                                      thickness=-1)
                        if rounded_edges > 0:
                            blur_area = cv2.blur(blur_area, (rounded_edges, rounded_edges), 0)
                        clear_area = np.full((bounds.y_max - bounds.y_min, bounds.x_max - bounds.x_min, 3), 1,
                                             dtype=np.float64) - blur_area
                        blurred_box = cv2.blur(crop_obj, (blur_ratio, blur_ratio), 0)
                        im0[bounds.y_min:bounds.y_max,
                        bounds.x_min:bounds.x_max] = blurred_box  # clear_area * im0 + blur_area * blurred_box
            self.plotted_img = im0.astype(np.uint8)
            Anonymize.write_media(self)

    def write_media(self):
        im0 = self.plotted_img
        if 'fps' not in self.meta_data:  # 'image'
            cv2.imwrite(self.output_path, im0)
        else:  # 'video' or 'stream'
            self.vid_writer.write(im0)

    def copy_audio(self):
        # copy over audio stream from original video to edited video
        if which("ffmpeg") is not None:
            ffmpeg_exe = "ffmpeg"
        else:
            ffmpeg_exe = os.getenv("FFMPEG_BINARY")
            if not ffmpeg_exe:
                print("FFMPEG could not be found! "
                      "Please make sure the ffmpeg.exe is available under the environment variable 'FFMPEG_BINARY'.")
                return
        if 'audio_codec' in self.meta_data:
            sp.run([ffmpeg_exe, "-y", "-i", './media/temp', "-i", self.input_path, "-c", "copy",
                    "-map", "0:0", "-map", "1:1", "-shortest", self.output_path, ], stdout=sp.DEVNULL, )


def stop_process():
    print('Process stopped')
    exit()


# def free_gpu_cache():
#     print("Initial GPU Usage")
#     gpu_usage()
#     torch.cuda.empty_cache()
#     cuda.select_device(0)
#     cuda.close()
#     cuda.select_device(0)
#     print("GPU Usage after emptying the cache")
#     gpu_usage()


if __name__ == '__main__':
    print('CUDA is currently available: ' + str(torch.cuda.is_available()))
    torch.cuda.empty_cache()
    gc.collect()
    # free_gpu_cache()
    model = Anonymize()
    Anonymize.load_model(model)
    Anonymize.process(model)
