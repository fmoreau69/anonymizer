# YOLO Models Directory Structure

## Overview
This directory contains YOLO models organized by task type for better organization and easier model selection.

## Directory Structure

```
models/
├── detect/           # Object detection models
├── segment/          # Instance segmentation models
├── classify/         # Image classification models
├── pose/             # Pose estimation models
├── obb/              # Oriented bounding box detection models
└── *.pt              # Legacy models (root directory, for backward compatibility)
```

## Model Types

### Detection (`detect/`)
Models for detecting objects and drawing bounding boxes around them.
- General object detection (COCO classes)
- Specialized detection (faces & license plates)

**Current models:**
- `yolov8n.pt` - Nano model (fastest, smallest)
- `yolov8s_faces&plates_*.pt` - Small models for face & plate detection
- `yolov8m_faces&plates_*.pt` - Medium models for face & plate detection
- `yolov8n_faces&plates_*.pt` - Nano models for face & plate detection
- `yolov9t-face-lindevs.pt` - YOLOv9 tiny for face detection
- `yolov9s-face-lindevs.pt` - YOLOv9 small for face detection

### Segmentation (`segment/`)
Models for pixel-level segmentation of objects.

**Current models:**
- `yolov8n-seg.pt` - Nano segmentation model

### Classification (`classify/`)
Models for image classification tasks.
*Currently empty - add your classification models here*

### Pose (`pose/`)
Models for human pose estimation and keypoint detection.
*Currently empty - add your pose estimation models here*

### OBB (`obb/`)
Models for oriented bounding box detection (rotated objects).
*Currently empty - add your OBB models here*

## Usage

### From Django App

The `yolo_utils.py` module automatically handles model path resolution:

```python
from wama.anonymizer.utils.yolo_utils import get_model_path

# Old format (still works - backward compatible)
path = get_model_path('yolov8n.pt')

# New format (with category)
path = get_model_path('detect/yolov8n.pt')

# Auto-detection (searches in subdirectories)
path = get_model_path('yolov8n-seg.pt')  # Will find in segment/
```

### Model Selection in Forms

The UserSettings form now displays models grouped by type:

```
┌─ Detection
│  ├─ yolov8n.pt
│  ├─ yolov8m_faces&plates_720p.pt
│  └─ ...
├─ Segmentation
│  └─ yolov8n-seg.pt
└─ Legacy (Root Directory)
   └─ ... (backward compatibility)
```

## Adding New Models

1. Download your YOLO model (`.pt` file)
2. Identify the model type (detect, segment, classify, pose, or obb)
3. Place the model in the appropriate subdirectory:
   ```bash
   # Example: Adding a new detection model
   cp my_custom_model.pt anonymizer/models/detect/

   # Example: Adding a new segmentation model
   cp my_seg_model.pt anonymizer/models/segment/
   ```
4. The model will automatically appear in the dropdown menu grouped by type

## Backward Compatibility

Models in the root directory are preserved for backward compatibility:
- Existing code referencing models by filename only will continue to work
- The system searches both root and subdirectories
- Root models appear under "Legacy (Root Directory)" in the UI

## Migration Notes

All models have been **copied** (not moved) to subdirectories to ensure no existing references are broken. The original files remain in the root directory.

To complete the migration:
1. Verify all systems are working with the new structure
2. Update any hardcoded model references to use the new format
3. Optionally remove duplicate models from the root directory once confident

## Model Naming Conventions

For clarity, we recommend:
- Detection models: `yolov{version}{size}[_specialty][_resolution].pt`
  - Example: `yolov8m_faces&plates_720p.pt`
- Segmentation models: `yolov{version}{size}-seg.pt`
  - Example: `yolov8n-seg.pt`
- Pose models: `yolov{version}{size}-pose.pt`
- Classification models: `yolov{version}{size}-cls.pt`

## API Reference

### `get_model_path(filename: str) -> str`
Returns absolute path to a model file. Searches root first, then subdirectories.

### `list_models_by_type() -> Dict[str, List[str]]`
Returns dictionary mapping model type to list of model filenames.

### `get_model_choices_grouped() -> List[Tuple[str, List[Tuple[str, str]]]]`
Returns grouped choices for Django forms with optgroups.
