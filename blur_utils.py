import cv2
import numpy as np
from .bounds import Bounds


def apply_mask_blur(im0, mask, blur_ratio, progressive_blur=0):
    """
    Apply blur using a segmentation mask.

    Args:
        im0: Input image (numpy array)
        mask: Binary segmentation mask (numpy array, same size as im0)
        blur_ratio: Blur kernel size (must be odd)
        progressive_blur: Progressive blur strength for smooth edges (0 to disable)

    Returns:
        Modified image with mask-based blur applied
    """
    # Ensure mask is binary
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Normalize blur ratio
    blur_ratio = normalize_blur_ratio(blur_ratio)

    # Create blurred version of the entire image
    blurred = cv2.GaussianBlur(im0, (blur_ratio, blur_ratio), 0)

    # Apply progressive blur to mask edges if requested
    if progressive_blur > 0:
        blur_strength = max(1, int(progressive_blur))
        if blur_strength % 2 == 0:
            blur_strength += 1
        # Smooth the mask edges
        smooth_mask = cv2.GaussianBlur(mask, (blur_strength, blur_strength), 0)
    else:
        smooth_mask = mask

    # Convert mask to 3 channels if needed
    if len(smooth_mask.shape) == 2:
        smooth_mask_3ch = cv2.merge([smooth_mask] * 3)
    else:
        smooth_mask_3ch = smooth_mask

    # Blend original and blurred images using the mask
    smooth_mask_3ch = smooth_mask_3ch.astype(np.float32) / 255.0
    result = (blurred * smooth_mask_3ch + im0 * (1 - smooth_mask_3ch)).astype(np.uint8)

    return result


def blur_segmentation(im0, segmentation_mask, blur_ratio, progressive_blur=0):
    """
    Apply blur to a segmented region.

    Args:
        im0: Input image (numpy array)
        segmentation_mask: Segmentation mask from YOLO (H x W, values 0-255)
        blur_ratio: Blur kernel size
        progressive_blur: Progressive blur strength for smooth edges

    Returns:
        Modified image with segmentation-based blur applied
    """
    if segmentation_mask is None or segmentation_mask.size == 0:
        return im0

    # Ensure mask is the same size as the image
    if segmentation_mask.shape[:2] != im0.shape[:2]:
        segmentation_mask = cv2.resize(segmentation_mask, (im0.shape[1], im0.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)

    return apply_mask_blur(im0, segmentation_mask, blur_ratio, progressive_blur)


def apply_progressive_blur(im0, bounds, blur_ratio, progressive_blur):
    """
    Apply progressive blur with elliptical gradient mask.
    The blur is stronger at the center and fades at the edges.

    Args:
        im0: Input image (numpy array)
        bounds: Bounds object defining the area to blur
        blur_ratio: Blur kernel size (must be odd)
        progressive_blur: Progressive blur strength (must be odd)

    Returns:
        Modified image with progressive blur applied
    """
    x, y = bounds.x_min, bounds.y_min
    w, h = bounds.x_max - bounds.x_min, bounds.y_max - bounds.y_min

    # Validate dimensions
    if w <= 0 or h <= 0:
        print(f"[apply_progressive_blur] Invalid dimensions: w={w}, h={h}")
        return im0

    # Area to be blurred
    blur_area = im0[y:y + h, x:x + w]

    # Check if blur area is valid
    if blur_area.size == 0:
        print(f"[apply_progressive_blur] Empty blur area")
        return im0

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
    blended = (blurred * (smooth_mask_3ch / 255.0) + blur_area * (1 - smooth_mask_3ch / 255.0)).astype(np.uint8)
    im0[y:y + h, x:x + w] = blended

    return im0


def apply_simple_blur(im0, bounds, blur_ratio):
    """
    Apply simple Gaussian blur to a region.

    Args:
        im0: Input image (numpy array)
        bounds: Bounds object defining the area to blur
        blur_ratio: Blur kernel size (must be odd)

    Returns:
        Modified image with blur applied
    """
    x, y = bounds.x_min, bounds.y_min
    w, h = bounds.x_max - bounds.x_min, bounds.y_max - bounds.y_min

    # Validate dimensions
    if w <= 0 or h <= 0:
        print(f"[apply_simple_blur] Invalid dimensions: w={w}, h={h}")
        return im0

    # Area to be blurred
    blur_area = im0[y:y + h, x:x + w]

    # Check if blur area is valid
    if blur_area.size == 0:
        print(f"[apply_simple_blur] Empty blur area")
        return im0

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(blur_area, (blur_ratio, blur_ratio), 0)
    im0[y:y + h, x:x + w] = blurred

    return im0


def normalize_blur_ratio(blur_ratio):
    """
    Ensure blur ratio is valid (positive and odd).

    Args:
        blur_ratio: Input blur ratio value

    Returns:
        Normalized blur ratio (positive odd integer)
    """
    blur_ratio = int(blur_ratio)
    if blur_ratio <= 0:
        blur_ratio = 1
    if blur_ratio % 2 == 0:
        blur_ratio += 1
    return blur_ratio


def blur_detection(im0, detection_box, label, blur_ratio, rounded_edges, progressive_blur, roi_enlargement):
    """
    Blur a single detection on the image.

    Args:
        im0: Input image (numpy array)
        detection_box: Detection bounding box (xyxy format)
        label: Class label of the detection
        blur_ratio: Blur kernel size
        rounded_edges: Amount to expand the blur area
        progressive_blur: Progressive blur strength (0 to disable)
        roi_enlargement: Scale factor for enlarging the ROI

    Returns:
        Modified image with detection blurred
    """
    # Validate detection_box
    if detection_box is None or len(detection_box) < 4:
        print(f"[blur_detection] Invalid detection_box: {detection_box}")
        return im0

    # Extract bounding box coordinates
    x, y = int(detection_box[0]), int(detection_box[1])
    w, h = int(detection_box[2]) - x, int(detection_box[3]) - y

    # Validate dimensions
    if w <= 0 or h <= 0:
        print(f"[blur_detection] Invalid dimensions: w={w}, h={h} from bbox={detection_box}")
        return im0

    # Ensure coordinates are within image bounds
    img_h, img_w = im0.shape[:2]
    if x < 0 or y < 0 or x >= img_w or y >= img_h:
        print(f"[blur_detection] Coordinates out of bounds: x={x}, y={y} for image {img_w}x{img_h}")
        # Clamp to valid range
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)

    # Final dimension check
    if w <= 0 or h <= 0:
        print(f"[blur_detection] Dimensions still invalid after clamping: w={w}, h={h}")
        return im0

    # Create bounds and apply transformations
    try:
        bounds = Bounds(x, y, x + w, y + h).scale(im0.shape, roi_enlargement).expand(im0.shape, rounded_edges)
    except Exception as e:
        print(f"[blur_detection] Error creating bounds: {e}")
        return im0

    # Apply appropriate blur type
    try:
        if label in ['face', 'person'] and progressive_blur > 0:
            im0 = apply_progressive_blur(im0, bounds, blur_ratio, progressive_blur)
        else:
            im0 = apply_simple_blur(im0, bounds, blur_ratio)
    except Exception as e:
        print(f"[blur_detection] Error applying blur: {e}")
        return im0

    return im0
