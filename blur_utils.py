import cv2
import numpy as np
from .bounds import Bounds


def apply_mask_blur(im0, mask, blur_ratio, progressive_blur=0):
    """
    Apply blur using a segmentation mask.

    Args:
        im0: Input image (numpy array BGR)
        mask: Binary segmentation mask (numpy array, same size as im0, values 0-255)
        blur_ratio: Blur kernel size (must be odd)
        progressive_blur: Progressive blur strength for smooth edges (0 to disable)

    Returns:
        Modified image with mask-based blur applied
    """
    # Ensure mask is uint8
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Ensure mask is 2D
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]

    # Normalize blur ratio
    blur_ratio = normalize_blur_ratio(blur_ratio)

    # Create blurred version of the entire image
    blurred = cv2.GaussianBlur(im0, (blur_ratio, blur_ratio), 0)

    # Apply progressive blur to mask edges if requested
    if progressive_blur > 0:
        blur_strength = max(3, int(progressive_blur))
        if blur_strength % 2 == 0:
            blur_strength += 1
        # Smooth the mask edges for gradual transition
        smooth_mask = cv2.GaussianBlur(mask, (blur_strength, blur_strength), 0)
    else:
        smooth_mask = mask

    # Normalize mask to float [0, 1]
    alpha = smooth_mask.astype(np.float32) / 255.0

    # Debug: Check alpha values
    alpha_max = alpha.max()
    alpha_mean = alpha[alpha > 0].mean() if np.any(alpha > 0) else 0
    if alpha_max < 0.5:
        print(f"[apply_mask_blur] Warning: alpha values are low - max={alpha_max:.3f}, mean={alpha_mean:.3f}")

    # Expand alpha to 3 channels for proper blending
    alpha_3ch = np.dstack([alpha, alpha, alpha])

    # Convert images to float for proper blending
    im0_float = im0.astype(np.float32)
    blurred_float = blurred.astype(np.float32)

    # Blend: where mask is 1 (white) use blurred, where 0 (black) use original
    result = (blurred_float * alpha_3ch + im0_float * (1.0 - alpha_3ch))

    # Clip values to valid range and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Debug: Verify blur was actually applied by comparing pixel differences in masked area
    mask_binary = mask > 127
    if np.any(mask_binary):
        diff = np.abs(result.astype(np.float32) - im0.astype(np.float32))
        masked_diff = diff[mask_binary]
        avg_diff = masked_diff.mean() if masked_diff.size > 0 else 0
        if avg_diff < 1.0:
            print(f"[apply_mask_blur] Warning: blur effect is minimal - avg pixel change={avg_diff:.2f}")

    return result


def blur_segmentation(im0, segmentation_mask, blur_ratio, progressive_blur=0):
    """
    Apply blur to a segmented region.

    Args:
        im0: Input image (numpy array BGR)
        segmentation_mask: Segmentation mask from YOLO (H x W, values 0-255 or 0-1)
        blur_ratio: Blur kernel size
        progressive_blur: Progressive blur strength for smooth edges

    Returns:
        Modified image with segmentation-based blur applied
    """
    if segmentation_mask is None or segmentation_mask.size == 0:
        print(f"[blur_segmentation] Skipping: mask is None or empty")
        return im0

    # Handle multi-dimensional masks (take first channel if needed)
    if len(segmentation_mask.shape) > 2:
        segmentation_mask = segmentation_mask[:, :, 0]

    # Normalize mask to 0-255 range
    if segmentation_mask.max() <= 1.0:
        segmentation_mask = (segmentation_mask * 255).astype(np.uint8)
    else:
        segmentation_mask = segmentation_mask.astype(np.uint8)

    # Check if mask has any non-zero values
    mask_sum = segmentation_mask.sum()
    if mask_sum == 0:
        print(f"[blur_segmentation] Warning: mask is all zeros, skipping")
        return im0

    # Ensure mask is the same size as the image
    # Use INTER_NEAREST to preserve binary mask edges (no interpolation artifacts)
    original_shape = segmentation_mask.shape
    if segmentation_mask.shape[:2] != im0.shape[:2]:
        print(f"[blur_segmentation] Resizing mask from {original_shape} to {im0.shape[:2]}")
        segmentation_mask = cv2.resize(
            segmentation_mask,
            (im0.shape[1], im0.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        # Check mask again after resize
        if segmentation_mask.sum() == 0:
            print(f"[blur_segmentation] Warning: mask became all zeros after resize "
                  f"from {original_shape} to {segmentation_mask.shape}")
            return im0

    # Calculate mask coverage
    total_pixels = segmentation_mask.shape[0] * segmentation_mask.shape[1]
    nonzero_pixels = np.count_nonzero(segmentation_mask)
    high_value_pixels = np.count_nonzero(segmentation_mask > 127)  # Pixels with significant blur
    coverage_pct = (nonzero_pixels / total_pixels) * 100
    active_pct = (high_value_pixels / total_pixels) * 100

    # Find bounding box of the mask for debugging
    nonzero_coords = np.argwhere(segmentation_mask > 0)
    if len(nonzero_coords) > 0:
        y_min, x_min = nonzero_coords.min(axis=0)
        y_max, x_max = nonzero_coords.max(axis=0)
        print(f"[blur_segmentation] Mask coverage: {coverage_pct:.2f}% ({nonzero_pixels} px), "
              f"active (>127): {active_pct:.2f}% ({high_value_pixels} px), "
              f"bbox: x={x_min}-{x_max}, y={y_min}-{y_max}")

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
