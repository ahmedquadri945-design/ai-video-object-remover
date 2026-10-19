"""
Mask Processor Module for Video Object Removal.

Provides tools for:
- Decoding canvas brush strokes and base64 masks into binary numpy arrays.
- Morphological dilation to eliminate boundary color bleeding and edge halos.
- Contour smoothing and temporal consistency stabilization across video frames.
"""

from typing import List, Optional, Tuple, Union, Dict, Any
import base64
import io
import numpy as np
import cv2


class MaskProcessor:
    """
    Handles mask generation from UI interactions (brush strokes, polygon points, base64 strings),
    dilation to prevent edge artifacts, and temporal smoothing across video sequences.
    """

    def __init__(
        self,
        default_dilation_radius: int = 5,
        default_blur_kernel: int = 3,
        temporal_window_size: int = 3,
    ) -> None:
        """
        Initialize MaskProcessor.

        Args:
            default_dilation_radius: Radius (in pixels) for morphological dilation.
            default_blur_kernel: Kernel size for edge smoothing blur (must be odd).
            temporal_window_size: Number of adjacent frames to consider for temporal stabilization.
        """
        self.default_dilation_radius = max(0, default_dilation_radius)
        self.default_blur_kernel = default_blur_kernel if default_blur_kernel % 2 != 0 else default_blur_kernel + 1
        self.temporal_window_size = temporal_window_size

    # =========================================================================
    # 1. Base64 & Canvas Decoding
    # =========================================================================

    @staticmethod
    def base64_to_mask(
        base64_str: str,
        target_shape: Optional[Tuple[int, int]] = None,
        threshold: int = 127,
        use_alpha: bool = True,
    ) -> np.ndarray:
        """
        Decode a base64 encoded image string (e.g. from an HTML5 canvas) into a binary mask.

        Args:
            base64_str: Base64 string, optionally prefixed with 'data:image/...;base64,'.
            target_shape: Optional (height, width) to resize the mask to.
            threshold: Pixel intensity threshold (0-255) for binarization.
            use_alpha: If True and the image has an alpha channel, use alpha channel as mask.

        Returns:
            np.ndarray: Binary mask uint8 array (values 0 or 255), shape (H, W).
        """
        # Strip data URL prefix if present
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]

        image_data = base64.b64decode(base64_str.strip())
        nparr = np.frombuffer(image_data, np.uint8)
        decoded = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if decoded is None:
            raise ValueError("Failed to decode base64 image data")

        if decoded.ndim == 2:
            mask = decoded
        elif decoded.ndim == 3:
            if decoded.shape[2] == 4 and use_alpha:
                mask = decoded[:, :, 3]
            elif decoded.shape[2] == 3:
                mask = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
            else:
                mask = decoded[:, :, 0]
        else:
            raise ValueError(f"Unsupported image dimensions: {decoded.shape}")

        # Binarize
        _, binary_mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)

        # Resize if target_shape is provided
        if target_shape is not None:
            target_h, target_w = target_shape
            if binary_mask.shape[:2] != (target_h, target_w):
                binary_mask = cv2.resize(
                    binary_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
                )

        return binary_mask.astype(np.uint8)

    @staticmethod
    def mask_to_base64(mask: np.ndarray, format: str = "PNG") -> str:
        """
        Encode a binary mask uint8 array to a base64 string.

        Args:
            mask: Binary mask numpy array (H, W).
            format: Target image format ('PNG', 'JPEG').

        Returns:
            str: Data URL base64 string.
        """
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255

        ext = f".{format.lower()}"
        success, buffer = cv2.imencode(ext, mask)
        if not success:
            raise ValueError(f"Failed to encode mask to {format}")

        encoded = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/{format.lower()};base64,{encoded}"

    # =========================================================================
    # 2. Vector Strokes & Polygon Rasterization
    # =========================================================================

    @staticmethod
    def rasterize_strokes(
        height: int,
        width: int,
        strokes: List[List[Tuple[float, float]]],
        brush_radius: int = 15,
    ) -> np.ndarray:
        """
        Rasterize brush stroke paths into a binary mask.

        Args:
            height: Canvas height.
            width: Canvas width.
            strokes: List of paths, where each path is a list of (x, y) coordinates.
            brush_radius: Radius of the brush stroke in pixels.

        Returns:
            np.ndarray: Binary mask uint8 array (values 0 or 255), shape (H, W).
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        thickness = max(1, brush_radius * 2)

        for stroke in strokes:
            if not stroke:
                continue

            pts = np.array(stroke, dtype=np.int32)
            if len(pts) == 1:
                # Single dot
                cv2.circle(mask, tuple(pts[0]), brush_radius, 255, -1)
            else:
                # Draw lines between consecutive points
                for i in range(len(pts) - 1):
                    cv2.line(
                        mask,
                        tuple(pts[i]),
                        tuple(pts[i + 1]),
                        255,
                        thickness=thickness,
                        lineType=cv2.LINE_AA,
                    )
                # Place filled circles at joints to ensure smooth stroke round caps
                for pt in pts:
                    cv2.circle(mask, tuple(pt), brush_radius, 255, -1)

        return mask

    @staticmethod
    def rasterize_polygon(
        height: int,
        width: int,
        polygon_points: List[Tuple[float, float]],
    ) -> np.ndarray:
        """
        Rasterize a closed polygon into a binary mask.

        Args:
            height: Canvas height.
            width: Canvas width.
            polygon_points: List of (x, y) coordinates defining polygon vertices.

        Returns:
            np.ndarray: Binary mask uint8 array (0 or 255), shape (H, W).
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        if len(polygon_points) < 3:
            return mask

        pts = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
        return mask

    @staticmethod
    def rasterize_bounding_box(
        height: int,
        width: int,
        box: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """
        Rasterize a bounding box [x_min, y_min, x_max, y_max] into a binary mask.

        Args:
            height: Canvas height.
            width: Canvas width.
            box: (x1, y1, x2, y2)

        Returns:
            np.ndarray: Binary mask uint8 array (0 or 255), shape (H, W).
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))

        mask[y1 : y2 + 1, x1 : x2 + 1] = 255
        return mask

    # =========================================================================
    # 3. Morphological Dilation (Edge Halo Prevention)
    # =========================================================================

    def dilate_mask(
        self,
        mask: np.ndarray,
        radius: Optional[int] = None,
        iterations: int = 1,
    ) -> np.ndarray:
        """
        Apply morphological dilation using an elliptical structuring element.
        Essential for video inpainting to remove boundary edge halos / bleed of the removed object.

        Args:
            mask: Binary mask numpy array (H, W).
            radius: Dilation radius in pixels. If None, uses default_dilation_radius.
            iterations: Number of dilation iterations.

        Returns:
            np.ndarray: Dilated binary mask (0 or 255).
        """
        r = self.default_dilation_radius if radius is None else radius
        if r <= 0:
            return mask.copy()

        # Elliptical kernel provides smoother, isotropic expansion than rectangular kernel
        kernel_size = 2 * r + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        dilated = cv2.dilate(mask, kernel, iterations=iterations)
        return dilated

    # =========================================================================
    # 4. Mask Smoothing & Contour Refinement
    # =========================================================================

    def smooth_mask(
        self,
        mask: np.ndarray,
        blur_kernel: Optional[int] = None,
        threshold: int = 127,
    ) -> np.ndarray:
        """
        Smooth rough edges and ragged contours of a binary mask.

        Args:
            mask: Binary mask (H, W).
            blur_kernel: Gaussian blur kernel size (odd integer). If None, uses default.
            threshold: Re-binarization threshold after blur.

        Returns:
            np.ndarray: Refined smooth binary mask.
        """
        k = self.default_blur_kernel if blur_kernel is None else blur_kernel
        if k <= 1:
            return mask.copy()
        if k % 2 == 0:
            k += 1

        blurred = cv2.GaussianBlur(mask, (k, k), 0)
        _, smoothed = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        return smoothed

    def clean_mask(
        self,
        mask: np.ndarray,
        min_contour_area: int = 50,
        fill_holes: bool = True,
    ) -> np.ndarray:
        """
        Remove small isolated noise specks and optionally fill internal holes.

        Args:
            mask: Binary mask (H, W).
            min_contour_area: Minimum area in pixels to keep an isolated mask component.
            fill_holes: Whether to fill interior holes inside detected contours.

        Returns:
            np.ndarray: Cleaned binary mask.
        """
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_CCOMP if fill_holes else cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        cleaned = np.zeros_like(mask)
        if not contours:
            return cleaned

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area >= min_contour_area:
                if fill_holes:
                    # Draw external contours filled
                    if hierarchy is not None and hierarchy[0][i][3] == -1:
                        cv2.drawContours(cleaned, [cnt], -1, 255, -1)
                else:
                    cv2.drawContours(cleaned, [cnt], -1, 255, -1)

        return cleaned

    # =========================================================================
    # 5. Temporal Mask Consistency & Stabilization
    # =========================================================================

    def temporal_stabilize_masks(
        self,
        masks: List[np.ndarray],
        window_size: Optional[int] = None,
        threshold: float = 0.5,
    ) -> List[np.ndarray]:
        """
        Stabilize a sequence of per-frame masks temporally to reduce high-frequency flicker.
        Applies a temporal sliding window moving average voting.

        Args:
            masks: List of per-frame binary masks (H, W) uint8.
            window_size: Odd integer size of sliding temporal window.
            threshold: Voting threshold (0.0 to 1.0) above which a pixel is considered masked.

        Returns:
            List[np.ndarray]: Temporally smoothed masks.
        """
        if not masks:
            return []

        w = self.temporal_window_size if window_size is None else window_size
        if w <= 1 or len(masks) < 3:
            return [m.copy() for m in masks]

        half_w = w // 2
        num_frames = len(masks)
        float_masks = [(m > 0).astype(np.float32) for m in masks]
        stabilized = []

        for i in range(num_frames):
            start_idx = max(0, i - half_w)
            end_idx = min(num_frames, i + half_w + 1)
            window = float_masks[start_idx:end_idx]

            # Mean activation in the temporal window
            avg_mask = np.mean(window, axis=0)
            binary_result = ((avg_mask >= threshold) * 255).astype(np.uint8)
            stabilized.append(binary_result)

        return stabilized

    # =========================================================================
    # 6. Mask Utilities
    # =========================================================================

    @staticmethod
    def get_mask_bounding_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the bounding box (x_min, y_min, x_max, y_max) of the non-zero region in the mask.
        Returns None if mask is empty.
        """
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    @staticmethod
    def overlay_mask_on_frame(
        frame: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (255, 0, 0),
        alpha: float = 0.45,
    ) -> np.ndarray:
        """
        Overlay a binary mask onto a BGR/RGB video frame with transparency for visualization.

        Args:
            frame: (H, W, 3) image array.
            mask: (H, W) binary mask array.
            color: Color tuple (e.g. Red=(255, 0, 0)).
            alpha: Mask opacity (0.0 to 1.0).

        Returns:
            np.ndarray: Blended image.
        """
        overlay = frame.copy()
        mask_bool = mask > 0
        colored_mask = np.zeros_like(frame)
        colored_mask[mask_bool] = color

        blended = frame.copy()
        blended[mask_bool] = cv2.addWeighted(
            frame[mask_bool], 1.0 - alpha, colored_mask[mask_bool], alpha, 0
        )
        return blended
