"""
Audio Reactive Transform nodes for ComfyUI.
Takes a single image and audio weights (float list) to produce a batch of
frames with per-frame scale, rotation, and opacity driven by audio intensity.
"""

import torch
import numpy as np
import math


class AudioReactiveTransform:
    """
    Takes a single image and a list of audio weights, then produces
    a batch of frames with per-frame scale, rotation, and opacity
    driven by the audio weights.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "audio_weights": ("FLOAT_LIST",),
                "scale_min": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05}),
                "scale_max": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 5.0, "step": 0.05}),
                "rotation_min": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "rotation_max": ("FLOAT", {"default": 15.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "opacity_min": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "opacity_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "background_color": (["black", "white"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image_batch", "frame_count")
    FUNCTION = "apply_transform"
    CATEGORY = "nodesweet-hellorob"

    def apply_transform(
        self,
        image,
        audio_weights,
        scale_min,
        scale_max,
        rotation_min,
        rotation_max,
        opacity_min,
        opacity_max,
        background_color,
    ):
        # image shape: (1, H, W, C) — single input image
        # audio_weights: list of floats, one per frame, typically 0.0-1.0
        img = image[0]  # (H, W, C)
        h, w, c = img.shape
        num_frames = len(audio_weights)

        # Normalize weights to 0-1 range
        weights = np.array(audio_weights, dtype=np.float32)
        w_min, w_max = weights.min(), weights.max()
        if w_max - w_min > 1e-6:
            weights_norm = (weights - w_min) / (w_max - w_min)
        else:
            weights_norm = np.zeros_like(weights)

        # Convert source image to numpy
        img_np = img.cpu().numpy()  # (H, W, C) float32 0-1

        bg_value = 0.0 if background_color == "black" else 1.0

        frames = []
        for i in range(num_frames):
            t = float(weights_norm[i])

            # Interpolate parameters from audio weight
            scale = scale_min + t * (scale_max - scale_min)
            angle = rotation_min + t * (rotation_max - rotation_min)
            opacity = opacity_min + t * (opacity_max - opacity_min)

            frame = self._transform_frame(img_np, h, w, c, scale, angle, opacity, bg_value)
            frames.append(frame)

        # Stack into batch tensor (N, H, W, C)
        batch = np.stack(frames, axis=0)
        batch_tensor = torch.from_numpy(batch).float()

        return (batch_tensor, num_frames)

    def _transform_frame(self, img, h, w, c, scale, angle, opacity, bg_value):
        """Apply scale, rotation, and opacity to a single frame using numpy."""
        output = np.full((h, w, c), bg_value, dtype=np.float32)

        cx, cy = w / 2.0, h / 2.0

        # Inverse affine transform (output -> input mapping)
        angle_rad = math.radians(-angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        inv_scale = 1.0 / scale

        # Coordinate grids
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

        # Translate to center, inverse rotate, inverse scale
        dx = xx - cx
        dy = yy - cy
        src_x = (cos_a * dx - sin_a * dy) * inv_scale + cx
        src_y = (sin_a * dx + cos_a * dy) * inv_scale + cy

        # Bilinear interpolation
        src_x0 = np.floor(src_x).astype(np.int32)
        src_y0 = np.floor(src_y).astype(np.int32)
        src_x1 = src_x0 + 1
        src_y1 = src_y0 + 1

        fx = src_x - src_x0.astype(np.float32)
        fy = src_y - src_y0.astype(np.float32)

        valid = (src_x0 >= 0) & (src_y0 >= 0) & (src_x1 < w) & (src_y1 < h)

        sx0 = np.clip(src_x0, 0, w - 1)
        sy0 = np.clip(src_y0, 0, h - 1)
        sx1 = np.clip(src_x1, 0, w - 1)
        sy1 = np.clip(src_y1, 0, h - 1)

        for ch in range(c):
            v00 = img[sy0, sx0, ch]
            v10 = img[sy0, sx1, ch]
            v01 = img[sy1, sx0, ch]
            v11 = img[sy1, sx1, ch]

            value = (
                v00 * (1 - fx) * (1 - fy)
                + v10 * fx * (1 - fy)
                + v01 * (1 - fx) * fy
                + v11 * fx * fy
            )

            blended = bg_value * (1 - opacity) + value * opacity
            output[:, :, ch] = np.where(valid, blended, bg_value)

        return output


class AudioWeightsRemap:
    """
    Remaps audio_weights with curve shaping (power, smoothing)
    for more artistic control before feeding into transforms.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_weights": ("FLOAT_LIST",),
                "power": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "smoothing": ("INT", {"default": 1, "min": 1, "max": 30, "step": 1}),
                "invert": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("FLOAT_LIST",)
    RETURN_NAMES = ("remapped_weights",)
    FUNCTION = "remap"
    CATEGORY = "nodesweet-hellorob"

    def remap(self, audio_weights, power, smoothing, invert):
        weights = np.array(audio_weights, dtype=np.float32)

        # Normalize to 0-1
        w_min, w_max = weights.min(), weights.max()
        if w_max - w_min > 1e-6:
            weights = (weights - w_min) / (w_max - w_min)
        else:
            weights = np.zeros_like(weights)

        # Power curve (> 1 = sharper peaks, < 1 = softer)
        weights = np.power(weights, power)

        # Moving average smoothing
        if smoothing > 1:
            kernel = np.ones(smoothing) / smoothing
            weights = np.convolve(weights, kernel, mode="same")

        # Re-normalize
        w_min, w_max = weights.min(), weights.max()
        if w_max - w_min > 1e-6:
            weights = (weights - w_min) / (w_max - w_min)

        if invert:
            weights = 1.0 - weights

        return (weights.tolist(),)
