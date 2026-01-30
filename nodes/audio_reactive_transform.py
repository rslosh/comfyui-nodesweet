"""
Audio Reactive Transform nodes for ComfyUI.
Takes a foreground image (e.g. white text) and audio weights (float list) to produce
a batch of frames with per-frame scale, rotation, and opacity driven by audio intensity.
Optionally composites the transformed foreground over a static background image using
a mask to isolate which regions get transformed.
"""

import torch
import numpy as np
import math


class AudioReactiveTransform:
    """
    Takes a foreground image and audio weights, applies per-frame scale,
    rotation, and opacity transforms, then composites over an optional
    static background image. An optional mask controls which foreground
    pixels are affected (e.g. isolate white text from a photo).
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
            },
            "optional": {
                "background_image": ("IMAGE",),
                "mask": ("MASK",),
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
        background_image=None,
        mask=None,
    ):
        # image: foreground (e.g. white text over photo, or standalone)
        # background_image: optional static background (e.g. the photo)
        # mask: optional (H, W) mask — white = foreground to transform, black = ignore
        fg = image[0].cpu().numpy()  # (H, W, C) float32 0-1
        h, w, c = fg.shape
        num_frames = len(audio_weights)

        # Background: use provided image, or default to black
        if background_image is not None:
            bg = background_image[0].cpu().numpy()
            # Resize bg to match fg if needed
            if bg.shape[0] != h or bg.shape[1] != w:
                bg = np.array(
                    torch.nn.functional.interpolate(
                        torch.from_numpy(bg).permute(2, 0, 1).unsqueeze(0),
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    )[0].permute(1, 2, 0).numpy()
                )
        else:
            bg = np.zeros((h, w, c), dtype=np.float32)

        # Mask: (H, W) float 0-1, white = foreground region
        if mask is not None:
            mask_np = mask[0].cpu().numpy() if len(mask.shape) == 3 else mask.cpu().numpy()
            # Resize mask to match fg if needed
            if mask_np.shape[0] != h or mask_np.shape[1] != w:
                mask_np = np.array(
                    torch.nn.functional.interpolate(
                        torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0),
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0].numpy()
                )
        else:
            # No mask: treat entire foreground image as the layer to transform
            mask_np = None

        # Normalize weights to 0-1 range
        weights = np.array(audio_weights, dtype=np.float32)
        w_min, w_max = weights.min(), weights.max()
        if w_max - w_min > 1e-6:
            weights_norm = (weights - w_min) / (w_max - w_min)
        else:
            weights_norm = np.zeros_like(weights)

        frames = []
        for i in range(num_frames):
            t = float(weights_norm[i])

            scale = scale_min + t * (scale_max - scale_min)
            angle = rotation_min + t * (rotation_max - rotation_min)
            opacity = opacity_min + t * (opacity_max - opacity_min)

            frame = self._transform_and_composite(
                fg, bg, mask_np, h, w, c, scale, angle, opacity
            )
            frames.append(frame)

        batch = np.stack(frames, axis=0)
        batch_tensor = torch.from_numpy(batch).float()

        return (batch_tensor, num_frames)

    def _transform_and_composite(self, fg, bg, mask_np, h, w, c, scale, angle, opacity):
        """
        Transform the foreground (or masked region), then composite over background.
        - If mask is provided: extract masked region from fg, transform it,
          blend it over bg with audio-driven opacity.
        - If no mask: transform entire fg, blend over bg.
        """
        # Build the foreground layer to transform
        if mask_np is not None:
            # Extract only masked pixels from fg, rest becomes transparent (bg shows through)
            fg_layer = fg.copy()
        else:
            fg_layer = fg.copy()

        # Apply scale + rotation to the foreground layer
        transformed = self._affine_transform(fg_layer, h, w, c, scale, angle)

        # Also transform the mask if present
        if mask_np is not None:
            mask_3d = np.stack([mask_np] * c, axis=-1)
            transformed_mask = self._affine_transform(mask_3d, h, w, c, scale, angle)
            # Use single channel from transformed mask
            alpha = transformed_mask[:, :, 0] * opacity
        else:
            # No mask: use validity from transform (non-zero pixels) with opacity
            alpha = np.ones((h, w), dtype=np.float32) * opacity

        # Composite: output = bg * (1 - alpha) + transformed_fg * alpha
        alpha_3d = alpha[:, :, np.newaxis]
        output = bg * (1.0 - alpha_3d) + transformed * alpha_3d

        return np.clip(output, 0.0, 1.0)

    def _affine_transform(self, img, h, w, c, scale, angle):
        """Apply scale and rotation to an image using inverse mapping with bilinear interpolation."""
        output = np.zeros((h, w, c), dtype=np.float32)

        cx, cy = w / 2.0, h / 2.0

        angle_rad = math.radians(-angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        inv_scale = 1.0 / scale

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

        dx = xx - cx
        dy = yy - cy
        src_x = (cos_a * dx - sin_a * dy) * inv_scale + cx
        src_y = (sin_a * dx + cos_a * dy) * inv_scale + cy

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

            output[:, :, ch] = np.where(valid, value, 0.0)

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
