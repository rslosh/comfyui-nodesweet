"""
Easing curve nodes for ComfyUI.
Generates eased FLOAT sequences for animation and audio-reactive workflows.
"""

from ..lib.easing_functions import (
    get_easing_function,
    get_all_easing_names,
    create_bezier_easing,
)

PRESET_CHOICES = ["custom"] + get_all_easing_names()


class EaseCurve:
    """
    Generates a sequence of eased FLOAT values.
    Select a preset easing function or use custom bezier control points.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (PRESET_CHOICES,),
                "steps": ("INT", {"default": 30, "min": 2, "max": 10000}),
            },
            "optional": {
                "value_min": ("FLOAT", {"default": 0.0, "min": -1e6, "max": 1e6, "step": 0.01}),
                "value_max": ("FLOAT", {"default": 1.0, "min": -1e6, "max": 1e6, "step": 0.01}),
                "x1": ("FLOAT", {"default": 0.42, "min": 0.0, "max": 1.0, "step": 0.01}),
                "y1": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 2.0, "step": 0.01}),
                "x2": ("FLOAT", {"default": 0.58, "min": 0.0, "max": 1.0, "step": 0.01}),
                "y2": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("FLOATS",)
    RETURN_NAMES = ("eased_values",)
    FUNCTION = "generate"
    CATEGORY = "nodesweet-hellorob"

    def generate(self, preset, steps, value_min=0.0, value_max=1.0,
                 x1=0.42, y1=0.0, x2=0.58, y2=1.0):
        if preset == "custom":
            easing_fn = create_bezier_easing(x1, y1, x2, y2)
        else:
            easing_fn = get_easing_function(preset)
            if easing_fn is None:
                easing_fn = lambda t: t  # fallback to linear

        values = []
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0.0
            eased = easing_fn(t)
            value = value_min + eased * (value_max - value_min)
            values.append(value)

        return (values,)


class ApplyEasingToFloats:
    """
    Remaps existing FLOATS through an easing curve.
    Normalizes input to [0,1], applies easing, scales back to original range.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "floats_in": ("FLOATS",),
                "preset": (PRESET_CHOICES,),
            },
            "optional": {
                "x1": ("FLOAT", {"default": 0.42, "min": 0.0, "max": 1.0, "step": 0.01}),
                "y1": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 2.0, "step": 0.01}),
                "x2": ("FLOAT", {"default": 0.58, "min": 0.0, "max": 1.0, "step": 0.01}),
                "y2": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("FLOATS",)
    RETURN_NAMES = ("eased_floats",)
    FUNCTION = "apply_easing"
    CATEGORY = "nodesweet-hellorob"

    def apply_easing(self, floats_in, preset, x1=0.42, y1=0.0, x2=0.58, y2=1.0):
        if preset == "custom":
            easing_fn = create_bezier_easing(x1, y1, x2, y2)
        else:
            easing_fn = get_easing_function(preset)
            if easing_fn is None:
                easing_fn = lambda t: t

        # Find input range
        f_min = min(floats_in)
        f_max = max(floats_in)
        value_range = f_max - f_min

        if value_range < 1e-8:
            return (list(floats_in),)

        # Normalize to [0,1], apply easing, scale back
        values = []
        for v in floats_in:
            t = (v - f_min) / value_range
            eased = easing_fn(t)
            values.append(f_min + eased * value_range)

        return (values,)
