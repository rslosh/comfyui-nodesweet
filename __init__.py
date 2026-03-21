"""
ComfyUI NodeSweet
Custom nodes for batch processing, bbox detection, and audio-reactive transforms.
"""

import logging

logger = logging.getLogger(__name__)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def _register(class_name, display_name, import_fn):
    try:
        cls = import_fn()
        NODE_CLASS_MAPPINGS[class_name] = cls
        NODE_DISPLAY_NAME_MAPPINGS[class_name] = display_name
    except Exception as e:
        logger.warning(f"[NodeSweet] Failed to load {class_name}: {e}")

_register("BboxDetectorBatch", "BBox Detector (Batch)",
    lambda: __import__(".".join([__name__, "nodes", "bbox_batch_detector"]), fromlist=["BboxDetectorBatch"]).BboxDetectorBatch)
_register("BboxDetectorBatchForEach", "BBox Detector (Batch ForEach)",
    lambda: __import__(".".join([__name__, "nodes", "bbox_batch_detector_foreach"]), fromlist=["BboxDetectorBatchForEach"]).BboxDetectorBatchForEach)
_register("BboxDetectorBatchChunked", "BBox Detector (Batch Chunked)",
    lambda: __import__(".".join([__name__, "nodes", "bbox_batch_detector_chunked"]), fromlist=["BboxDetectorBatchChunked"]).BboxDetectorBatchChunked)
_register("MultilineStringRepeater", "Multiline String Repeater",
    lambda: __import__(".".join([__name__, "nodes", "multiline_string_repeater"]), fromlist=["MultilineStringRepeater"]).MultilineStringRepeater)
_register("AudioReactiveTransform", "Audio Reactive Transform",
    lambda: __import__(".".join([__name__, "nodes", "audio_reactive_transform"]), fromlist=["AudioReactiveTransform"]).AudioReactiveTransform)
_register("AudioWeightsRemap", "Audio Weights Remap",
    lambda: __import__(".".join([__name__, "nodes", "audio_reactive_transform"]), fromlist=["AudioWeightsRemap"]).AudioWeightsRemap)
_register("EaseCurve", "Ease Curve",
    lambda: __import__(".".join([__name__, "nodes", "easing_curve"]), fromlist=["EaseCurve"]).EaseCurve)
_register("ApplyEasingToFloats", "Apply Easing to Floats",
    lambda: __import__(".".join([__name__, "nodes", "easing_curve"]), fromlist=["ApplyEasingToFloats"]).ApplyEasingToFloats)
_register("LoadImageSetFromFolderSortedNode", "Load Image Dataset from Folder (Sorted)",
    lambda: __import__(".".join([__name__, "nodes", "sort_batch_image_loader"]), fromlist=["LoadImageSetFromFolderSortedNode"]).LoadImageSetFromFolderSortedNode)

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
__version__ = "1.1.0"
