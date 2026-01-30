"""
ComfyUI NodeSweet
Custom nodes for batch processing, bbox detection, and audio-reactive transforms.
"""

from .nodes.bbox_batch_detector import BboxDetectorBatch
from .nodes.bbox_batch_detector_foreach import BboxDetectorBatchForEach
from .nodes.bbox_batch_detector_chunked import BboxDetectorBatchChunked
from .nodes.multiline_string_repeater import MultilineStringRepeater
from .nodes.audio_reactive_transform import AudioReactiveTransform, AudioWeightsRemap

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "BboxDetectorBatch": BboxDetectorBatch,
    "BboxDetectorBatchForEach": BboxDetectorBatchForEach,
    "BboxDetectorBatchChunked": BboxDetectorBatchChunked,
    "MultilineStringRepeater": MultilineStringRepeater,
    "AudioReactiveTransform": AudioReactiveTransform,
    "AudioWeightsRemap": AudioWeightsRemap,
}

# Display names for ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "BboxDetectorBatch": "BBox Detector (Batch)",
    "BboxDetectorBatchForEach": "BBox Detector (Batch ForEach)",
    "BboxDetectorBatchChunked": "BBox Detector (Batch Chunked)",
    "SortImageSetFromFolderSortedNode": "Load Image Dataset from Folder (Sorted)",
    "MultilineStringRepeater": "Multiline String Repeater",
    "AudioReactiveTransform": "Audio Reactive Transform",
    "AudioWeightsRemap": "Audio Weights Remap",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "1.1.0"


