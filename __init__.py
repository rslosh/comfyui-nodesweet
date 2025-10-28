"""
ComfyUI Batch BBox Detector Node
Process image batches (video frames) with ultralytics bbox detection
"""

from .nodes.bbox_batch_detector import BboxDetectorBatch
from .nodes.bbox_batch_detector_foreach import BboxDetectorBatchForEach
from .nodes.bbox_batch_detector_chunked import BboxDetectorBatchChunked
from .nodes.multiline_string_repeater import MultilineStringRepeater

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "BboxDetectorBatch": BboxDetectorBatch,
    "BboxDetectorBatchForEach": BboxDetectorBatchForEach,
    "BboxDetectorBatchChunked": BboxDetectorBatchChunked,
    "MultilineStringRepeater": MultilineStringRepeater,
}

# Display names for ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "BboxDetectorBatch": "BBox Detector (Batch)",
    "BboxDetectorBatchForEach": "BBox Detector (Batch ForEach)",
    "BboxDetectorBatchChunked": "BBox Detector (Batch Chunked)",
    "SortImageSetFromFolderSortedNode": "Load Image Dataset from Folder (Sorted)",
    "MultilineStringRepeater": "Multiline String Repeater",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "1.0.0"


