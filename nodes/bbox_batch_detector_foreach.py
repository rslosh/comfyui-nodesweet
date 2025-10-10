"""
ForEach Batch BBox Detector Node for ComfyUI
List-based processor using INPUT_IS_LIST for memory-efficient workflows
"""

import torch


class BboxDetectorBatchForEach:
    """
    List-based processor using INPUT_IS_LIST for more memory-efficient workflows.
    Better for certain ComfyUI processing patterns.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox_detector": ("BBOX_DETECTOR",),
                "images": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dilation": ("INT", {"default": 4, "min": -512, "max": 512, "step": 1}),
            }
        }
    
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    FUNCTION = "detect_foreach"
    CATEGORY = "ImpactPack/Detector"
    
    def detect_foreach(self, bbox_detector, images, threshold, dilation):
        """
        Process images as a list using ForEach pattern.
        
        Args:
            bbox_detector: List containing detector instance
            images: List of image tensors
            threshold: List with threshold value
            dilation: List with dilation value
            
        Returns:
            Lists of (images, masks)
        """
        # Extract single values from lists
        detector = bbox_detector[0]
        thresh = threshold[0]
        dilate = dilation[0]
        
        print(f"Processing {len(images)} images with BboxDetectorBatchForEach...")
        
        output_images = []
        output_masks = []
        
        for i, image in enumerate(images):
            # Handle both batched and single images
            if image.dim() == 4:
                # Batched image [B, H, W, C] - process first frame
                single_image = image[0]
            else:
                # Single image [H, W, C]
                single_image = image
            
            height, width = single_image.shape[:2]
            device = single_image.device
            
            try:
                # Detector expects batch dimension
                mask = detector.detect_combined(
                    single_image.unsqueeze(0),
                    thresh,
                    dilate
                )
                
                # Handle None or empty detections
                if mask is None:
                    mask = torch.zeros((height, width), dtype=torch.float32, device=device)
                elif mask.dim() == 3:
                    mask = mask.squeeze(0)
                
                output_masks.append(mask)
                output_images.append(single_image)
                
            except Exception as e:
                print(f"Error processing image {i}/{len(images)}: {str(e)}")
                output_masks.append(torch.zeros((height, width), dtype=torch.float32, device=device))
                output_images.append(single_image)
        
        print(f"ForEach processing complete. Processed {len(output_images)} images.")
        
        return (output_images, output_masks)

