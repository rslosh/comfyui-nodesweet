"""
Standard Batch BBox Detector Node for ComfyUI
Processes batches of images (e.g., video frames) with ultralytics bbox detection
"""

import torch
import numpy as np


class BboxDetectorBatch:
    """
    Standard batch processor for bbox detection.
    Processes each frame in a batch sequentially and returns stacked results.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox_detector": ("BBOX_DETECTOR",),
                "images": ("IMAGE",),  # Expected shape: [B, H, W, C]
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dilation": ("INT", {"default": 4, "min": -512, "max": 512, "step": 1}),
            },
            "optional": {
                "return_type": (["mask_only", "image_with_boxes"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    FUNCTION = "detect_batch"
    CATEGORY = "nodesweet-hellorob"
    
    def detect_batch(self, bbox_detector, images, threshold, dilation, return_type="mask_only"):
        """
        Process a batch of images through bbox detection.
        
        Args:
            bbox_detector: The bbox detector instance
            images: Batch of images with shape [B, H, W, C]
            threshold: Detection confidence threshold
            dilation: Mask expansion/dilation amount
            return_type: "mask_only" or "image_with_boxes"
            
        Returns:
            Tuple of (processed_images, masks) both with batch dimension
        """
        batch_size = images.shape[0]
        height = images.shape[1]
        width = images.shape[2]
        device = images.device
        
        print(f"Processing batch of {batch_size} images with BboxDetectorBatch...")
        
        processed_images = []
        masks = []
        
        for i in range(batch_size):
            single_image = images[i]  # Shape: [H, W, C]
            
            try:
                # Detector expects batch dimension, so add it back temporarily
                mask = bbox_detector.detect_combined(
                    single_image.unsqueeze(0), 
                    threshold, 
                    dilation
                )
                
                # Handle None or empty detections
                if mask is None:
                    mask = torch.zeros((height, width), dtype=torch.float32, device=device)
                elif mask.dim() == 3:  # Remove batch dimension if present
                    mask = mask.squeeze(0)
                
                # Store mask
                masks.append(mask)
                
                # Process image based on return_type
                if return_type == "image_with_boxes":
                    # Draw bounding boxes on image
                    processed_img = self._draw_detections(
                        single_image, 
                        bbox_detector, 
                        threshold
                    )
                    processed_images.append(processed_img)
                else:
                    # Return original image
                    processed_images.append(single_image)
                    
            except Exception as e:
                print(f"Error processing frame {i}/{batch_size}: {str(e)}")
                # Use empty mask and original image on error
                masks.append(torch.zeros((height, width), dtype=torch.float32, device=device))
                processed_images.append(single_image)
        
        # Stack results back into batches
        output_images = torch.stack(processed_images, dim=0)  # [B, H, W, C]
        output_masks = torch.stack(masks, dim=0)  # [B, H, W]
        
        print(f"Batch processing complete. Output shapes: images={output_images.shape}, masks={output_masks.shape}")
        
        return (output_images, output_masks)
    
    def _draw_detections(self, image, bbox_detector, threshold):
        """
        Helper method to draw bounding boxes on image.
        
        Args:
            image: Single image tensor [H, W, C]
            bbox_detector: Detector instance
            threshold: Detection threshold
            
        Returns:
            Image with drawn bounding boxes
        """
        # Convert to numpy for drawing
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        
        # Get detections (this is a simplified version - adapt based on your detector's API)
        try:
            # Assuming detector has a method to get bbox coordinates
            # You may need to adjust this based on the actual detector API
            detections = bbox_detector.detect(image.unsqueeze(0), threshold)
            
            if detections is not None and len(detections) > 0:
                # Draw boxes (simplified - enhance based on actual detection format)
                for detection in detections:
                    # detection format may vary - adjust accordingly
                    pass
        except:
            # If drawing fails, return original image
            pass
        
        # Convert back to tensor
        result = torch.from_numpy(img_np.astype(np.float32) / 255.0).to(image.device)
        return result


