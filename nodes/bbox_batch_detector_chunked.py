"""
Chunked Batch BBox Detector Node for ComfyUI
Processes large batches (300+ frames) in chunks to prevent OOM errors
RECOMMENDED for video frame processing
"""

import torch


class BboxDetectorBatchChunked:
    """
    Chunked batch processor for large batches (300+ frames).
    Processes in chunks to prevent out-of-memory errors.
    RECOMMENDED for video frame processing.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox_detector": ("BBOX_DETECTOR",),
                "images": ("IMAGE",),  # Expected shape: [B, H, W, C]
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dilation": ("INT", {"default": 4, "min": -512, "max": 512, "step": 1}),
                "chunk_size": ("INT", {"default": 32, "min": 1, "max": 128, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    FUNCTION = "detect_chunked"
    CATEGORY = "nodesweet-hellorob"
    
    def detect_chunked(self, bbox_detector, images, threshold, dilation, chunk_size=32):
        """
        Process large batches in chunks to manage memory efficiently.
        
        Args:
            bbox_detector: The bbox detector instance
            images: Batch of images with shape [B, H, W, C]
            threshold: Detection confidence threshold
            dilation: Mask expansion/dilation amount
            chunk_size: Number of frames to process at once (adjust based on VRAM)
            
        Returns:
            Tuple of (processed_images, masks) both with batch dimension
        """
        batch_size = images.shape[0]
        height = images.shape[1]
        width = images.shape[2]
        device = images.device
        
        print(f"Processing {batch_size} images in chunks of {chunk_size}...")
        
        all_images = []
        all_masks = []
        
        # Process in chunks
        num_chunks = (batch_size + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, batch_size)
            chunk_images = images[start_idx:end_idx]
            
            print(f"Processing chunk {chunk_idx + 1}/{num_chunks} (frames {start_idx}-{end_idx})...")
            
            chunk_processed_images = []
            chunk_masks = []
            
            for i in range(chunk_images.shape[0]):
                single_image = chunk_images[i]  # Shape: [H, W, C]
                global_idx = start_idx + i
                
                try:
                    # Detector expects batch dimension
                    mask = bbox_detector.detect_combined(
                        single_image.unsqueeze(0),
                        threshold,
                        dilation
                    )
                    
                    # Handle None or empty detections
                    if mask is None:
                        mask = torch.zeros((height, width), dtype=torch.float32, device=device)
                    elif mask.dim() == 3:
                        mask = mask.squeeze(0)
                    
                    chunk_masks.append(mask)
                    chunk_processed_images.append(single_image)
                    
                except Exception as e:
                    print(f"Error processing frame {global_idx}/{batch_size}: {str(e)}")
                    chunk_masks.append(torch.zeros((height, width), dtype=torch.float32, device=device))
                    chunk_processed_images.append(single_image)
            
            # Append chunk results
            all_images.extend(chunk_processed_images)
            all_masks.extend(chunk_masks)
            
            # Clear cache between chunks to manage memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            progress = ((chunk_idx + 1) / num_chunks) * 100
            print(f"Progress: {progress:.1f}% ({end_idx}/{batch_size} frames)")
        
        # Stack all results
        output_images = torch.stack(all_images, dim=0)  # [B, H, W, C]
        output_masks = torch.stack(all_masks, dim=0)  # [B, H, W]
        
        print(f"Chunked processing complete. Output shapes: images={output_images.shape}, masks={output_masks.shape}")
        
        return (output_images, output_masks)

