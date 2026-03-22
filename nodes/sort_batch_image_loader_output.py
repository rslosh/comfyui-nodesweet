import os
import re
import torch
import torch.nn.functional as F
from typing import Any, Tuple, List

from comfy.comfy_types.node_typing import IO
import folder_paths

from comfy_extras.nodes_dataset import load_and_process_images


def _natural_key(s: str, case_sensitive: bool) -> Tuple[Any, ...]:
    if not case_sensitive:
        s = s.lower()
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", s))


def _get_output_subfolders() -> List[str]:
    """Returns a list of all subfolder paths in the output directory, recursively.
    Includes '.' for the root output directory itself."""
    output_dir = folder_paths.get_output_directory()
    folders = ["."]

    try:
        if not os.path.exists(output_dir):
            return folders

        for root, dirs, _ in os.walk(output_dir):
            rel_path = os.path.relpath(root, output_dir)
            if rel_path != ".":
                folders.append(rel_path.replace(os.sep, "/"))

        return sorted(folders)
    except FileNotFoundError:
        return folders


class LoadImageSetFromOutputFolderSortedNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": (_get_output_subfolders(), {"tooltip": "The output subfolder to load images from. '.' = root output folder."}),
            },
            "optional": {
                "resize_method": (
                    ["None", "Stretch", "Crop", "Pad"],
                    {"default": "None"},
                ),
                "sort_order": (
                    ["Ascending", "Descending", "None"],
                    {"default": "Ascending", "tooltip": "Sort images by filename."},
                ),
                "natural_sort": (
                    IO.BOOLEAN,
                    {"default": True, "tooltip": "Use natural sort (e.g. img2.png before img10.png)."},
                ),
                "case_sensitive": (
                    IO.BOOLEAN,
                    {"default": False, "tooltip": "Case-sensitive sorting."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_images"
    CATEGORY = "loaders"
    EXPERIMENTAL = False
    DESCRIPTION = "Loads a batch of images from a selected output subfolder, sorted by filename."

    def load_images(self, folder: str, resize_method: str, sort_order: str = "Ascending", natural_sort: bool = True, case_sensitive: bool = False):
        output_dir = folder_paths.get_output_directory()
        target_dir = output_dir if folder == "." else os.path.join(output_dir, folder)

        valid_extensions = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".jpe", ".apng", ".tif", ".tiff"]
        image_files: List[str] = [
            f for f in os.listdir(target_dir)
            if any(f.lower().endswith(ext) for ext in valid_extensions)
        ]

        if sort_order != "None":
            reverse = sort_order == "Descending"
            if natural_sort:
                image_files.sort(key=lambda s: _natural_key(s, case_sensitive), reverse=reverse)
            else:
                image_files.sort(key=(None if case_sensitive else str.lower), reverse=reverse)

        tensors = load_and_process_images(image_files, target_dir)

        all_same_size = len(set((t.shape[1], t.shape[2]) for t in tensors)) == 1
        if resize_method == "None" or all_same_size:
            output_tensor = torch.cat(tensors, dim=0)
        else:
            target_h = max(t.shape[1] for t in tensors)
            target_w = max(t.shape[2] for t in tensors)
            resized = []
            for t in tensors:
                if t.shape[1] == target_h and t.shape[2] == target_w:
                    resized.append(t)
                    continue
                t_bchw = t.permute(0, 3, 1, 2)
                if resize_method == "Pad":
                    pad_h = target_h - t.shape[1]
                    pad_w = target_w - t.shape[2]
                    r = F.pad(t_bchw, (0, pad_w, 0, pad_h))
                else:  # Stretch or Crop
                    r = F.interpolate(t_bchw, size=(target_h, target_w), mode="bilinear", align_corners=False)
                resized.append(r.permute(0, 2, 3, 1))
            output_tensor = torch.cat(resized, dim=0)

        return (output_tensor,)


NODE_CLASS_MAPPINGS = {
    "LoadImageSetFromOutputFolderSortedNode": LoadImageSetFromOutputFolderSortedNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageSetFromOutputFolderSortedNode": "Load Image Dataset from Output Folder (Sorted)",
}
