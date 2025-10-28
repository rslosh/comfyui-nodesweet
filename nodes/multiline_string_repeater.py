"""
Multiline String Repeater Node for ComfyUI
Repeats each line in a multiline string a specified number of times
"""


class MultilineStringRepeater:
    """
    Takes a multiline string and repeats each line a specified number of times.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "multiline_text": ("STRING", {
                    "multiline": True, 
                    "default": "",
                    "tooltip": "The input text where each line will be repeated"
                }),
                "repeat_each_line": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 100, 
                    "step": 1,
                    "tooltip": "Number of times to repeat"
                }),
                "repeat_type": (["Line by Line", "Block"], {
                    "default": "Line by Line",
                    "tooltip": "Line by Line: repeat each line individually (hello\\nhello\\nworld\\nworld). Block: repeat all lines as a group (hello\\nworld\\nhello\\nworld)"
                }),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("multiline_text",)
    FUNCTION = "repeat_lines"
    CATEGORY = "nodesweet-hellorob"
    DESCRIPTION = "Repeats lines in a multiline string. Choose 'Line by Line' to repeat each line individually, or 'Block' to repeat the entire text as a group."
    
    def repeat_lines(self, multiline_text: str, repeat_each_line: int, repeat_type: str):
        """
        Process the multiline string and repeat lines based on the selected mode.
        
        Args:
            multiline_text: Input multiline string
            repeat_each_line: Number of times to repeat
            repeat_type: "Line by Line" or "Block"
            
        Returns:
            Tuple containing the processed multiline string
        """
        if not multiline_text:
            return ("",)
        
        lines = multiline_text.split('\n')
        repeated_lines = []
        
        if repeat_type == "Line by Line":
            # Repeat each line individually before moving to the next
            for line in lines:
                for _ in range(repeat_each_line):
                    repeated_lines.append(line)
        else:  # Block mode
            # Repeat the entire block of lines
            for _ in range(repeat_each_line):
                repeated_lines.extend(lines)
        
        result = '\n'.join(repeated_lines)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "MultilineStringRepeater": MultilineStringRepeater,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultilineStringRepeater": "Multiline String Repeater",
}

