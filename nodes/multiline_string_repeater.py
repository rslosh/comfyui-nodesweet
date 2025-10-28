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
                "multiline_text": ("STRING", {"multiline": True, "default": ""}),
                "repeat_each_line": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("multiline_text",)
    FUNCTION = "repeat_lines"
    CATEGORY = "nodesweet-hellorob"
    DESCRIPTION = "Repeats each line in a multiline string a specified number of times."
    
    def repeat_lines(self, multiline_text: str, repeat_each_line: int):
        """
        Process the multiline string and repeat each line.
        
        Args:
            multiline_text: Input multiline string
            repeat_each_line: Number of times to repeat each line
            
        Returns:
            Tuple containing the processed multiline string
        """
        if not multiline_text:
            return ("",)
        
        lines = multiline_text.split('\n')
        repeated_lines = []
        
        for line in lines:
            for _ in range(repeat_each_line):
                repeated_lines.append(line)
        
        result = '\n'.join(repeated_lines)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "MultilineStringRepeater": MultilineStringRepeater,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultilineStringRepeater": "Multiline String Repeater",
}

