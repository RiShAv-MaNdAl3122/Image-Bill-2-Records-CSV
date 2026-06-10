import os
from PIL import Image

def is_valid_image(image_path: str) -> bool:
    """
    Checks if the image exists, can be opened by PIL, and has a reasonable resolution.
    """
    if not os.path.exists(image_path):
        return False
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            # Minimum dimension limit to filter out tiny images/slices
            return w > 100 and h > 100
    except Exception:
        return False
