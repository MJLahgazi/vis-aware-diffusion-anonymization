import glob
import os
from typing import List

from config import SUPPORTED_FORMATS


def get_image_files(folder: str) -> List[str]:
    files = []
    for ext in SUPPORTED_FORMATS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(set(files))
