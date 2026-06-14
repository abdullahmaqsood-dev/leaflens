from pathlib import Path


def ensure_image_path(image_path: str) -> Path:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return path
