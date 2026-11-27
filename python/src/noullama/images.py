import base64
import mimetypes
from os import PathLike
from pathlib import Path

_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)


def to_data_url(image: str | bytes | PathLike) -> str:
    if isinstance(image, str) and image.startswith("data:"):
        return image

    path = None if isinstance(image, bytes) else Path(image)
    data = image if path is None else path.read_bytes()
    mime = next((kind for signature, kind in _SIGNATURES if data.startswith(signature)), None)
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        mime = "image/webp"
    if mime is None and path is not None:
        mime, _ = mimetypes.guess_type(str(path))
    if mime is None or not mime.startswith(("image/", "audio/")):
        name = str(path) if path is not None else "bytes"
        raise ValueError(f"Can't tell the image type of {name}; pass a data: URL instead.")

    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"
