"""Constants for OCR server."""

from typing import Final

# Version
VERSION: Final[str] = "1.0.0"

# gRPC settings
DEFAULT_HOST: Final[str] = "0.0.0.0"
DEFAULT_PORT: Final[int] = 50051
DEFAULT_MAX_WORKERS: Final[int] = 1
MAX_MESSAGE_SIZE: Final[int] = 50 * 1024 * 1024  # 50MB

# OCR settings
DEFAULT_GPU_MEM_FRACTION: Final[float] = 0.9
DEFAULT_DET_LIMIT_SIDE_LEN: Final[int] = 960
DEFAULT_DET_THRESH: Final[float] = 0.1
DEFAULT_DET_BOX_THRESH: Final[float] = 0.2
DEFAULT_DET_UNCLIP_RATIO: Final[float] = 2.0

# Client settings
DEFAULT_TIMEOUT: Final[float] = 10.0
DEFAULT_JPEG_QUALITY: Final[int] = 85

# Image limits
MAX_IMAGE_SIZE: Final[int] = 100 * 1024 * 1024  # 100MB
MIN_IMAGE_SIZE: Final[int] = 100  # bytes

# Merge settings
DEFAULT_GAP_RATIO: Final[float] = 0.3
DEFAULT_HEIGHT_RATIO: Final[float] = 0.5
