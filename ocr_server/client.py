"""OCR gRPC Client."""

import logging
from types import TracebackType
from typing import Optional

import cv2
import grpc
import numpy as np

from .constants import DEFAULT_JPEG_QUALITY, DEFAULT_TIMEOUT, MAX_MESSAGE_SIZE

try:
    from . import ocr_pb2
    from . import ocr_pb2_grpc
except ImportError:
    import ocr_pb2
    import ocr_pb2_grpc

logger = logging.getLogger(__name__)


class OcrClient:
    """gRPC client for OCR service."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50051,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize OCR client.

        Args:
            host: OCR server hostname or IP
            port: OCR server port
            timeout: Request timeout in seconds
        """
        self._host = host
        self._port = port
        self._timeout = timeout
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[ocr_pb2_grpc.OcrServiceStub] = None

    def connect(self) -> None:
        """Connect to OCR server."""
        if self._channel is not None:
            return

        target = f"{self._host}:{self._port}"
        self._channel = grpc.insecure_channel(
            target,
            options=[
                ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
                ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
            ],
        )
        self._stub = ocr_pb2_grpc.OcrServiceStub(self._channel)
        logger.info("Connected to OCR server at %s", target)

    def close(self) -> None:
        """Close connection."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None
            logger.info("Disconnected from OCR server")

    def __enter__(self) -> "OcrClient":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        self.close()
        return False

    def recognize(
        self,
        image_bgr: np.ndarray,
        roi: Optional[tuple[float, float, float, float]] = None,
    ) -> list[dict[str, object]]:
        """Recognize text in an image.

        Args:
            image_bgr: Input image in BGR format (numpy array)
            roi: Optional ROI as (left, top, right, bottom) normalized coordinates (0-1)

        Returns:
            List of dicts with keys: text, x, y, width, height, confidence

        Raises:
            RuntimeError: If not connected to server
            grpc.RpcError: If gRPC call fails
        """
        if self._stub is None:
            self.connect()
        if self._stub is None:
            raise RuntimeError("Not connected to OCR server")

        ox, oy = 0, 0
        if roi is not None:
            h, w = image_bgr.shape[:2]
            l, t, r, b = roi
            x1 = max(0, int(w * l))
            y1 = max(0, int(h * t))
            x2 = min(w, int(w * r))
            y2 = min(h, int(h * b))
            if x2 > x1 and y2 > y1:
                image_bgr = image_bgr[y1:y2, x1:x2]
                ox, oy = x1, y1

        success, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, DEFAULT_JPEG_QUALITY])
        if not success or encoded is None:
            raise ValueError("Failed to encode image to JPEG")
        image_bytes = encoded.tobytes()

        request = ocr_pb2.OcrRequest(image=image_bytes)

        response = self._stub.Recognize(request, timeout=self._timeout)

        results: list[dict[str, object]] = []
        for item in response.items:
            results.append({
                "text": item.text,
                "x": item.x + ox,
                "y": item.y + oy,
                "width": item.width,
                "height": item.height,
                "confidence": item.confidence,
            })
        return results

    def health_check(self) -> dict[str, object]:
        """Check server health.

        Returns:
            Dict with keys: healthy, device, version

        Raises:
            RuntimeError: If not connected to server
            grpc.RpcError: If gRPC call fails
        """
        if self._stub is None:
            self.connect()
        if self._stub is None:
            raise RuntimeError("Not connected to OCR server")

        response = self._stub.Health(ocr_pb2.HealthRequest(), timeout=5.0)
        return {
            "healthy": response.healthy,
            "device": response.device,
            "version": response.version,
        }
