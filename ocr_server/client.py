"""OCR gRPC Client for hlddz integration."""

import io
from typing import Optional

import cv2
import numpy as np

import grpc

from . import ocr_pb2
from . import ocr_pb2_grpc


class OcrClient:
    """gRPC client for OCR service."""

    def __init__(self, host: str = "localhost", port: int = 50051, timeout: float = 10.0):
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
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )
        self._stub = ocr_pb2_grpc.OcrServiceStub(self._channel)

    def close(self) -> None:
        """Close connection."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def recognize(
        self,
        image_bgr: np.ndarray,
        roi: Optional[tuple[float, float, float, float]] = None,
    ) -> list[dict]:
        """Recognize text in an image.

        Args:
            image_bgr: Input image in BGR format (numpy array)
            roi: Optional ROI as (left, top, right, bottom) normalized coordinates (0-1)

        Returns:
            List of dicts with keys: text, x, y, width, height, confidence
        """
        if self._stub is None:
            self.connect()
        assert self._stub is not None

        # Crop image locally if ROI specified (reduces transfer size)
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

        # Encode image to JPEG (faster than PNG, ~75% faster transfer)
        success, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            return []
        image_bytes = encoded.tobytes()

        # Build request (no ROI needed, already cropped)
        request = ocr_pb2.OcrRequest(image=image_bytes)

        # Call gRPC
        try:
            response = self._stub.Recognize(request, timeout=self._timeout)
        except grpc.RpcError as e:
            print(f"gRPC error: {e}")
            return []

        # Parse response (adjust coordinates back to original image)
        results = []
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

    def health_check(self) -> dict:
        """Check server health.
        
        Returns:
            Dict with keys: healthy, device, version
        """
        if self._stub is None:
            self.connect()
        assert self._stub is not None
        
        try:
            response = self._stub.Health(ocr_pb2.HealthRequest(), timeout=5.0)
            return {
                "healthy": response.healthy,
                "device": response.device,
                "version": response.version,
            }
        except grpc.RpcError as e:
            return {"healthy": False, "device": "unknown", "version": "unknown", "error": str(e)}
