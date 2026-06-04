"""OCR gRPC Server with PaddleOCR backend."""

import logging
import os
import sys
import threading
import time
from concurrent import futures
from pathlib import Path
from typing import Optional

import cv2
import grpc
import numpy as np

from .constants import (
    DEFAULT_DET_BOX_THRESH,
    DEFAULT_DET_LIMIT_SIDE_LEN,
    DEFAULT_DET_THRESH,
    DEFAULT_DET_UNCLIP_RATIO,
    DEFAULT_GAP_RATIO,
    DEFAULT_GPU_MEM_FRACTION,
    DEFAULT_HEIGHT_RATIO,
    DEFAULT_HOST,
    DEFAULT_MAX_WORKERS,
    DEFAULT_PORT,
    MAX_IMAGE_SIZE,
    MAX_MESSAGE_SIZE,
    VERSION,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Add cuDNN DLL paths for CUDA 12.x
_NVIDIA_BASE = Path(sys.executable).parent.parent / "Lib" / "site-packages" / "nvidia"
if _NVIDIA_BASE.exists():
    for sub in ("cudnn", "cublas", "cuda_runtime", "cufft", "curand",
                "cusolver", "cusparse", "nvjit", "nvrtc"):
        bin_dir = _NVIDIA_BASE / sub / "bin"
        if bin_dir.exists():
            os.add_dll_directory(str(bin_dir))

# Add TensorRT DLL path
_TENSORRT_LIBS = Path(sys.executable).parent.parent / "Lib" / "site-packages" / "tensorrt_libs"
if _TENSORRT_LIBS.exists():
    os.add_dll_directory(str(_TENSORRT_LIBS))

# GPU memory allocation - must be set before importing paddle
os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = os.environ.get(
    "OCR_GPU_MEM", str(DEFAULT_GPU_MEM_FRACTION)
)

from paddleocr import PaddleOCR  # noqa: E402

# Import generated protobuf files
try:
    from . import ocr_pb2
    from . import ocr_pb2_grpc
except ImportError:
    import ocr_pb2
    import ocr_pb2_grpc


def _merge_adjacent_items(
    items: list[dict],
    gap_ratio: float = DEFAULT_GAP_RATIO,
    height_ratio: float = DEFAULT_HEIGHT_RATIO,
) -> list[dict]:
    """Merge text items that are on the same line and close together.

    Args:
        items: List of dicts with keys: text, x, y, width, height, confidence
        gap_ratio: Max gap as ratio of average height to merge
        height_ratio: Max height difference as ratio to consider same line

    Returns:
        Merged list of items
    """
    if len(items) <= 1:
        return items

    sorted_items = sorted(items, key=lambda it: it["x"])
    merged: list[dict] = []
    i = 0

    while i < len(sorted_items):
        current = dict(sorted_items[i])
        j = i + 1

        while j < len(sorted_items):
            next_item = sorted_items[j]
            avg_height = (current["height"] + next_item["height"]) / 2
            height_diff = abs(current["height"] - next_item["height"])

            same_line = (
                height_diff < avg_height * height_ratio
                and abs(current["y"] - next_item["y"]) < avg_height * height_ratio
            )

            current_right = current["x"] + current["width"]
            gap = next_item["x"] - current_right
            close_enough = gap < avg_height * gap_ratio

            if same_line and close_enough:
                new_right = max(current_right, next_item["x"] + next_item["width"])
                new_bottom = max(
                    current["y"] + current["height"],
                    next_item["y"] + next_item["height"],
                )
                current["y"] = min(current["y"], next_item["y"])
                current["width"] = new_right - current["x"]
                current["height"] = new_bottom - current["y"]
                current["text"] = current["text"] + next_item["text"]
                current["confidence"] = min(current["confidence"], next_item["confidence"])
                j += 1
            else:
                break

        merged.append(current)
        i = j

    return merged


class OcrServicer(ocr_pb2_grpc.OcrServiceServicer):
    """OCR Service implementation."""

    def __init__(self, init_async: bool = False) -> None:
        self._ocr_engine: Optional[PaddleOCR] = None
        self._device: str = "cpu"
        self._ready: bool = False
        self._init_error: Optional[str] = None
        self._lock = threading.Lock()

        if init_async:
            threading.Thread(target=self._init_engine, daemon=True).start()
        else:
            self._init_engine()

    def _init_engine(self) -> None:
        """Initialize OCR engine at startup."""
        try:
            import paddle

            use_gpu = paddle.is_compiled_with_cuda()
            if use_gpu:
                paddle.device.set_device("gpu:0")
                self._device = "gpu"
            else:
                self._device = "cpu"

            logger.info("Initializing OCR engine (device=%s)...", self._device)

            model_type = "server" if use_gpu else "mobile"
            det_model = f"PP-OCRv5_{model_type}_det"
            rec_model = f"PP-OCRv5_{model_type}_rec"

            self._ocr_engine = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_detection_model_name=det_model,
                text_recognition_model_name=rec_model,
                text_det_limit_side_len=DEFAULT_DET_LIMIT_SIDE_LEN,
                text_det_limit_type="max",
                text_det_thresh=DEFAULT_DET_THRESH,
                text_det_box_thresh=DEFAULT_DET_BOX_THRESH,
                text_det_unclip_ratio=DEFAULT_DET_UNCLIP_RATIO,
                device=self._device,
            )

            with self._lock:
                self._ready = True
            logger.info("OCR engine initialized.")

        except (ImportError, RuntimeError, OSError) as e:
            with self._lock:
                self._init_error = str(e)
            logger.exception("OCR engine initialization failed")

    def _validate_image(self, image_bytes: bytes) -> bool:
        """Validate image data."""
        if len(image_bytes) > MAX_IMAGE_SIZE:
            logger.warning("Image too large: %d bytes", len(image_bytes))
            return False
        if len(image_bytes) < 100:
            logger.warning("Image too small: %d bytes", len(image_bytes))
            return False
        return True

    def Recognize(
        self,
        request: ocr_pb2.OcrRequest,
        context: grpc.ServicerContext,
    ) -> ocr_pb2.OcrResponse:
        """Handle OCR recognition request."""
        with self._lock:
            if not self._ready:
                if self._init_error:
                    context.set_code(grpc.StatusCode.UNAVAILABLE)
                    context.set_details(f"OCR engine initialization failed: {self._init_error}")
                else:
                    context.set_code(grpc.StatusCode.UNAVAILABLE)
                    context.set_details("OCR engine is initializing, please retry")
                return ocr_pb2.OcrResponse(items=[], count=0, processing_time_ms=0)

        start_time = time.perf_counter()

        if not self._validate_image(request.image):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Invalid image data")
            return ocr_pb2.OcrResponse(items=[], count=0, processing_time_ms=0)

        nparr = np.frombuffer(request.image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Failed to decode image")
            return ocr_pb2.OcrResponse(items=[], count=0, processing_time_ms=0)

        engine = self._ocr_engine
        if engine is None:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("OCR engine not available")
            return ocr_pb2.OcrResponse(items=[], count=0, processing_time_ms=0)

        result = engine.predict(img)

        raw_items: list[dict] = []
        for page in result:
            if not hasattr(page, "get"):
                continue
            rec_texts = page.get("rec_texts", [])
            rec_scores = page.get("rec_scores", [])
            rec_boxes = page.get("rec_boxes", [])

            for text, score, box in zip(rec_texts, rec_scores, rec_boxes):
                if box is not None and isinstance(box, np.ndarray):
                    if box.ndim == 1 and len(box) == 4:
                        x1, y1, x2, y2 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                        raw_items.append({
                            "text": text,
                            "x": x1, "y": y1,
                            "width": x2 - x1,
                            "height": y2 - y1,
                            "confidence": float(score),
                        })

        merged_items = _merge_adjacent_items(raw_items)

        items = [
            ocr_pb2.OcrItem(
                text=it["text"],
                x=it["x"], y=it["y"],
                width=it["width"], height=it["height"],
                confidence=it["confidence"],
            )
            for it in merged_items
        ]

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info("Recognized %d items in %.1fms", len(items), elapsed)
        return ocr_pb2.OcrResponse(items=items, count=len(items), processing_time_ms=elapsed)

    def Health(
        self,
        request: ocr_pb2.HealthRequest,
        context: grpc.ServicerContext,
    ) -> ocr_pb2.HealthResponse:
        """Handle health check request."""
        with self._lock:
            if not self._ready:
                status = "initializing" if not self._init_error else "error"
                return ocr_pb2.HealthResponse(
                    healthy=False,
                    device=status,
                    version=VERSION,
                )
            return ocr_pb2.HealthResponse(
                healthy=True,
                device=self._device,
                version=VERSION,
            )


def serve(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    max_workers: int = DEFAULT_MAX_WORKERS,
    async_init: bool = True,
) -> None:
    """Start the gRPC server.

    Args:
        host: Server bind address
        port: Server port
        max_workers: Number of worker threads
        async_init: Initialize OCR engine in background (server starts immediately)
    """
    servicer = OcrServicer(init_async=async_init)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
        ],
    )
    ocr_pb2_grpc.add_OcrServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logger.info("OCR gRPC server started on %s:%d", host, port)

    if async_init:
        logger.info("OCR engine is initializing in background...")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(grace=5)


def main() -> None:
    """Entry point for the OCR server."""
    import argparse

    parser = argparse.ArgumentParser(description="OCR gRPC Server")
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Server bind address")
    parser.add_argument("--port", type=int, default=int(os.environ.get("OCR_PORT", str(DEFAULT_PORT))), help="Server port")
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS, help="Number of worker threads")
    parser.add_argument("--sync", action="store_true", help="Initialize engine before accepting connections")
    args = parser.parse_args()
    serve(host=args.host, port=args.port, max_workers=args.workers, async_init=not args.sync)


if __name__ == "__main__":
    main()
