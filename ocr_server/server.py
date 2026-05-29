"""OCR gRPC Server with PaddleOCR backend."""

import io
import os
import sys
import time
from pathlib import Path
from concurrent import futures

import cv2
import numpy as np

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
os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = os.environ.get("OCR_GPU_MEM", "0.9")

import grpc
from paddleocr import PaddleOCR

# Import generated protobuf files
try:
    from . import ocr_pb2
    from . import ocr_pb2_grpc
except ImportError:
    import ocr_pb2
    import ocr_pb2_grpc


class OcrServicer(ocr_pb2_grpc.OcrServiceServicer):
    """OCR Service implementation."""

    def __init__(self, init_async: bool = False):
        self._ocr_engine = None
        self._device = "cpu"
        self._ready = False
        self._init_error = None
        
        if init_async:
            import threading
            threading.Thread(target=self._init_engine, daemon=True).start()
        else:
            self._init_engine()

    def _init_engine(self):
        """Initialize OCR engine at startup."""
        try:
            import paddle
            
            # Force GPU if CUDA compiled
            use_gpu = paddle.is_compiled_with_cuda()
            if use_gpu:
                paddle.device.set_device("gpu:0")
                self._device = "gpu"
            else:
                self._device = "cpu"
            
            print(f"Initializing OCR engine (device={self._device})...")
            
            # Use server models for best accuracy with GPU
            model_type = "server" if use_gpu else "mobile"
            det_model = f"PP-OCRv5_{model_type}_det"
            rec_model = f"PP-OCRv5_{model_type}_rec"
            
            self._ocr_engine = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_detection_model_name=det_model,
                text_recognition_model_name=rec_model,
                text_det_limit_side_len=640,
                text_det_limit_type="max",
                text_det_thresh=0.15,
                text_det_box_thresh=0.3,
                device=self._device,
            )
            self._ready = True
            print("OCR engine initialized.")
        except Exception as e:
            self._init_error = str(e)
            print(f"OCR engine initialization failed: {e}")

    def _get_engine(self) -> PaddleOCR:
        """Get OCR engine (already initialized)."""
        return self._ocr_engine

    def Recognize(self, request, context):
        """Handle OCR recognition request."""
        # Check if engine is ready
        if not self._ready:
            if self._init_error:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details(f"OCR engine initialization failed: {self._init_error}")
            else:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("OCR engine is initializing, please retry")
            return ocr_pb2.OcrResponse(items=[], count=0, processing_time_ms=0)
        
        start_time = time.perf_counter()
        
        # Decode image
        nparr = np.frombuffer(request.image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return ocr_pb2.OcrResponse(items=[], count=0, processing_time_ms=0)
        
        # Run OCR
        engine = self._get_engine()
        result = engine.predict(img)
        
        # Parse results
        items: list[ocr_pb2.OcrItem] = []
        for page in result:
            if not hasattr(page, "get"):
                continue
            rec_texts = page.get("rec_texts", [])
            rec_scores = page.get("rec_scores", [])
            rec_boxes = page.get("rec_boxes", [])
            
            for text, score, box in zip(rec_texts, rec_scores, rec_boxes):
                if box is not None:
                    if isinstance(box, np.ndarray):
                        if box.ndim == 1 and len(box) == 4:
                            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                            items.append(ocr_pb2.OcrItem(
                                text=text,
                                x=x1, y=y1,
                                width=x2 - x1,
                                height=y2 - y1,
                                confidence=float(score),
                            ))
        
        elapsed = (time.perf_counter() - start_time) * 1000
        return ocr_pb2.OcrResponse(items=items, count=len(items), processing_time_ms=elapsed)

    def Health(self, request, context):
        """Handle health check request."""
        if not self._ready:
            status = "initializing" if not self._init_error else "error"
            return ocr_pb2.HealthResponse(
                healthy=False,
                device=status,
                version="1.0.0",
            )
        return ocr_pb2.HealthResponse(
            healthy=True,
            device=self._device,
            version="1.0.0",
        )


def serve(host: str = "0.0.0.0", port: int = 50051, max_workers: int = 1, async_init: bool = True):
    """Start the gRPC server.

    Args:
        host: Server bind address
        port: Server port
        max_workers: Number of worker threads
        async_init: Initialize OCR engine in background (server starts immediately)
    """
    # Initialize servicer
    servicer = OcrServicer(init_async=async_init)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),  # 50MB
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),  # 50MB
        ],
    )
    ocr_pb2_grpc.add_OcrServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    print(f"OCR gRPC server started on {host}:{port}")

    if async_init:
        print("OCR engine is initializing in background...")

    server.wait_for_termination()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="OCR gRPC Server")
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server bind address")
    parser.add_argument("--port", type=int, default=int(os.environ.get("OCR_PORT", "50051")), help="Server port")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker threads")
    parser.add_argument("--sync", action="store_true", help="Initialize engine before accepting connections")
    args = parser.parse_args()
    serve(host=args.host, port=args.port, max_workers=args.workers, async_init=not args.sync)


if __name__ == "__main__":
    main()
