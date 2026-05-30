# OCR Server

## Quick Start

### 1. Install

```bash
cd ocr-server

# Server (full)
pip install -e .
pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/

# Client only (for other projects to depend on)
pip install -e ".[client]"
```

### 2. Start Server

```bash
# Option 1: After install (recommended)
ocr-server --port 50051

# Option 2: From ocr_server directory
cd ocr_server
python -m ocr_server.server --port 50051

# Option 3: Sync mode (wait for engine init before accepting connections)
ocr-server --sync
```

### 3. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_PORT` | `50051` | Server port |
| `OCR_GPU_MEM` | `0.9` | GPU memory fraction |

### 4. Server Behavior

- **Async init** (default): Server starts accepting connections immediately, engine initializes in background
- **Sync init** (`--sync`): Server waits for engine to initialize before accepting connections
- During initialization, `Health` returns `healthy=False, device=initializing`
- `Recognize` returns `UNAVAILABLE` if engine is not ready

### 5. Test Client

```python
from ocr_server.client import OcrClient
import cv2

with OcrClient(host="localhost", port=50051) as client:
    # Health check
    health = client.health_check()
    print(f"Server: {health}")
    
    # OCR
    img = cv2.imread("test.png")
    results = client.recognize(img)
    for r in results:
        print(f"{r['text']} @ ({r['x']}, {r['y']}) conf={r['confidence']:.2f}")
```

## Architecture

```
┌─────────────┐      gRPC       ┌──────────────┐
│   Client    │  ─────────────> │    Server    │
│   (any)     │   protobuf      │  (PaddleOCR) │
└─────────────┘                 └──────────────┘
```

## API

### Recognize

- **Input**: `image` (bytes), optional `roi` (left, top, right, bottom)
- **Output**: List of `{text, x, y, width, height, confidence}`

### Health

- **Output**: `{healthy, device, version}`

## License

MIT
