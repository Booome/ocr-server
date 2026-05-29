# OCR gRPC Server - Design Document

## 概述

基于 PaddleOCR 的 gRPC OCR 服务，支持 GPU/CPU 自动切换，用于局域网内的高性能文字识别。

## 架构

```
┌─────────────────┐      gRPC      ┌──────────────────┐
│   hlddz Client  │  ───────────>  │   OCR Server     │
│  (ocr_client)   │   protobuf     │  (PaddleOCR)     │
└─────────────────┘                └──────────────────┘
                                          │
                                    ┌─────┴─────┐
                                    ▼           ▼
                                  GPU         CPU
```

## 技术栈

- **Protocol**: gRPC + protobuf
- **Framework**: grpcio, paddleocr
- **Models**: PP-OCRv5 mobile (平衡速度/精度)
- **GPU**: CUDA 12.9 + cuDNN 9.x

## API

### Recognize

Request:
- `image`: 图片数据 (bytes)
- `roi`: 可选区域 (left, top, right, bottom, 0-1)

Response:
- `items`: 检测到的文字列表
- `processing_time_ms`: 处理时间
- `count`: 检测数量

## 性能目标

- 传输延迟: <5ms (局域网)
- 推理延迟: ~150-200ms (1920x1080)
- 支持并发: 多worker线程池

## 部署

服务端 (GPU机器):
```bash
ocr-server --port 50051 --host 0.0.0.0
```

客户端:
```python
client = OcrClient(host="192.168.1.100", port=50051)
```

## 文件结构

```
ocr_server/
├── proto/
│   └── ocr.proto
├── ocr_server/
│   ├── __init__.py
│   ├── server.py
│   └── client.py
├── pyproject.toml
└── README.md
```
