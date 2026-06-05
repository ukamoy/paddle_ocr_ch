# PaddleOCR FastAPI ARM 服务

## 功能

- 支持粘贴网络图片 URL进行OCR
- 支持单张网络图片 URL 识别
- 异步下载网络图片（使用 `httpx.AsyncClient`）
- ARM CPU 优化（适用于 Oracle A1 / Ampere）
- FastAPI + Gunicorn 高并发稳定部署

---

## 环境要求

- Python 3.10
- ARM CPU（测试：Oracle A1 4~8vCPU）
- Docker（可选）

---

## 安装依赖

```bash
pip install -r requirements.txt
```


---
## 启动服务
本地开发
```bash
uvicorn main:app --host 0.0.0.0 --port 8010 --reload
```
ARM 生产部署（Gunicorn + UvicornWorker）
```bash
export OCR_CONCURRENCY_PER_WORKER=1

gunicorn main:app \
  -k uvicorn.workers.UvicornWorker \
  -w 5 \
  --threads 1 \
  --timeout 120 \
  --bind 0.0.0.0:8010
```
OCR 路由在每个 worker 内按 `OCR_CONCURRENCY_PER_WORKER` 排队处理。`5` 个 worker 且每个 worker 限制 `1` 个 OCR 时，总同时处理量约为 `5`；超过的请求会等待空位，不会返回 `429`。
---
