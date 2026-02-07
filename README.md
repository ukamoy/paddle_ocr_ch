# PaddleOCR FastAPI ARM 服务

## 功能

- 支持粘贴网络图片 URL进行OCR
- 支持多张图片批量识别, 一行一个url
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
```bash
gunicorn main:app \
  -k uvicorn.workers.UvicornWorker \
  -w 2 \
  --threads 1 \
  --timeout 120 \
  --bind 0.0.0.0:8010
```
---
