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

使用 Docker 启动容器（首次运行会自动构建镜像）：
```bash
sh deploy.sh
```
默认通过 `http://localhost:8010` 访问。可通过 `PORT` 指定宿主机端口，通过 `OCR_CONCURRENCY_PER_WORKER` 设置每个 worker 的 OCR 并发数，例如：
```bash
PORT=8080 OCR_CONCURRENCY_PER_WORKER=1 sh deploy.sh
```
再次运行脚本会替换并重启同名容器 `paddle-ocr-ch`。首次启动会下载 PP-OCRv6 small 检测和识别模型到项目目录 `.paddlex/`，之后重启或重建容器会复用该缓存。

`deploy.sh` 会将项目目录下的 `.paddlex/` 挂载到容器的 `/root/.paddlex/`，这是 PaddleOCR 3.x 使用的模型缓存位置。容器使用单个 worker，避免重复加载 OCR 模型造成内存占用过高。模型目录不会提交到 Git。

ARM 生产部署（Gunicorn + UvicornWorker）
非 Docker 部署首次启动多 worker 前，请先单进程初始化模型缓存；默认位置为 `~/.paddlex/`：
```bash
python -c "from ocr import ocr"
```
首次初始化需要网络连接；完成后再启动 Gunicorn：
```bash
export OCR_CONCURRENCY_PER_WORKER=1

gunicorn main:app \
  -k uvicorn.workers.UvicornWorker \
  -w 1 \
  --threads 1 \
  --timeout 120 \
  --bind 0.0.0.0:8010
```
PP-StructureV3 会在每个 worker 内各自加载一份模型，因此建议保持 `-w 1`，避免多个 worker 重复占用大量内存。请求由单个 OCR 执行线程排队处理。
---
