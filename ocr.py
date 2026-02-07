from paddleocr import PaddleOCR
from PIL import Image
from io import BytesIO
import numpy as np
import httpx
import asyncio
from concurrent.futures import ThreadPoolExecutor

ocr_executor = ThreadPoolExecutor(max_workers=1)

async def run_ocr(img):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        ocr_executor,
        ocr.ocr,
        img
    )
http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(
        connect=5.0,
        read=10.0,
        write=10.0,
        pool=5.0,
    ),
    limits=httpx.Limits(
        max_connections=50,
        max_keepalive_connections=10
    ),
    follow_redirects=True,
)
ocr = PaddleOCR(
    use_gpu=False,
    lang='ch',                # 语言：中文
    use_angle_cls=False,      # ARM 必关, 禁用方向分类
    det_limit_side_len=640,
    det_db_unclip_ratio=1.2,
    rec_batch_num=2,
    cpu_threads=2,
    max_batch_size=1,
    enable_mkldnn=False,
    ir_optim=True,
    show_log=False,            # 生产关日志
    device="cpu",              # 指定 CPU 推理
)

def parse_ocr_result(result):
    lines = []
    for box, (text, score) in result[0]:
        lines.append({
            "text": text,
            "confidence": float(score),
            "box": [[int(x), int(y)] for x, y in box]
        })
    return lines

async def fetch_image(url: str) -> bytes:
    resp = await http_client.get(url)
    resp.raise_for_status()
    return resp.content

async def ocr_from_url(url: str) -> str:
    img_bytes = await fetch_image(url)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img = np.array(img)  # 转成 numpy.ndarray

    result = await run_ocr(img)
    data = parse_ocr_result(result)
    return {
        "texts": "\n".join([line["text"] for line in data]),
        "raw": data
    }

