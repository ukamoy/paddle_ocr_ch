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
    # v='PP-OCRv5',
    # text_detection_model_dir="models/PP-OCRv5_server_det_infer",
    # text_recognition_model_dir="models/PP-OCRv5_server_rec_infer",
    # text_classifier_model_dir="models/PP-LCNet_x1_0_textline_ori_infer",
    det_limit_side_len=960,
    det_db_unclip_ratio=2.8, # 数值加大后对英文空格识别有效
    rec_batch_num=4,
    cpu_threads=2,
    max_batch_size=1,
    enable_mkldnn=False,
    ir_optim=True,
    show_log=False,            # 生产关日志
    device="cpu",              # 指定 CPU 推理
)

def parse_ocr_result(result):
    if not result or result[0] is None:
        return []
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
    print(f"Image downloaded, size={len(img_bytes)} bytes")
    result = await run_ocr(img)
    data = parse_ocr_result(result)
    print(f"OCR parsed lines count={len(data)}")
    return {
        "texts": "\n".join([line["text"] for line in data]),
        "raw": data
    }

