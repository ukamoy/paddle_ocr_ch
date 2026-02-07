from paddleocr import PaddleOCR
import requests
from PIL import Image
from io import BytesIO
import numpy as np

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
    show_log=False            # 生产关日志
    device="cpu",       # 指定 CPU 推理
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

def ocr_from_url(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()

    img = Image.open(BytesIO(r.content)).convert("RGB")
    img = np.array(img)  # 转成 numpy.ndarray
    result = ocr.ocr(img)
    data = parse_ocr_result(result)
    return {
        "texts": "\n".join([line["text"] for line in data]),
        "raw": data
    }

