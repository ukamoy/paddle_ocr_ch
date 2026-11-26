from html.parser import HTMLParser

from paddleocr import FormulaRecognition, PPStructureV3
from PIL import Image
from PIL import ImageOps
from io import BytesIO
import numpy as np
import httpx
import asyncio
import gc
import logging
import time
import re
from functools import partial
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
ocr_executor = ThreadPoolExecutor(max_workers=1)
MAX_IMAGE_SIDE = 3800  # Keep below PaddleX's 4000 px max_side_limit.
MIN_IMAGE_SIDE = 768
IMAGE_TILE_OVERLAP = 512
FORMULA_CROP_PADDING = 0.12
formula_ocr = None


def _fit_inference_image(img):
    """Guard model calls from oversized tiles without shrinking whole pages."""
    height, width = img.shape[:2]
    longest_side = max(height, width)
    if longest_side <= MAX_IMAGE_SIDE:
        return img
    scale = MAX_IMAGE_SIDE / longest_side
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    logger.info("Resizing OCR input once from %dx%d to %dx%d", width, height, *size)
    return np.asarray(
        Image.fromarray(img).resize(size, Image.Resampling.LANCZOS)
    )


async def run_ocr(img):
    loop = asyncio.get_running_loop()
    img = _fit_inference_image(img)
    predict = partial(ocr.predict, input=img, use_formula_recognition=False)
    return await loop.run_in_executor(ocr_executor, predict)


def _run_formula_ocr(img):
    """Run whole-image formula recognition as a fallback for formula-only images."""
    global formula_ocr
    if formula_ocr is None:
        logger.info("Initializing standalone formula recognition fallback")
        formula_ocr = FormulaRecognition(
            model_name="PP-FormulaNet_plus-M",
            device="cpu",
            enable_mkldnn=False,
            cpu_threads=1,
        )
    img = _fit_inference_image(img)
    formula_img = Image.fromarray(img).convert("L")
    pixels = np.asarray(formula_img)
    border = np.concatenate((pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1]))
    if np.median(border) < 127:
        logger.info("Inverting dark-background image for formula recognition")
        formula_img = ImageOps.invert(formula_img)
    formula_input = np.asarray(formula_img.convert("RGB"))
    return formula_ocr.predict(input=formula_input, batch_size=1)


async def run_formula_ocr(img):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(ocr_executor, _run_formula_ocr, img)


async def recognize_formula_blocks(img, lines):
    """Recognize formula layout blocks from padded image crops."""
    height, width = img.shape[:2]
    recovered_text = []
    for line in lines:
        if "formula" not in str(line.get("type", "")).lower():
            continue
        points = np.asarray(line.get("box") or [], dtype=float)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
            continue
        x1, y1 = np.floor(points[:, :2].min(axis=0)).astype(int)
        x2, y2 = np.ceil(points[:, :2].max(axis=0)).astype(int)
        pad_x = max(8, round((x2 - x1) * FORMULA_CROP_PADDING))
        pad_y = max(8, round((y2 - y1) * FORMULA_CROP_PADDING))
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2]
        try:
            result = await run_formula_ocr(crop)
            parsed = parse_formula_result(result)
            if parsed:
                original_text = str(line.get("text", "")).strip()
                # Keep useful prose emitted by layout parsing instead of
                # replacing the entire mixed block with the formula string.
                if any("\u3400" <= char <= "\u9fff" for char in original_text):
                    recovered_text.append(
                        {
                            **line,
                            "text": original_text,
                            "type": "text",
                        }
                    )
                line["text"] = parsed[0]["text"]
        except Exception:
            logger.exception("Formula block recognition failed; preserving layout OCR text")
    lines.extend(recovered_text)
    deduplicated = []
    for line in lines:
        text = str(line.get("text", "")).strip()
        if not text:
            continue
        if any(
            str(previous.get("text", "")).strip() == text
            and _boxes_intersect(previous.get("box", []), line.get("box", []))
            for previous in deduplicated
        ):
            continue
        deduplicated.append(line)
    formula_lines = [
        line for line in deduplicated
        if "formula" in str(line.get("type", "")).lower()
    ]
    lines = []
    for line in deduplicated:
        if "formula" in str(line.get("type", "")).lower():
            lines.append(line)
            continue
        overlapping_formula = any(
            _boxes_intersect(line.get("box", []), formula.get("box", []))
            for formula in formula_lines
        )
        if not overlapping_formula:
            lines.append(line)
            continue

        # The ordinary OCR often repeats the formula block as plain text.
        # Keep any prose before the first inline math expression, while the
        # formula result itself is emitted by the formula line.
        text = str(line.get("text", "")).strip()
        math_start = re.search(
            r"(?:[A-Za-z]\s*=\s*[\{\[(]|[A-Za-z]\s*\([^)]*\)\s*=|[∈∑∏])",
            text,
        )
        has_chinese = any("\u3400" <= char <= "\u9fff" for char in text)
        if math_start:
            text = text[:math_start.start()].strip(" ，,。:：;；")
        elif not has_chinese:
            text = ""
        if text:
            lines.append({**line, "text": text})
    return sorted(
        lines,
        key=lambda line: (
            min((point[1] for point in line.get("box", [])), default=0),
            min((point[0] for point in line.get("box", [])), default=0),
            "formula" in str(line.get("type", "")).lower(),
        ),
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
ocr = PPStructureV3(
    layout_detection_model_name="PP-DocLayout-M",
    text_detection_model_name="PP-OCRv6_small_det",
    text_recognition_model_name="PP-OCRv6_small_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    use_seal_recognition=False,
    use_chart_recognition=False,
    use_table_recognition=True,
    # Formula blocks are recognized separately by FormulaRecognition so this
    # pipeline does not initialize PaddleX's integrated formula component.
    use_formula_recognition=False,
    formula_recognition_model_name="PP-FormulaNet_plus-M",
    formula_recognition_batch_size=1,
    enable_mkldnn=False,
    cpu_threads=1,
    device="cpu",
)


def parse_ocr_result(result):
    """Convert PP-StructureV3 layout blocks to API-friendly content."""
    lines = []
    for page in result:
        page_result = page.json if hasattr(page, "json") else page
        if isinstance(page_result, dict) and "res" in page_result:
            page_result = page_result["res"]
        blocks = page_result.get("parsing_res_list", []) if isinstance(page_result, dict) else []
        page_line_count = len(lines)
        for block in blocks:
            text = str(block.get("block_content") or "").strip()
            block_type = block.get("block_label", "text")
            if not text and "formula" not in str(block_type).lower():
                continue
            box = _normalize_bbox(block.get("block_bbox"))
            lines.append(
                {
                    "text": text,
                    "type": block_type,
                    "confidence": None,
                    "box": box,
                }
            )

        # Formula results are exposed separately from parsing_res_list in
        # PP-StructureV3. Some formula-only pages have no layout text blocks.
        formula_results = page_result.get("formula_res_list", []) if isinstance(page_result, dict) else []
        existing_formulas = {
            line["text"].strip("$")
            for line in lines[page_line_count:]
            if "formula" in str(line.get("type", "")).lower()
        }
        for formula_result in formula_results:
            formula = str(formula_result.get("rec_formula") or "").strip()
            if not formula or formula.strip("$") in existing_formulas:
                continue
            lines.append(
                {
                    "text": formula,
                    "type": "formula",
                    "confidence": None,
                    "box": _normalize_bbox(formula_result.get("rec_polys")),
                }
            )
            existing_formulas.add(formula.strip("$"))

        # A mixed Chinese/text-and-math line can be classified as one formula
        # block. Preserve ordinary OCR fragments inside it before formula OCR
        # replaces the block content with LaTeX.
        if isinstance(page_result, dict):
            formula_boxes = [
                line["box"]
                for line in lines[page_line_count:]
                if "formula" in str(line.get("type", "")).lower()
            ]
            ocr_result = page_result.get("overall_ocr_res") or {}
            texts = ocr_result.get("rec_texts", [])
            scores = ocr_result.get("rec_scores", [])
            boxes = ocr_result.get("dt_polys", [])
            for index, text in enumerate(texts):
                text = str(text or "").strip()
                box = _normalize_bbox(boxes[index]) if index < len(boxes) else []
                if not text or not any(_boxes_intersect(box, item) for item in formula_boxes):
                    continue
                duplicate = any(
                    line.get("text", "").strip() == text
                    and _boxes_intersect(box, line.get("box", []))
                    for line in lines[page_line_count:]
                )
                if duplicate:
                    continue
                score = scores[index] if index < len(scores) else None
                lines.append(
                    {
                        "text": text,
                        "type": "text",
                        "confidence": float(score) if score is not None else None,
                        "box": box,
                    }
                )

            page_lines = lines[page_line_count:]
            page_lines.sort(
                key=lambda line: (
                    min((point[1] for point in line["box"]), default=0),
                    min((point[0] for point in line["box"]), default=0),
                )
            )
            lines[page_line_count:] = page_lines

        # Layout parsing can return no blocks even when the general OCR stage
        # recognized text. Preserve those OCR results instead of returning empty.
        if len(lines) == page_line_count and isinstance(page_result, dict):
            ocr_result = page_result.get("overall_ocr_res") or {}
            texts = ocr_result.get("rec_texts", [])
            scores = ocr_result.get("rec_scores", [])
            boxes = ocr_result.get("dt_polys", [])
            for index, text in enumerate(texts):
                text = str(text or "").strip()
                if not text:
                    continue
                score = scores[index] if index < len(scores) else None
                lines.append(
                    {
                        "text": text,
                        "type": "text",
                        "confidence": float(score) if score is not None else None,
                        "box": _normalize_bbox(boxes[index]) if index < len(boxes) else [],
                    }
                )
    return lines


def _tile_starts(length: int) -> list[int]:
    if length <= MAX_IMAGE_SIDE:
        return [0]
    stride = MAX_IMAGE_SIDE - IMAGE_TILE_OVERLAP
    starts = [0]
    while starts[-1] + MAX_IMAGE_SIDE < length:
        starts.append(starts[-1] + stride)
    return starts


async def run_ocr_on_tiles(img):
    """Run layout/OCR on overlapping native-resolution tiles and merge boxes."""
    height, width = img.shape[:2]
    x_starts = _tile_starts(width)
    y_starts = _tile_starts(height)
    if len(x_starts) * len(y_starts) > 1:
        logger.info(
            "Splitting %dx%d image into %d OCR tiles",
            width,
            height,
            len(x_starts) * len(y_starts),
        )

    merged = []
    for y1 in y_starts:
        for x1 in x_starts:
            tile = img[y1:min(y1 + MAX_IMAGE_SIDE, height), x1:min(x1 + MAX_IMAGE_SIDE, width)]
            result = await run_ocr(tile)
            tile_lines = parse_ocr_result(result)
            del result
            for line in tile_lines:
                box = line.get("box") or []
                if box:
                    line["box"] = [
                        [int(point[0]) + x1, int(point[1]) + y1]
                        for point in box
                    ]
                duplicate = any(
                    line["text"] == previous["text"]
                    and _boxes_intersect(line.get("box", []), previous.get("box", []))
                    for previous in merged
                )
                if not duplicate:
                    merged.append(line)
    return sorted(
        merged,
        key=lambda line: (
            min((point[1] for point in line.get("box", [])), default=0),
            min((point[0] for point in line.get("box", [])), default=0),
        ),
    )


def parse_formula_result(result):
    formulas = []
    for page in result:
        page_result = page.json if hasattr(page, "json") else page
        if isinstance(page_result, dict) and "res" in page_result:
            page_result = page_result["res"]
        if not isinstance(page_result, dict):
            continue
        formula = str(page_result.get("rec_formula") or "").strip()
        if formula:
            formulas.append(
                {
                    "text": formula,
                    "type": "formula",
                    "confidence": None,
                    "box": [],
                }
            )
    return formulas


def _normalize_bbox(box):
    """Normalize either [x1, y1, x2, y2] or polygon coordinates."""
    if box is None:
        return []
    try:
        coords = np.asarray(box, dtype=float)
    except (TypeError, ValueError):
        return []

    if coords.size == 4 and coords.ndim == 1:
        x1, y1, x2, y2 = coords
        points = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    elif coords.ndim == 2 and coords.shape[1] >= 2:
        points = coords[:, :2]
    elif coords.ndim == 1 and coords.size >= 6 and coords.size % 2 == 0:
        points = coords.reshape(-1, 2)
    else:
        return []

    return [[int(x), int(y)] for x, y in points]


def _boxes_intersect(box_a, box_b):
    """Return whether two normalized polygon boxes overlap."""
    if not box_a or not box_b:
        return False
    a = np.asarray(box_a, dtype=float)
    b = np.asarray(box_b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] < 2 or b.shape[1] < 2:
        return False
    return not (
        a[:, 0].max() < b[:, 0].min()
        or b[:, 0].max() < a[:, 0].min()
        or a[:, 1].max() < b[:, 1].min()
        or b[:, 1].max() < a[:, 1].min()
    )


def _json_safe(value):
    """Convert NumPy values in OCR output to JSON-serializable Python types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class _TableMarkdownParser(HTMLParser):
    """Small HTML table reader for table HTML emitted by PP-StructureV3."""

    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None
        self.in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = []
            self.in_cell = True

    def handle_data(self, data):
        if self.in_cell and self.cell is not None:
            self.cell.append(data.strip())

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.in_cell and self.row is not None and self.cell is not None:
            self.row.append(" ".join(part for part in self.cell if part))
            self.cell = None
            self.in_cell = False
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def _table_html_to_markdown(html: str) -> str:
    parser = _TableMarkdownParser()
    parser.feed(html)
    rows = parser.rows
    if not rows:
        return html
    width = max(map(len, rows))
    rows = [row + [""] * (width - len(row)) for row in rows]
    escaped = [[cell.replace("|", r"\|").replace("\n", " ") for cell in row] for row in rows]
    separator = ["---"] * width
    return "\n".join(
        ["| " + " | ".join(escaped[0]) + " |", "| " + " | ".join(separator) + " |"]
        + ["| " + " | ".join(row) + " |" for row in escaped[1:]]
    )


def _lines_to_text(lines):
    parts = []
    for line in lines:
        content = line["text"]
        if "table" in str(line.get("type", "")).lower():
            parser = _TableMarkdownParser()
            parser.feed(content)
            if parser.rows:
                content = "\n".join("\t".join(row) for row in parser.rows)
        parts.append(content)
    return "\n".join(parts)


def _same_visual_line(box_a, box_b) -> bool:
    """Whether two boxes occupy the same text row based on vertical overlap."""
    if not box_a or not box_b:
        return False
    a = np.asarray(box_a, dtype=float)
    b = np.asarray(box_b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] < 2 or b.shape[1] < 2:
        return False
    a_top, a_bottom = a[:, 1].min(), a[:, 1].max()
    b_top, b_bottom = b[:, 1].min(), b[:, 1].max()
    overlap = min(a_bottom, b_bottom) - max(a_top, b_top)
    smaller_height = min(a_bottom - a_top, b_bottom - b_top)
    return smaller_height > 0 and overlap >= smaller_height * 0.5


def _normalize_inline_math(text: str) -> str:
    """Normalize punctuation and Unicode subscripts in a detected math span."""
    text = text.replace("，", ",").replace("、", ",").replace("⋯", r"\dots").replace("…", r"\dots")
    # Convert visible set braces before adding braces for LaTeX subscripts.
    text = re.sub(r"(?<!\\)\{([^{}]*,[^{}]*)\}", r"\\{\1\\}", text)

    subscript_digits = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    text = re.sub(
        r"([A-Za-zΑ-ω])([₀₁₂₃₄₅₆₇₈₉]+)",
        lambda match: f"{match.group(1)}_{{{match.group(2).translate(subscript_digits)}}}",
        text,
    )
    return text


def _wrap_inline_math(text: str) -> str:
    """Wrap recognizable math spans in inline Markdown math delimiters."""
    math_operators = r"=<>≤≥∈∉≠≈±−×÷·⋅∑∫√→↔∧∨|+*/^_"
    function_expression = (
        r"(?<![$A-Za-zΑ-ω0-9₀-₉])(?:[A-Za-zΑ-ω][A-Za-zΑ-ω0-9]*\s*)?"
        rf"\([^()\n]*[{math_operators}][^()\n]*\)"
        rf"(?:\s*[,，、;；]\s*(?:[A-Za-zΑ-ω][A-Za-zΑ-ω0-9]*\s*)?"
        rf"\([^()\n]*[{math_operators}][^()\n]*\))*(?!\$)"
    )
    relation_expression = (
        rf"(?<![$A-Za-zΑ-ω0-9₀-₉])[A-Za-zΑ-ω0-9₀-₉]+"
        rf"(?:\s*[{math_operators}]\s*"
        rf"[A-Za-zΑ-ω0-9₀-₉{{}}(),.\[\]，、⋯…]+)+(?!\$)"
    )
    math_span = re.compile(f"{function_expression}|{relation_expression}")
    return math_span.sub(
        lambda match: f"${_normalize_inline_math(match.group(0).strip())}$",
        text,
    )


def blocks_to_markdown(lines):
    parts = []
    for line in lines:
        content = line["text"]
        block_type = line.get("type", "text").lower()
        if "table" in block_type:
            content = _table_html_to_markdown(content)
        elif "formula" in block_type:
            inline = any(
                "formula" not in str(other.get("type", "")).lower()
                and _same_visual_line(line.get("box", []), other.get("box", []))
                for other in lines
            )
            delimiter = "$" if inline else "$$"
            content = f"{delimiter}{content.strip('$')}{delimiter}"
        else:
            content = _wrap_inline_math(content)
        parts.append((content, line))

    rendered = []
    for index, (content, line) in enumerate(parts):
        if index == 0:
            rendered.append(content)
            continue
        previous = parts[index - 1][1]
        same_line = _same_visual_line(
            line.get("box", []), previous.get("box", [])
        )
        separator = " " if same_line else "\n\n"
        rendered.append(separator + content)
    return "".join(rendered)


async def fetch_image(url: str) -> bytes:
    resp = await http_client.get(url)
    resp.raise_for_status()
    return resp.content

async def ocr_from_url(url: str, mode: str = "text") -> dict:
    if mode not in {"text", "markdown", "latex"}:
        raise ValueError("mode must be one of: text, markdown, latex")
    latex = mode == "latex"
    img_bytes = await fetch_image(url)

    logger.info("Image downloaded, size=%d bytes", len(img_bytes))
    try:
        with Image.open(BytesIO(img_bytes)) as image:
            width, height = image.size
            # Preserve transparent drawings and glyphs by placing them on a
            # white canvas before converting to RGB.
            if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                rgba = image.convert("RGBA")
                white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                image = Image.alpha_composite(white, rgba).convert("RGB")
            else:
                image = image.convert("RGB")
            longest_side = max(image.size)
            if longest_side < MIN_IMAGE_SIDE:
                scale = MIN_IMAGE_SIDE / longest_side
                resized = (
                    max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)),
                )
                image = image.resize(resized, Image.Resampling.LANCZOS)
            img = np.array(image)
    except Exception:
        logger.exception("Failed to decode downloaded image")
        raise
    del img_bytes

    logger.info(
        "Image dimensions: %dx%d (processed at %dx%d)",
        width,
        height,
        img.shape[1],
        img.shape[0],
    )
    inference_started = time.perf_counter()
    # Let PP-Structure locate and classify regions, then run formula OCR only
    # on formula crops. This avoids enabling the broken integrated formula
    # pipeline for the entire page.
    data = await run_ocr_on_tiles(img)
    if latex:
        data = await recognize_formula_blocks(img, data)
    has_formula_block = any(
        "formula" in str(line.get("type", "")).lower()
        for line in data
    )
    if latex and not has_formula_block:
        logger.info(
            "No formula blocks found in layout output; trying whole-image formula OCR"
        )
        formula_result = await run_formula_ocr(img)
        fallback_formulas = parse_formula_result(formula_result)
        if fallback_formulas:
            # Layout OCR can return a flattened equation as ordinary text.
            # Prefer the formula model's structural result when available.
            data = fallback_formulas
        del formula_result
    inference_seconds = time.perf_counter() - inference_started
    del img
    gc.collect()
    logger.info(
        "OCR completed in %.2f seconds; parsed blocks=%d",
        inference_seconds,
        len(data),
    )
    plain_text = _lines_to_text(data)
    markdown = blocks_to_markdown(data)
    return {
        # Keep the legacy field; its value follows the requested output mode.
        "texts": plain_text if mode == "text" else markdown,
        "mode": mode,
        "raw": _json_safe(data)
    }
