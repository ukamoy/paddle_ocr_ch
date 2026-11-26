import asyncio
import logging
import os
from typing import Literal

import httpx
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ocr import ocr_from_url

logger = logging.getLogger(__name__)
app = FastAPI()

OCR_CONCURRENCY_PER_WORKER = max(1, int(os.getenv("OCR_CONCURRENCY_PER_WORKER", "1")))
ocr_semaphore = asyncio.Semaphore(OCR_CONCURRENCY_PER_WORKER)


class OCRRequest(BaseModel):
    url: str
    mode: Literal["text", "markdown", "latex"] = "text"


@app.post("/ocr")
async def ocr_api(req: OCRRequest):
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")

    async with ocr_semaphore:
        try:
            res = await ocr_from_url(url, mode=req.mode)
            return res
        except httpx.HTTPStatusError as e:
            status_code = 404 if e.response.status_code == 404 else 502
            logger.warning(
                "Image URL returned HTTP %d: %s",
                e.response.status_code,
                url,
            )
            raise HTTPException(
                status_code=status_code,
                detail=f"Image URL returned HTTP {e.response.status_code}",
            ) from e
        except httpx.TimeoutException as e:
            logger.warning("Timed out downloading image URL: %s", url)
            raise HTTPException(
                status_code=504,
                detail="Timed out while downloading image from URL",
            ) from e
        except httpx.RequestError as e:
            logger.warning("Failed to download image URL %s: %s", url, e)
            raise HTTPException(
                status_code=502,
                detail="Failed to download image from URL",
            ) from e
        except Exception as e:
            logger.exception("OCR request failed")
            raise HTTPException(status_code=500, detail=f"ERROR: {e}") from e


app.mount("/", StaticFiles(directory="static", html=True), name="static")
