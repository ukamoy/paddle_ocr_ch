import asyncio
import os

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ocr import ocr_from_url

app = FastAPI()

OCR_CONCURRENCY_PER_WORKER = max(1, int(os.getenv("OCR_CONCURRENCY_PER_WORKER", "1")))
ocr_semaphore = asyncio.Semaphore(OCR_CONCURRENCY_PER_WORKER)


class OCRRequest(BaseModel):
    url: str


@app.post("/ocr")
async def ocr_api(req: OCRRequest):
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")

    async with ocr_semaphore:
        try:
            res = await ocr_from_url(url)
            print("main:", res)
            return res
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ERROR: {e}") from e


app.mount("/", StaticFiles(directory="static", html=True), name="static")
