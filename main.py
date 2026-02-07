from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from ocr import ocr_from_url

app = FastAPI()

class OCRRequest(BaseModel):
    urls: List[str]

@app.post("/ocr")
async def ocr_api(req: OCRRequest):
    res = {}
    for url in req.urls:
        try:
            res[url] = await ocr_from_url(url)
        except Exception as e:
            res[url] = {"err": f"ERROR: {e}"}
    print(res)
    return {
        "texts": "\n".join([data["texts"] for data in res.values() if "texts" in data]),
        "raw":res
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")