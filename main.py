from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from ocr import ocr_from_url

app = FastAPI()

class OCRRequest(BaseModel):
    urls: List[str]

@app.post("/ocr")
def ocr_api(req: OCRRequest):
    res = {}
    for url in req.urls:
        try:
            res[url] = ocr_from_url(url)
        except Exception as e:
            res[url] = f"ERROR: {e}"
    
    return {
        "texts": "\n".join([data["texts"] for data in res.values()]),
        "raw":res
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")