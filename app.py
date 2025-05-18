# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from llm_main import main
import os
from shutil import copyfileobj
from tempfile import NamedTemporaryFile

app = FastAPI()

@app.post("/analyze")
async def analyze_images(
    planogram_image: UploadFile = File(...),
    real_image: UploadFile = File(...)
):
    # Save uploaded files temporarily
    with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_planogram, \
         NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_real:
        copyfileobj(planogram_image.file, tmp_planogram)
        copyfileobj(real_image.file, tmp_real)
        planogram_path = tmp_planogram.name
        real_path = tmp_real.name

    # Replace the hardcoded paths in llm_main.py
    original_main = main
    def wrapped_main():
        return original_main(planogram_path, real_path)

    try:
        analysis, detections, tokens = wrapped_main()
        return {
            "analysis": analysis,
            "detections": detections.dict(),
            "tokens_used": tokens
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))