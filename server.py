import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import cv2
import easyocr
import numpy as np
import os
import torch
import nest_asyncio
from scipy.signal import find_peaks


# pip install torch torchvision torchaudio
# pip install nest_asyncio

# Jupyter Notebook에서 비동기 서버 실행을 위해 필요
nest_asyncio.apply()

# FastAPI 애플리케이션 생성
app = FastAPI()

# GPU 사용 가능 여부 확인
use_gpu = torch.cuda.is_available()

# 전처리 함수
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.flatten()
    peaks, _ = find_peaks(hist, height=0)
    
    if len(peaks) >= 2:
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
    return binary_image

# OCR 수행 함수
def perform_ocr(image_path, languages=['en', 'ko']):
    try:
        image = preprocess_image(image_path)
        if not use_gpu:
            raise RuntimeError("GPU is not available. Please check your CUDA installation.")
        reader = easyocr.Reader(languages, gpu=True)  # GPU를 강제 사용하도록 설정
        ocr_results = reader.readtext(image)
        extracted_texts = [item[1] for item in ocr_results]
        return ''.join(extracted_texts)
    except Exception as e:
        return str(e)

# FastAPI 엔드포인트 정의
@app.post("/perform_ocr/")
async def process_image(image_path: str):
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    result = perform_ocr(image_path)
    return JSONResponse(content={"result": result})

# Jupyter Notebook에서 FastAPI 서버 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
