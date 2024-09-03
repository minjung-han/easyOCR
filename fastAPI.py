from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import easyocr
import numpy as np
from pydantic import BaseModel
import os
import torch
from scipy.signal import find_peaks

app = FastAPI()

class OCRConfig(BaseModel):
    ocr_languages: list
    use_gpu: bool = torch.cuda.is_available()

def preprocess_image(image):
    """전처리 함수: 이미지를 그레이스케일로 변환하고 이진화"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 이미지 히스토그램 계산
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.flatten()
    
    # 히스토그램에서 피크(peak) 찾기
    peaks, _ = find_peaks(hist, height=0)

    if len(peaks) >= 2:
        # 피크가 2개 이상인 경우, Otsu의 이진화 적용
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # 피크가 2개 미만인 경우, 어댑티브 이진화 적용
        binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)

    return binary_image

def perform_ocr(preprocessed_image, languages, use_gpu):
    """OCR 수행 함수"""
    try:
        reader = easyocr.Reader(languages, gpu=use_gpu)
        ocr_results = reader.readtext(preprocessed_image)
        extracted_texts = [item[1] for item in ocr_results]
        result_string = ''.join(extracted_texts)
        return result_string
    except Exception as e:
        return str(e)

@app.post("/perform_ocr/")
async def process_image(file: UploadFile = File(...), config: OCRConfig = OCRConfig(ocr_languages=['en', 'ko'])):
    try:
        # 파일 읽기 및 이미지로 변환
        image_data = await file.read()
        np_img = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # 이미지 전처리
        preprocessed_image = preprocess_image(image)

        # OCR 수행
        result = perform_ocr(preprocessed_image, config.ocr_languages, config.use_gpu)

        return JSONResponse(content={"result": result})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
