import xgboost as xgb
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Load Model
model = xgb.Booster()
try:
    model.load_model("liquidity_predictor_v1.json")
    print(">>> DA LOAD MODEL THANH CONG!")
except Exception as e:
    print(f">>> LOI LOAD MODEL: {e}")

class InputData(BaseModel):
    features: List[float]

@app.get("/")
def home():
    return {"message": "EWA AI Service is Running!"}

@app.post("/predict")
def predict(data: InputData):
    try:
        # 1. Chuyển list thành mảng numpy
        input_vector = np.array([data.features])
        
        # 2. QUAN TRỌNG: Khai báo tên cột (Feature Names) cho khớp với lúc Train
        # Thứ tự phải y hệt trong file create_model.py
        feature_names = ['Doanh_thu', 'Chi_phi', 'Ton_kho', 'Lai_suat']
        
        # 3. Tạo DMatrix có gắn tên cột
        dmatrix = xgb.DMatrix(input_vector, feature_names=feature_names)
        
        # 4. Dự báo
        prediction = model.predict(dmatrix)[0]
        
        return {
            "status": "success",
            "predicted_cash": float(prediction),
            "currency": "VND",
            "note": "Du bao dua tren mo hinh tai chinh EWA"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}