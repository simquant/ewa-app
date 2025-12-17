import xgboost as xgb
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Khởi tạo model
# Lưu ý: Vì ta đã lưu dạng Booster, ta sẽ load dạng Booster để tránh lỗi
model = xgb.Booster()

try:
    model.load_model("liquidity_predictor_v1.json")
    print(">>> THANH CONG: Da load model len Server!")
except Exception as e:
    print(f">>> LOI: Khong load duoc model. Chi tiet: {e}")

# Định nghĩa dữ liệu đầu vào (Input)
class InputData(BaseModel):
    features: List[float] # Nhận vào list các con số

@app.get("/")
def home():
    return {"message": "Server EWA dang chay ngon lanh!"}

@app.post("/predict")
def predict(data: InputData):
    try:
        # Chuyển dữ liệu thành định dạng DMatrix (của XGBoost)
        input_vector = np.array([data.features])
        dmatrix = xgb.DMatrix(input_vector)
        
        # Dự báo
        prediction = model.predict(dmatrix)[0]
        
        return {
            "status": "success",
            "predicted_cash": float(prediction),
            "currency": "VND"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}