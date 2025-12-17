import xgboost as xgb
import numpy as np
import pandas as pd

# 1. Tạo dữ liệu giả (Dummy Data)
X = np.random.rand(100, 10) # 100 dòng, 10 đặc trưng
y = np.random.rand(100) * 1000000 # Doanh số ngẫu nhiên

# 2. Train model nhanh
model = xgb.XGBRegressor()
model.fit(X, y)

# 3. Lưu model ra file JSON
model.get_booster().save_model("liquidity_predictor_v1.json")
print("Da tao xong file model: liquidity_predictor_v1.json")