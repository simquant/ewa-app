import pandas as pd
import xgboost as xgb
import os

print("--- BAT DAU HUAN LUYEN MODEL MOI ---")

# 1. Đọc dữ liệu từ file CSV (đã tạo ở bước trước)
if not os.path.exists("data_train.csv"):
    print("LOI: Khong tim thay file 'data_train.csv'. Ban da chay generate_data.py chua?")
    exit()

print("Dang doc du lieu tu data_train.csv...")
df = pd.read_csv("data_train.csv")

# 2. Định nghĩa Input (X) và Output (y)
# Đây chính là thứ tự 4 con số bạn sẽ nhập trên Web sau này
features = ['Doanh_thu', 'Chi_phi', 'Ton_kho', 'Lai_suat']
X = df[features]
y = df['Dong_tien_thuc_te']

# 3. Huấn luyện mô hình (Training)
# XGBoost sẽ tự tìm ra công thức: Dòng tiền = Doanh thu - Chi phí - ...
print(f"Dang train model voi {len(df)} dong du lieu...")
model = xgb.XGBRegressor()
model.fit(X, y)

# 4. Lưu kết quả (Save Model)
model_file = "liquidity_predictor_v1.json"
model.get_booster().save_model(model_file)

print("\n" + "="*40)
print(f">>> THANH CONG! Model da duoc luu tai: {model_file}")
print(">>> GHI NHO: Tren Web/API, ban phai nhap 4 so theo thu tu nay:")
print(f"    {features}")
print("="*40 + "\n")