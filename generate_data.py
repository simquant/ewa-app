import pandas as pd
import numpy as np

# --- CẤU HÌNH ---
# Số lượng mẫu dữ liệu muốn tạo (1000 tháng/kỳ kinh doanh)
num_rows = 1000 

print(f"Dang bat dau sinh {num_rows} dong du lieu gia lap...")

# --- 1. SINH CÁC BIẾN ĐỘC LẬP (FEATURES) ---
# Dùng hàm random để tạo sự biến động thị trường

# a. Doanh thu: Random từ 100 triệu đến 5 tỷ VND
# Giả lập doanh nghiệp SME có quy mô vừa phải
doanh_thu = np.random.randint(100, 5000, num_rows) * 1000000

# b. Tỷ lệ Giá vốn hàng bán (COGS/Revenue): Random từ 50% đến 85%
# Biên lợi nhuận gộp thay đổi tùy thời điểm
ty_le_chi_phi = np.random.uniform(0.50, 0.85, num_rows)

# c. Tồn kho: Random từ 50 triệu đến 1 tỷ VND
ton_kho = np.random.randint(50, 1000, num_rows) * 1000000

# d. Lãi suất thị trường (Interest Rate): Biến động từ 4.5% đến 12.0%
lai_suat = np.random.uniform(4.5, 12.0, num_rows).round(2)

# --- 2. TẠO DATAFRAME ---
df = pd.DataFrame({
    'Doanh_thu': doanh_thu,
    'Ton_kho': ton_kho,
    'Lai_suat': lai_suat
})

# Tính Chi phí hoạt động dựa trên tỷ lệ đã random
df['Chi_phi'] = (df['Doanh_thu'] * ty_le_chi_phi).astype(int)

# --- 3. TÍNH BIẾN MỤC TIÊU (TARGET): DÒNG TIỀN THỰC TẾ ---
# Công thức logic: 
# Dòng tiền = (Doanh thu - Chi phí) - Chi phí vốn vay - Chi phí lưu kho + Yếu tố bất ngờ

# Giả định chi phí vốn vay chịu ảnh hưởng bởi Lãi suất * Tồn kho (vốn chôn vào hàng)
chi_phi_von = df['Ton_kho'] * (df['Lai_suat'] / 100 / 12) # Chia 12 tháng

# Yếu tố bất ngờ (Noise/Random Error) - Rất quan trọng trong Quant
# Mô phỏng khách bùng tiền, thiên tai, hoặc lộc bất ngờ... (dao động +/- 50 triệu)
noise = np.random.normal(0, 50000000, num_rows)

df['Dong_tien_thuc_te'] = (
    df['Doanh_thu'] 
    - df['Chi_phi'] 
    - chi_phi_von 
    + noise
).astype(int)

# --- 4. XỬ LÝ & LƯU FILE ---
# Sắp xếp lại thứ tự cột cho đẹp
cols = ['Doanh_thu', 'Chi_phi', 'Ton_kho', 'Lai_suat', 'Dong_tien_thuc_te']
df = df[cols]

# Lưu ra file CSV
file_name = "data_train.csv"
df.to_csv(file_name, index=False)

print("="*40)
print(f">>> THANH CONG! Da tao file '{file_name}'")
print(f">>> Xem thu 5 dong dau tien:\n")
print(df.head())
print("="*40)