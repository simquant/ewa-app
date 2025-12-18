import xgboost as xgb
import os

# Tên file model của bạn
model_file = "liquidity_predictor_v1.json"

# Kiểm tra xem file có tồn tại không
if not os.path.exists(model_file):
    print(f"LOI: Khong tim thay file {model_file} trong thu muc nay!")
else:
    try:
        # Load model
        model = xgb.Booster()
        model.load_model(model_file)
        
        print("\n" + "="*30)
        print(">>> KET QUA TRA KHAO MODEL <<<")
        print("="*30)
        
        # Lấy tên các biến
        names = model.feature_names
        
        if not names:
            print("Model nay KHONG luu ten bien (ban da train bang numpy array thuan).")
            print("Ban phai xem lai code train cu hoac thu nhap theo logic kinh doanh.")
        else:
            print(f"Model yeu cau {len(names)} con so theo thu tu sau:")
            for i, name in enumerate(names):
                print(f"{i+1}. {name}")
                
        print("="*30 + "\n")
        
    except Exception as e:
        print(f"Co loi xay ra: {e}")