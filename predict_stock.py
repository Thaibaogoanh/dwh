import pandas as pd
import numpy as np
import pandas_ta as ta
from sqlalchemy import create_engine
from tensorflow.keras.models import load_model
import joblib
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. INPUT CỦA BẠN (CÓ THÊM MÃ CỔ PHIẾU)
# ==============================================================================
INPUT_TICKER = 'VPB'      # <--- Input Ticker Code
INPUT_OPEN   = 27600.0
INPUT_HIGH   = 28200.0
INPUT_LOW    = 27600.0
INPUT_CLOSE  = 27800.0
INPUT_VOLUME = 11788681.0

print(f"--- Đang dự báo cho {INPUT_TICKER} (Global Model) ---")

# ==============================================================================
# 2. LOAD CHECKPOINTS
# ==============================================================================
try:
    model = load_model('global_lstm_model.keras')
    scalers_dict = joblib.load('scalers_dict.pkl')
    ticker_binarizer = joblib.load('ticker_binarizer.pkl')
    print("-> Đã load Model, Scalers và Binarizer.")
except:
    print("Lỗi: Thiếu file checkpoint. Hãy chạy train_global_model.py trước.")
    exit()

# Kiểm tra xem mã này có trong tập train không
if INPUT_TICKER not in scalers_dict:
    print(f"Lỗi: Model chưa từng học mã {INPUT_TICKER} này (Chưa có Scaler).")
    exit()

scaler = scalers_dict[INPUT_TICKER] # Lấy scaler riêng của VCB

# ==============================================================================
# 3. KẾT NỐI DB & LẤY LỊCH SỬ
# ==============================================================================
DB_USER = 'postgres.llthiavkzkvklakkapgz'
DB_PASSWORD = 'bao19012004.'
DB_HOST = 'aws-1-ap-south-1.pooler.supabase.com'
DB_PORT = '5432'
DB_NAME = 'postgres'
connection_str = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
engine = create_engine(connection_str)

query = f"""
SELECT f1.close_price, f1.open_price, f1.high_price, f1.low_price, f1.volume
FROM data_warehouse.fact_daily_trading f1
JOIN data_warehouse.dim_ticker t ON f1.ticker_key = t.ticker_key
JOIN data_warehouse.dim_date d ON f1.date_key = d.date_key
WHERE t.ticker_code = '{INPUT_TICKER}'
ORDER BY d.full_date DESC LIMIT 200
"""
df_history = pd.read_sql(query, engine)
df_history = df_history.iloc[::-1].reset_index(drop=True)

# Ép kiểu float
for c in ['close_price', 'open_price', 'high_price', 'low_price', 'volume']:
    df_history[c] = df_history[c].astype(float)

# ==============================================================================
# 4. GHÉP & TÍNH TOÁN CHỈ SỐ
# ==============================================================================
new_row = pd.DataFrame([{
    'close_price': float(INPUT_CLOSE), 'open_price': float(INPUT_OPEN),
    'high_price': float(INPUT_HIGH), 'low_price': float(INPUT_LOW),
    'volume': float(INPUT_VOLUME)
}])
df_full = pd.concat([df_history, new_row], ignore_index=True)

# Tính toán indicators (Logic y hệt file train)
df_full['ma_10'] = ta.sma(df_full['close_price'], length=10)
df_full['ma_30'] = ta.sma(df_full['close_price'], length=30)
df_full['rsi'] = ta.rsi(df_full['close_price'], length=14)
macd = ta.macd(df_full['close_price'])
df_full['macd'] = macd.iloc[:, 0] if macd is not None else 0
bbands = ta.bbands(df_full['close_price'], length=20, std=2)
if bbands is not None:
    df_full['bollinger_upper'] = bbands.iloc[:, 2]
    df_full['bollinger_lower'] = bbands.iloc[:, 0]
else:
    df_full['bollinger_upper'] = 0; df_full['bollinger_lower'] = 0

df_full = df_full.fillna(method='ffill').fillna(0)

# ==============================================================================
# 5. CHUẨN BỊ INPUT VECTOR (Numeric + One-Hot)
# ==============================================================================
LOOK_BACK = 60
df_input = df_full.tail(LOOK_BACK)

feature_cols = ['close_price', 'open_price', 'high_price', 'low_price', 'volume', 
                'ma_10', 'ma_30', 'rsi', 'macd', 'bollinger_upper', 'bollinger_lower']

# A. Scale dữ liệu số (bằng scaler riêng của VCB)
numeric_values = df_input[feature_cols].values
input_scaled_numeric = scaler.transform(numeric_values) # (60, 11)

# B. Tạo One-Hot Vector cho Ticker
# ticker_binarizer.transform trả về mảng (1, số_lượng_mã). 
# Ta cần nhân bản nó lên 60 lần để khớp với chuỗi thời gian (60, số_lượng_mã)
ticker_vec_1row = ticker_binarizer.transform([INPUT_TICKER]) # Ví dụ: [[0, 1, 0...]]
ticker_vec_60rows = np.repeat(ticker_vec_1row, LOOK_BACK, axis=0) # Lặp lại 60 lần

# C. Ghép lại
# Kết quả là mảng (60, 11 + số_lượng_mã)
final_input = np.hstack([input_scaled_numeric, ticker_vec_60rows])

# Reshape cho LSTM (1, 60, features)
input_reshaped = np.array([final_input])

# ==============================================================================
# 6. DỰ ĐOÁN
# ==============================================================================
print("-> Đang chạy Model...")
predicted_scaled = model.predict(input_reshaped, verbose=0)

# Inverse Transform
# Tạo ma trận giả khớp với số cột numeric (11 cột) để inverse
pred_matrix = np.zeros((1, len(feature_cols)))
pred_matrix[0, 0] = predicted_scaled[0, 0]
final_price = scaler.inverse_transform(pred_matrix)[0, 0]

print(f"\n===========================================")
print(f"Input: {INPUT_TICKER} - Close: {INPUT_CLOSE:,.0f}")
print(f"👉 DỰ BÁO: {final_price:,.0f} VND")
print(f"===========================================")