import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import warnings

warnings.filterwarnings("ignore")

# 1. CẤU HÌNH
DB_USER = 'postgres.llthiavkzkvklakkapgz'
DB_PASSWORD = 'bao19012004.'
DB_HOST = 'aws-1-ap-south-1.pooler.supabase.com'
DB_PORT = '5432'
DB_NAME = 'postgres'
LOOK_BACK = 60 

connection_str = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
engine = create_engine(connection_str)

print("--- ĐANG TẢI DỮ LIỆU TOÀN BỘ NGÀNH ---")

# 2. LẤY DỮ LIỆU CỦA TẤT CẢ CÁC MÃ
query = """
SELECT 
    t.ticker_code,
    d.full_date,
    f1.close_price, f1.open_price, f1.high_price, f1.low_price, f1.volume,
    f2.ma_10, f2.ma_30, f2.rsi, f2.macd, f2.bollinger_upper, f2.bollinger_lower
FROM data_warehouse.fact_daily_trading f1
JOIN data_warehouse.fact_technical_indicators f2 
    ON f1.date_key = f2.date_key AND f1.ticker_key = f2.ticker_key
JOIN data_warehouse.dim_ticker t ON f1.ticker_key = t.ticker_key
JOIN data_warehouse.dim_date d ON f1.date_key = d.date_key
ORDER BY t.ticker_code, d.full_date ASC
"""
df = pd.read_sql(query, engine)

# === [BƯỚC SỬA LỖI QUAN TRỌNG] ===
# Ép kiểu dữ liệu từ Decimal (Object) sang Float
numeric_cols = ['close_price', 'open_price', 'high_price', 'low_price', 'volume', 
                'ma_10', 'ma_30', 'rsi', 'macd', 'bollinger_upper', 'bollinger_lower']

for col in numeric_cols:
    df[col] = df[col].astype(float)
# ==================================

df = df.dropna()

# 3. XỬ LÝ PREPROCESSING
print(f"-> Tổng số dòng dữ liệu: {len(df)}")

# A. One-Hot Encoding cho Ticker Code
ticker_binarizer = LabelBinarizer()
ticker_onehot = ticker_binarizer.fit_transform(df['ticker_code'])
num_tickers = len(ticker_binarizer.classes_)
print(f"-> Đã mã hóa {num_tickers} mã cổ phiếu.")

# B. Scaling
scalers_dict = {}
groups = df.groupby('ticker_code')

# Tạo DataFrame rỗng để chứa dữ liệu đã scale
df_scaled_numeric = pd.DataFrame(index=df.index, columns=numeric_cols)

for ticker, group in groups:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_val = scaler.fit_transform(group[numeric_cols])
    df_scaled_numeric.loc[group.index, numeric_cols] = scaled_val
    scalers_dict[ticker] = scaler

print("-> Đã Scaling dữ liệu theo từng mã.")

# C. Ghép dữ liệu
numeric_data = df_scaled_numeric.values
final_data = np.hstack([numeric_data, ticker_onehot])

# Đảm bảo toàn bộ mảng là float32 để TensorFlow nhận diện
final_data = final_data.astype('float32')

# 4. TẠO SEQUENCE (X, y) CHO LSTM
X, y = [], []

current_idx = 0
for ticker, group in groups:
    group_len = len(group)
    group_data = final_data[current_idx : current_idx + group_len]
    
    if group_len > LOOK_BACK:
        for i in range(LOOK_BACK, group_len):
            X.append(group_data[i-LOOK_BACK:i])
            y.append(group_data[i, 0]) # Close price
            
    current_idx += group_len

# Chuyển sang numpy array và ép kiểu float32 lần nữa cho chắc
X = np.array(X).astype('float32')
y = np.array(y).astype('float32')

print(f"-> Shape Input X: {X.shape} (Batch, 60, {X.shape[2]})")

# 5. TRAIN MODEL
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1)) 

model.compile(optimizer='adam', loss='mean_squared_error')

print("--- Bắt đầu Train Global Model ---")
model.fit(X, y, epochs=30, batch_size=64, verbose=1)

# 6. LƯU CHECKPOINT
model.save('global_lstm_model.keras')
joblib.dump(scalers_dict, 'scalers_dict.pkl')
joblib.dump(ticker_binarizer, 'ticker_binarizer.pkl')

print("\n=== HOÀN TẤT! ĐÃ LƯU 3 FILE CHECKPOINT ===")