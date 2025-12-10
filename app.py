import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
from sqlalchemy import create_engine
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import joblib
from tensorflow.keras.models import load_model
import os

# Cấu hình trang
st.set_page_config(
    page_title="Bank Stock AI Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CẤU HÌNH DATABASE ---
DB_USER = "postgres.llthiavkzkvklakkapgz"
DB_PASSWORD = "bao19012004."
DB_HOST = "aws-1-ap-south-1.pooler.supabase.com"
DB_PORT = "5432"
DB_NAME = "postgres"

# --- CSS TÙY CHỈNH ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 8px;
    }
    div[data-testid="stMetricValue"] {
        color: #4CAF50;
    }
    .stButton>button {
        width: 100%;
        background-color: #2962FF;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- HÀM HỖ TRỢ (UTILS) ---
@st.cache_resource
def get_database_connection():
    connection_str = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(connection_str)

@st.cache_data
def get_ticker_list():
    engine = get_database_connection()
    query = """
        SELECT dt.ticker_code, dc.company_name
        FROM data_warehouse.dim_ticker dt
        LEFT JOIN data_warehouse.dim_company dc ON dt.company_key = dc.company_key
        ORDER BY dt.ticker_code
    """
    return pd.read_sql(query, engine)

@st.cache_resource
def load_ai_models():
    """Load Model LSTM và các Scaler đã train"""
    try:
        # Đường dẫn file (Sửa lại nếu bạn để trong thư mục khác)
        model = load_model('global_lstm_model.keras')
        scalers = joblib.load('scalers_dict.pkl')
        binarizer = joblib.load('ticker_binarizer.pkl')
        return model, scalers, binarizer
    except Exception as e:
        return None, None, None

def calculate_technical_indicators(df):
    """Tính toán chỉ báo kỹ thuật (Logic giống hệt lúc Train)"""
    df['ma_10'] = ta.sma(df['close_price'], length=10)
    df['ma_30'] = ta.sma(df['close_price'], length=30)
    df['rsi'] = ta.rsi(df['close_price'], length=14)
    
    macd = ta.macd(df['close_price'])
    df['macd'] = macd.iloc[:, 0] if macd is not None else 0
    
    bbands = ta.bbands(df['close_price'], length=20, std=2)
    if bbands is not None:
        df['bollinger_upper'] = bbands.iloc[:, 2]
        df['bollinger_lower'] = bbands.iloc[:, 0]
    else:
        df['bollinger_upper'] = 0
        df['bollinger_lower'] = 0
    
    # Fill NA để tránh lỗi
    df = df.fillna(method='ffill').fillna(0)
    return df

# --- TRANG 1: MARKET DASHBOARD (CODE CŨ CỦA BẠN) ---
def show_dashboard(selected_ticker):
    st.title(f"📈 Phân Tích Kỹ Thuật: {selected_ticker}")
    
    # Hàm load data riêng cho dashboard
    @st.cache_data
    def load_dashboard_data(ticker):
        engine = get_database_connection()
        query = f"""
            SELECT dd.full_date, fdt.close_price, fdt.volume, fdt.open_price, fdt.high_price, fdt.low_price,
                   fti.rsi, fti.macd, fti.ma_10, fti.ma_200, fti.bollinger_upper, fti.bollinger_lower
            FROM data_warehouse.fact_daily_trading fdt
            JOIN data_warehouse.dim_ticker dt ON fdt.ticker_key = dt.ticker_key
            JOIN data_warehouse.dim_date dd ON fdt.date_key = dd.date_key
            LEFT JOIN data_warehouse.fact_technical_indicators fti 
                ON fdt.date_key = fti.date_key AND fdt.ticker_key = fti.ticker_key
            WHERE dt.ticker_code = '{ticker}'
            ORDER BY dd.full_date ASC
        """
        df = pd.read_sql(query, engine)
        df["full_date"] = pd.to_datetime(df["full_date"])
        return df

    with st.spinner("Đang tải dữ liệu thị trường..."):
        df = load_dashboard_data(selected_ticker)

    if not df.empty:
        # --- Date Filter ---
        col_filter1, _ = st.columns([2, 1])
        with col_filter1:
            min_date = df["full_date"].min().date()
            max_date = df["full_date"].max().date()
            default_start = max_date - datetime.timedelta(days=180)
            date_range = st.slider("Thời gian:", min_date, max_date, (default_start, max_date), format="DD/MM/YYYY")

        mask = (df["full_date"].dt.date >= date_range[0]) & (df["full_date"].dt.date <= date_range[1])
        df_filtered = df.loc[mask].copy()

        # --- Metrics ---
        if not df_filtered.empty:
            last_row = df_filtered.iloc[-1]
            prev_row = df_filtered.iloc[-2] if len(df_filtered) > 1 else last_row
            change = last_row["close_price"] - prev_row["close_price"]
            pct = (change / prev_row["close_price"]) * 100
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Giá Đóng Cửa", f"{last_row['close_price']:,.0f} ₫", f"{change:,.0f} ({pct:.2f}%)")
            c2.metric("Volume", f"{last_row['volume']:,}")
            c3.metric("RSI", f"{last_row['rsi']:.1f}")
            c4.metric("Ngày", last_row["full_date"].strftime("%d/%m"))

            # --- Charts ---
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
            
            # Candlestick & Indicators
            fig.add_trace(go.Candlestick(x=df_filtered["full_date"], open=df_filtered["open_price"], high=df_filtered["high_price"],
                                         low=df_filtered["low_price"], close=df_filtered["close_price"], name="Giá"), row=1, col=1)
            if "ma_10" in df_filtered: fig.add_trace(go.Scatter(x=df_filtered["full_date"], y=df_filtered["ma_10"], line=dict(color="orange", width=1), name="MA10"), row=1, col=1)
            if "bollinger_upper" in df_filtered: 
                fig.add_trace(go.Scatter(x=df_filtered["full_date"], y=df_filtered["bollinger_upper"], line=dict(color="gray", width=0), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_filtered["full_date"], y=df_filtered["bollinger_lower"], fill='tonexty', fillcolor='rgba(255,255,255,0.1)', line=dict(color="gray", width=0), name="BB"), row=1, col=1)

            # Volume
            colors = ['green' if c >= o else 'red' for c, o in zip(df_filtered["close_price"], df_filtered["open_price"])]
            fig.add_trace(go.Bar(x=df_filtered["full_date"], y=df_filtered["volume"], marker_color=colors, name="Vol"), row=2, col=1)

            # RSI
            fig.add_trace(go.Scatter(x=df_filtered["full_date"], y=df_filtered["rsi"], line=dict(color="purple"), name="RSI"), row=3, col=1)
            fig.add_hline(y=70, line_dash="dot", row=3, col=1); fig.add_hline(y=30, line_dash="dot", row=3, col=1)

            fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

# --- TRANG 2: AI PREDICTION (TÍNH NĂNG MỚI) ---
def show_prediction_page(selected_ticker):
    st.title(f"AI Dự Báo Giá: {selected_ticker}")
    st.markdown("Sử dụng mô hình **LSTM (Long Short-Term Memory)** để dự đoán giá đóng cửa phiên tiếp theo.")

    # 1. Load Model
    model, scalers, binarizer = load_ai_models()
    
    if model is None:
        st.error("⚠️ Không tìm thấy file model/scaler! Hãy đảm bảo bạn đã chạy 'train_global_model.py' và có file .keras/.pkl")
        return

    if selected_ticker not in scalers:
        st.warning(f"⚠️ Model chưa được học mã {selected_ticker}. Vui lòng chọn mã khác hoặc train lại model.")
        return

    # 2. Lấy dữ liệu gần nhất để gợi ý input
    engine = get_database_connection()
    query = f"""
        SELECT fdt.close_price, fdt.open_price, fdt.high_price, fdt.low_price, fdt.volume, dd.full_date
        FROM data_warehouse.fact_daily_trading fdt
        JOIN data_warehouse.dim_ticker dt ON fdt.ticker_key = dt.ticker_key
        JOIN data_warehouse.dim_date dd ON fdt.date_key = dd.date_key
        WHERE dt.ticker_code = '{selected_ticker}'
        ORDER BY dd.full_date DESC LIMIT 1
    """
    last_data = pd.read_sql(query, engine)
    
    if last_data.empty:
        st.error("Không có dữ liệu lịch sử.")
        return

    last_row = last_data.iloc[0]
    st.info(f"Dữ liệu phiên gần nhất ({last_row['full_date']}): Close={last_row['close_price']:,.0f}")

    # 3. Form nhập liệu
    with st.form("predict_form"):
        st.subheader("Nhập thông số phiên hôm nay (T)")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        inp_open = col1.number_input("Open", value=float(last_row['open_price']))
        inp_high = col2.number_input("High", value=float(last_row['high_price']))
        inp_low = col3.number_input("Low", value=float(last_row['low_price']))
        inp_close = col4.number_input("Close", value=float(last_row['close_price']))
        inp_vol = col5.number_input("Volume", value=float(last_row['volume']))
        
        submitted = st.form_submit_button("DỰ BÁO NGAY")

    # 4. Xử lý dự báo
    if submitted:
        with st.spinner("AI đang phân tích dữ liệu quá khứ và tính toán..."):
            try:
                # A. Lấy 200 ngày lịch sử
                hist_query = f"""
                    SELECT fdt.close_price, fdt.open_price, fdt.high_price, fdt.low_price, fdt.volume
                    FROM data_warehouse.fact_daily_trading fdt
                    JOIN data_warehouse.dim_ticker dt ON fdt.ticker_key = dt.ticker_key
                    JOIN data_warehouse.dim_date dd ON fdt.date_key = dd.date_key
                    WHERE dt.ticker_code = '{selected_ticker}'
                    ORDER BY dd.full_date DESC LIMIT 200
                """
                df_hist = pd.read_sql(hist_query, engine)
                df_hist = df_hist.iloc[::-1].reset_index(drop=True)
                
                # Ép kiểu float
                for c in ['close_price', 'open_price', 'high_price', 'low_price', 'volume']:
                    df_hist[c] = df_hist[c].astype(float)

                # B. Ghép Input vào
                new_row = pd.DataFrame([{
                    'close_price': inp_close, 'open_price': inp_open,
                    'high_price': inp_high, 'low_price': inp_low, 'volume': inp_vol
                }])
                df_combined = pd.concat([df_hist, new_row], ignore_index=True)

                # C. Tính Indicators
                df_full = calculate_technical_indicators(df_combined)

                # D. Chuẩn bị Input cho Model
                LOOK_BACK = 60
                if len(df_full) < LOOK_BACK:
                    st.error("Dữ liệu lịch sử không đủ 60 ngày.")
                else:
                    df_input = df_full.tail(LOOK_BACK)
                    feature_cols = ['close_price', 'open_price', 'high_price', 'low_price', 'volume', 
                                    'ma_10', 'ma_30', 'rsi', 'macd', 'bollinger_upper', 'bollinger_lower']
                    
                    # Scale Numeric
                    scaler = scalers[selected_ticker]
                    numeric_vals = scaler.transform(df_input[feature_cols].values)
                    
                    # One-Hot Encoding Ticker
                    ticker_vec = binarizer.transform([selected_ticker]) # (1, n_tickers)
                    ticker_vec_repeated = np.repeat(ticker_vec, LOOK_BACK, axis=0) # (60, n_tickers)
                    
                    # Ghép lại -> (1, 60, features)
                    final_input = np.hstack([numeric_vals, ticker_vec_repeated])
                    input_reshaped = np.array([final_input])

                    # E. Dự báo
                    pred_scaled = model.predict(input_reshaped, verbose=0)
                    
                    # Inverse Scale
                    pred_matrix = np.zeros((1, len(feature_cols)))
                    pred_matrix[0, 0] = pred_scaled[0, 0]
                    final_price = scaler.inverse_transform(pred_matrix)[0, 0]

                    # F. Hiển thị kết quả
                    st.success("Dự báo thành công!")
                    
                    col_res1, col_res2 = st.columns(2)
                    
                    diff = final_price - inp_close
                    pct_diff = (diff / inp_close) * 100
                    trend_icon = "↗TĂNG" if diff > 0 else "↘GIẢM"
                    trend_color = "green" if diff > 0 else "red"

                    col_res1.metric("Giá Dự Báo (T+1)", f"{final_price:,.0f} ₫", f"{diff:,.0f} ({pct_diff:.2f}%)")
                    col_res2.markdown(f"### Xu hướng: :{trend_color}[{trend_icon}]")
                    
                    # Vẽ biểu đồ mini
                    chart_data = df_full.tail(30).copy().reset_index(drop=True)
                    # Thêm điểm dự báo
                    st.line_chart(chart_data['close_price'])

            except Exception as e:
                st.error(f"Lỗi khi dự báo: {str(e)}")

# --- MAIN APP ---
def main():
    # Sidebar
    st.sidebar.title("Menu")
    app_mode = st.sidebar.radio("Chọn chức năng:", ["Market Dashboard", "AI Price Prediction"])
    
    st.sidebar.markdown("---")
    
    # Lấy danh sách mã
    df_tickers = get_ticker_list()
    ticker_options = df_tickers["ticker_code"] + " - " + df_tickers["company_name"].fillna("")
    
    selected_option = st.sidebar.selectbox("Mã cổ phiếu:", ticker_options)
    selected_ticker = selected_option.split(" - ")[0]

    # Điều hướng trang
    if app_mode == "Market Dashboard":
        show_dashboard(selected_ticker)
    elif app_mode == "AI Price Prediction":
        show_prediction_page(selected_ticker)

if __name__ == "__main__":
    main()