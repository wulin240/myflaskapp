from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import urllib.parse
from datetime import datetime, timedelta

app = Flask(__name__)

# ----------------- Supabase 設定 -----------------
# 請將這裡替換成您的 Supabase 專案資訊
SUPABASE_URL = "https://djhdpltrhlhqfxmwniki.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRqaGRwbHRyaGxocWZ4bXduaWtpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjMwNjQwODgsImV4cCI6MjA3ODY0MDA4OH0.jwSPe-HMHxv2xGCjS42O5Cjby0KtgsHEStlQWs0cyPk"
TABLE_NAME = "stock_data"
FAVORITE_TABLE = "favorites"

headers = {
    "apikey": SUPABASE_KEY.strip(),
    "Authorization": f"Bearer {SUPABASE_KEY.strip()}"
}

# ----------------- 輔助函數：最愛股票檢查 (修正位置：移到所有路由之前) -----------------
def is_favorite(stock_id):
    try:
        # 使用 headers 確保授權
        res = requests.get(f"{SUPABASE_URL}/rest/v1/{FAVORITE_TABLE}", headers=headers, params={"stock_id": f"eq.{stock_id}"}, timeout=10)
        res.raise_for_status()
        return len(res.json()) > 0
    except Exception as e:
        # 即使 Supabase 連線失敗，也不應該影響主程式執行
        print(f"⚠️ 檢查最愛失敗: {e}")
        return False
        
# ----------------- 抓取股票資料 -----------------
def fetch_stock_data(stock_id):
    stock_id_clean = stock_id.replace(".TW","").replace(".TWO","")
    params = {"stock_id": f"eq.{stock_id_clean}", "order": "date.asc", "select": "*, stock_name"} 

    try:
        res = requests.get(
            f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}", headers=headers, params=params, timeout=30
        )
        res.raise_for_status()
        data = res.json()
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"⚠️ Supabase 讀取 {stock_id} 失敗: {e}")
        return pd.DataFrame()

# ----------------- 數據處理核心功能 -----------------

def convert_to_weekly(df_daily):
    """將日線數據 (OHLCV) 轉換為週線數據。"""
    if df_daily.empty:
        return df_daily
    
    df = df_daily.set_index('date')
    weekly_data = df.resample('W').agg({
        'open': 'first',        
        'high': 'max',          
        'low': 'min',           
        'close': 'last',        
        'volume': 'sum'        
    })
    
    df_weekly = weekly_data.dropna(subset=['open'])
    df_weekly = df_weekly.reset_index()
    if not df_daily.empty and 'stock_name' in df_daily.columns:
        df_weekly['stock_name'] = df_daily['stock_name'].iloc[-1]
    if not df_daily.empty and 'stock_id' in df_daily.columns:
        df_weekly['stock_id'] = df_daily['stock_id'].iloc[-1]
    
    return df_weekly


def kline_merge(df):
    """🌟 修正 K線合併：採用更穩健的包含關係判定邏輯，處理邊界情況。"""
    if df.empty: return df
    df_raw = df.copy()
    processed_kline = []
    
    # 設置索引為 date (如果尚未設置)
    df_raw = df_raw.set_index('date')  
    
    current_kline = {col: df_raw.iloc[0][col] for col in ['open', 'high', 'low', 'close', 'volume']}
    current_kline['Index'] = df_raw.index[0] # 記錄合併 K 線的日期

    for i in range(1, len(df_raw)):
        next_row = df_raw.iloc[i]
        
        next_kline = {col: next_row[col] for col in ['open', 'high', 'low', 'close', 'volume']}
        
        # 判斷是否為包含關係：第二根 K 線被第一根 K 線包含 或 第二根 K 線包含第一根 K 線
        is_inclusion = (
            (next_row['high'] <= current_kline['high'] and next_row['low'] >= current_kline['low']) or
            (next_row['high'] >= current_kline['high'] and next_row['low'] <= current_kline['low'])
        )
        
        if is_inclusion:
            # 合併：更新高點/低點 (取極值)
            current_kline['high'] = max(current_kline['high'], next_row['high'])
            current_kline['low'] = min(current_kline['low'], next_row['low'])
            # 合併：成交量必須累加
            current_kline['volume'] += next_row['volume']
            # 合併：收盤價以**被包含 K 線的收盤價**為準（確保時間軸終點正確）
            current_kline['close'] = next_row['close']  
            # 🌟 修正邊界：更新合併 K 線的日期為最新 K 線的日期
            current_kline['Index'] = df_raw.index[i]  
        else:
            # 不包含：將當前合併 K 線加入結果，並開始新的 K 線
            processed_kline.append(current_kline)
            current_kline = next_kline
            current_kline['Index'] = df_raw.index[i]
            
    processed_kline.append(current_kline)
    
    # 恢復 date 欄位
    df_merged = pd.DataFrame(processed_kline).set_index('Index').rename_axis('date').reset_index()
    df_merged['date'] = pd.to_datetime(df_merged['date'])  
    return df_merged


def find_divergence(df_merged):
    """🌟 修正分型判斷：採用包含或等於的寬鬆標準。"""
    df = df_merged.copy()
    
    df['H_prev'], df['H_next'] = df['high'].shift(1), df['high'].shift(-1)
    df['L_prev'], df['L_next'] = df['low'].shift(1), df['low'].shift(-1)

    # 🌟 寬鬆頂分型：中間 K 線高點 >= 兩側 K 線高點
    df['Is_Top_Divergence'] = (df['high'] >= df['H_prev']) & (df['high'] >= df['H_next'])
    # 🌟 寬鬆底分型：中間 K 線低點 <= 兩側 K 線低點
    df['Is_Bottom_Divergence'] = (df['low'] <= df['L_prev']) & (df['low'] <= df['L_next'])

    df['Is_Top_Divergence'] = df['Is_Top_Divergence'].fillna(False)
    df['Is_Bottom_Divergence'] = df['Is_Bottom_Divergence'].fillna(False)
    
    df['Top_Price'] = np.where(df['Is_Top_Divergence'], df['high'], np.nan)
    df['Bottom_Price'] = np.where(df['Is_Bottom_Divergence'], df['low'], np.nan)
    return df


def filter_pivots_for_stroke(df_result, df_original):
    """過濾連續轉折點，並將分型結果合併回原始K線數據 (優化版)
    🌟 返回最後一個有效分型點的日期和類型，用於實時筆段延伸。
    """
    
    df_original['date'] = pd.to_datetime(df_original['date'])
    if df_original.empty:  
        df_original['Pivot_Type'] = 0
        df_original['Pivot_Price'] = np.nan
        return df_original, None, 0 # (df_final, last_pivot_date, last_pivot_type)

    pivot_points = df_result[df_result['Is_Top_Divergence'] | df_result['Is_Bottom_Divergence']].copy()

    if pivot_points.empty:  
        df_original['Pivot_Type'] = 0
        df_original['Pivot_Price'] = np.nan
        return df_original, None, 0
        
    # 應用連續轉折點過濾（確保頂底頂底交替）
    pivot_points['Type'] = np.where(pivot_points['Is_Top_Divergence'], 1, -1)
    final_pivots_list = []
    last_type = 0
    last_date = None
    last_price = np.nan

    for idx, row in pivot_points.iterrows():
        current_type = row['Type']
        if current_type != last_type:
            row['Pivot_Price_Calc'] = row['Top_Price'] if row['Type'] == 1 else row['Bottom_Price']
            final_pivots_list.append(row)
            last_type = current_type
            last_date = row['date']
            last_price = row['Pivot_Price_Calc']

    df_filtered = pd.DataFrame(final_pivots_list)
    
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])

    df_pivot_data = df_filtered[['date', 'Type', 'Pivot_Price_Calc']].rename(columns={
        'Type': 'Pivot_Type',
        'Pivot_Price_Calc': 'Pivot_Price'
    })
    
    # 將分型結果合併回原始數據 (df_original)
    df_merged = df_original.merge(
        df_pivot_data,
        on='date',  
        how='left'
    )
    
    df_merged['Pivot_Type'] = df_merged['Pivot_Type'].fillna(0).astype(int)
    
    return df_merged, last_date, last_type

def analyze_trend_by_pivots(pivot_df):
    """基於有效轉折點判斷頂底趨勢 (HH/HL)"""
    if pivot_df.empty or len(pivot_df) < 4:  
        return {'Overall_Trend': "結構數據不足 (需至少四個有效轉折點)"}

    tops = pivot_df[pivot_df['Pivot_Type'] == 1]['Pivot_Price'].dropna()
    bottoms = pivot_df[pivot_df['Pivot_Type'] == -1]['Pivot_Price'].dropna()

    if len(tops) < 2 or len(bottoms) < 2:
        return {'Overall_Trend': "結構數據不足 (需至少兩個頂點和兩個底點)"}

    T2, T1 = tops.iloc[-1], tops.iloc[-2]
    B2, B1 = bottoms.iloc[-1], bottoms.iloc[-2]

    is_hh, is_hl = T2 > T1, B2 > B1
    is_lh, is_ll = T2 < T1, B2 < B1

    trend_result = "盤整/待確認"
    if is_hh and is_hl: trend_result = "✅ 上升趨勢 (Higher Highs & Higher Lows)"
    elif is_lh and is_ll: trend_result = "🔻 下降趨勢 (Lower Highs & Lower Lows)"
    elif is_hh and is_ll: trend_result = "⚠️ 擴張結構 (高點抬高, 低點降低)"
    elif is_lh and is_hl: trend_result = "⏳ 收斂結構 (高點降低, 低點抬高)"
        
    return {'Overall_Trend': trend_result}

def check_rebound_signal(df_full_processed, trend_period=90):
    """結構回調起漲信號檢查"""
    if len(df_full_processed) < trend_period + 5:
        return False, "數據不足以判斷長線趨勢"

    df_check = df_full_processed.iloc[-trend_period:].copy()
    pivot_df = df_check[df_check['Pivot_Type'] != 0].copy()
    current = df_check.iloc[-1]
    prev = df_check.iloc[-2]

    # --- 1. 結構趨勢確認 (Stage I) ---
    trend_result = analyze_trend_by_pivots(pivot_df)['Overall_Trend']
    is_high_level_trend = ('上升趨勢' in trend_result)
    is_ma_aligned = (df_check['MA60'].iloc[-1] > df_check['MA60'].iloc[0]) and (current['close'] > current['MA60'])
    
    if not (is_high_level_trend and is_ma_aligned):
        return False, f"❌ 長線趨勢不符合 HH/HL 上升結構 ({trend_result})"

    # --- 2. 回調結構定位與確認 (Stage II) ---
    bottoms = pivot_df[pivot_df['Pivot_Type'] == -1]['Pivot_Price'].dropna()
    tops = pivot_df[pivot_df['Pivot_Type'] == 1]['Pivot_Price'].dropna()
    
    if len(bottoms) < 2 or len(tops) < 1:
        return False, "結構轉折點不足，無法定位回調區間"

    T_last = tops.iloc[-1]
    B_pre_T = bottoms.iloc[-2]  

    is_correcting = (current['close'] < T_last)
    is_holding_support = (current['low'] > B_pre_T)
    
    if not (is_correcting and is_holding_support):
        if current['close'] > T_last:
            return False, "✅ 已經突破前高，回調已結束，屬於新的上漲波段"
        return False, f"🚨 結構性回調失敗：低點已跌破結構支撐 B_pre_T ({B_pre_T:.2f})"


    # --- 3. 起漲信號 (Stage III) ---
    is_bullish_engulfing = (
        (current['close'] > current['open']) and  
        (current['close'] > prev['open']) and  
        (current['open'] < prev['close'])
    )
    is_rebound_confirmed = (
        current['close'] > current['MA20']  
        and (current['close'] > prev['high'] or is_bullish_engulfing)
    )

    if is_rebound_confirmed:
        return True, "✅ **【結構回調起漲信號】**：價格在 B_pre_T 支撐上確認反轉！"
    else:
        return False, "💡 **潛在起漲提示**：結構已確認為健康回調區間，等待強勢 K 線確認起漲！"


# ----------------- 整合生成圖表 (含趨勢分析和訊號檢查) -----------------
def generate_chart(stock_id_clean, start_date=None, end_date=None, simple_mode=False, num_rows=30, frequency='D'):
    df_original = fetch_stock_data(stock_id_clean)
    if df_original.empty: return None, f"{stock_id_clean} 無資料", "N/A", "N/A", "neutral" # 🌟 新增 trend_class 預設值

    df_full = df_original.copy()
    
    # === 數據頻率轉換 ===
    if frequency == 'W':
        df_full = convert_to_weekly(df_full)
    # ==================
    
    if start_date and end_date:
        df_full = df_full[
            (df_full['date'] >= pd.to_datetime(start_date)) &
            (df_full['date'] <= pd.to_datetime(end_date))
        ]

    if df_full.empty: return None, f"{stock_id_clean} 在 {start_date} ~ {end_date} 無資料", "N/A", "N/A", "neutral" # 🌟 新增 trend_class 預設值

    # --- 1. 技術指標計算 --- 
    df_tech = df_full.copy()
    # TP (Typical Price) 必須計算
    df_tech['TP'] = (df_tech['high'] + df_tech['low'] + df_tech['close']) / 3
    df_tech['line'] = df_tech.apply(lambda row: row['high'] if row['close'] > row['open'] else (row['low'] if row['close'] < row['open'] else (row['open'] + row['close']) / 2), axis=1)
    for ma in [5, 10, 20, 60]: df_tech[f"MA{ma}"] = df_tech['close'].rolling(ma).mean()
    df_tech['VOL5'] = df_tech['volume'].rolling(5).mean()
    df_tech['VOL20'] = df_tech['volume'].rolling(20).mean()
    df_tech['H-L'] = df_tech['high'] - df_tech['low']
    df_tech['H-PC'] = abs(df_tech['high'] - df_tech['close'].shift(1))
    df_tech['L-PC'] = abs(df_tech['low'] - df_tech['close'].shift(1))
    df_tech['TR'] = df_tech[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df_tech['ATR14'] = df_tech['TR'].rolling(14).mean().round(3)
    df_tech['stop_loss'] = df_tech['low'] - df_tech['ATR14'].fillna(0)
    
    # --- 2. 纏論轉折點處理 ---
    df_merged = kline_merge(df_tech.copy())
    df_divergence = find_divergence(df_merged)
    # 🌟 修改：接收最後一個有效分型點的日期和類型
    df_final, last_pivot_date, last_pivot_type = filter_pivots_for_stroke(df_divergence, df_tech.copy())

    # --- 3. 趨勢分析與信號檢查 ---
    df_display = df_final.tail(num_rows).copy()
    pivot_df_full = df_final[df_final['Pivot_Type'] != 0].copy()  
    
    trend_analysis = analyze_trend_by_pivots(pivot_df_full)
    is_rebound, rebound_desc = check_rebound_signal(df_final)

    # 確保取得最終的趨勢描述
    trend_desc_final = trend_analysis['Overall_Trend']
    
    # --- 🌟 關鍵修正：計算簡化趨勢分類 (Trend Class) 🌟 ---
    trend_class = 'neutral'
    
    # 看空判斷 (包括 '潛在趨勢反轉/持續下降 (下穿前底)' 的關鍵詞)
    if '下降趨勢' in trend_desc_final or '下穿前底' in trend_desc_final or '持續下降' in trend_desc_final or '潛在趨勢反轉' in trend_desc_final:
        trend_class = 'bearish' # 綠色
        
    # 看多判斷
    elif '上升趨勢' in trend_desc_final or '上穿前高' in trend_desc_final or '趨勢持續' in trend_desc_final:
        trend_class = 'bullish' # 紅色
        
    # ---------------------------------------------
    
    # 🌟 VWAP：重新計算，僅限於 df_display 範圍 (num_rows 筆)
    df_display['TPV_display'] = df_display['TP'] * df_display['volume']
    df_display['VWAP'] = df_display['TPV_display'].cumsum() / df_display['volume'].cumsum()
    
    # --- 4. 繪製圖表 ---  
    min_price = df_display[['low', 'MA5', 'MA10', 'MA20', 'MA60', 'VWAP']].min().min()
    max_price = df_display[['high', 'MA5', 'MA10', 'MA20', 'MA60', 'VWAP']].max().max()
    price_range = max_price - min_price
    yaxis_min = min_price - price_range / 4
    yaxis_max = max_price + price_range / 4

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        row_heights=[0.7, 0.15, 0.15],
        subplot_titles=(f"K線圖 ({frequency}線, 含纏論分型)", "成交量", "ATR")
    )

    # K線圖與指標
    fig.add_trace(go.Candlestick(x=df_display['date'], open=df_display['open'], high=df_display['high'], low=df_display['low'], close=df_display['close'], increasing_line_color='red', decreasing_line_color='green', name=f'{frequency}線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_display['date'], y=df_display['stop_loss'], mode='lines', line=dict(dash='dot'), name='止損價'), row=1, col=1)
    ma_colors = {5: 'blue', 10: 'orange', 20: 'purple', 60: 'black'}
    for ma in [5, 10, 20, 60]: fig.add_trace(go.Scatter(x=df_display['date'], y=df_display[f"MA{ma}"], mode='lines', line=dict(color=ma_colors[ma], width=1), name=f"MA{ma}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_display['date'], y=df_display['VWAP'], mode='lines', line=dict(color='orange', width=2, dash='solid'), name='主力成本線 (VWAP)'), row=1, col=1)

    # 成交量 & ATR
    vol_color = df_display.apply(lambda row: 'red' if row['close'] > row['open'] else ('green' if row['close'] < row['open'] else 'yellow'), axis=1)
    fig.add_trace(go.Bar(x=df_display['date'], y=df_display['volume'] / 1000, name='成交量', marker_color=vol_color), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_display['date'], y=df_display['VOL5'] / 1000, mode='lines', line=dict(color='blue', width=1), name='VOL5'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_display['date'], y=df_display['VOL20'] / 1000, mode='lines', line=dict(color='orange', width=1), name='VOL20'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_display['date'], y=df_display['ATR14'], mode='lines', line=dict(color='red', width=1), name='ATR14'), row=3, col=1)
    
    # 🌟 纏論分型標記和折線 (實時延伸筆段)
    
    # 1. 過濾已確認分型點
    df_pivots_display_filtered = pivot_df_full[
        (pivot_df_full['date'] >= df_display['date'].min()) &  
        (pivot_df_full['date'] <= df_display['date'].max())
    ].dropna(subset=['Pivot_Price']).copy()

    # 2. 準備實時延伸點
    extend_points = pd.DataFrame(columns=['date', 'Pivot_Price'])
    
    if last_pivot_date and not df_display.empty:
        # 找到最後一個分型點在 df_display 內的位置
        # 使用索引來處理日期可能不存在於 df_display 的情況 (如果 num_rows 太小)
        start_index = df_display[df_display['date'] == last_pivot_date].index
        
        if not start_index.empty:
            start_index = start_index[0]
            df_extension = df_display.loc[start_index:].copy()
            
            # 根據最後一個分型點的類型決定延伸線是連 High 還是 Low
            current_trend_status = trend_analysis['Overall_Trend']
            
            if last_pivot_type == 1: # 最後是頂分型 (向下走勢)
                # 延伸線連接 Low
                df_extension['Pivot_Price_Extension'] = df_extension['low']
                
                # 檢查是否下穿前底
                if len(df_pivots_display_filtered) >= 2:
                    B_pre = df_pivots_display_filtered.iloc[-2]['Pivot_Price']  
                    if df_extension['low'].min() < B_pre:
                        current_trend_status = "⚠️ **潛在趨勢反轉/持續下降 (下穿前底)**"
                
            elif last_pivot_type == -1: # 最後是底分型 (向上走勢)
                # 延伸線連接 High
                df_extension['Pivot_Price_Extension'] = df_extension['high']
                
                # 檢查是否上穿前高
                if len(df_pivots_display_filtered) >= 2:
                    T_pre = df_pivots_display_filtered.iloc[-2]['Pivot_Price']
                    if df_extension['high'].max() > T_pre:
                        current_trend_status = "✅ **趨勢持續 (上穿前高)**"
            
            # 確保延伸線從最後一個分型點的價格開始
            if 'Pivot_Price_Extension' in df_extension.columns:
                df_extension.loc[start_index, 'Pivot_Price_Extension'] = df_pivots_display_filtered.iloc[-1]['Pivot_Price']
                
                extend_points = df_extension[['date', 'Pivot_Price_Extension']].rename(columns={'Pivot_Price_Extension': 'Pivot_Price'})

            # 將實時突破判斷結果合併到主趨勢分析中
            trend_analysis['Overall_Trend'] = current_trend_status


    # 3. 合併已確認點和延伸點進行繪圖
    if not df_pivots_display_filtered.empty:
        plot_points = df_pivots_display_filtered[['date', 'Pivot_Price']].copy()
        
        # 附加延伸點 (確保不重複)
        if not extend_points.empty:
            start_date_filter = plot_points['date'].max()
            new_extension = extend_points[extend_points['date'] >= start_date_filter] # 使用 >= 確保連線點重複
            plot_points = pd.concat([plot_points, new_extension], ignore_index=True).drop_duplicates(subset=['date'], keep='last')
            
        # 繪製分型趨勢連線 (黑色折線)
        fig.add_trace(go.Scatter(
            x=plot_points['date'],  
            y=plot_points['Pivot_Price'],  
            mode='lines',  
            line=dict(color='black', width=2, dash='solid'),  
            name='分型趨勢連線 (實時筆段)'
        ), row=1, col=1)

        # 繪製圓圈標記 (只繪製已確認的分型點, 黑色, size=8)
        df_top = df_pivots_display_filtered[df_pivots_display_filtered['Pivot_Type']==1]
        fig.add_trace(go.Scatter(
            x=df_top['date'], y=df_top['Pivot_Price'], mode='markers',  
            marker=dict(size=8, color='black', symbol='circle', line=dict(width=1, color='black')),  
            name='頂分型', hoverinfo='text',
            text=[f"頂分型: {p:.2f}" for p in df_top['Pivot_Price']], uid='top_pivot_marker',
        ), row=1, col=1)
        
        df_bottom = df_pivots_display_filtered[df_pivots_display_filtered['Pivot_Type']==-1]
        fig.add_trace(go.Scatter(
            x=df_bottom['date'], y=df_bottom['Pivot_Price'], mode='markers',  
            marker=dict(size=8, color='black', symbol='circle', line=dict(width=1, color='black')),  
            name='底分型', hoverinfo='text',
            text=[f"底分型: {p:.2f}" for p in df_bottom['Pivot_Price']], uid='bottom_pivot_marker',
        ), row=1, col=1)
        
    stock_name = df_display['stock_name'].iloc[0] if 'stock_name' in df_display.columns and not df_display.empty else stock_id_clean
    first_date = df_display['date'].iloc[0].strftime("%Y-%m-%d")
    last_date = df_display['date'].iloc[-1].strftime("%Y-%m-%d")

    fig.update_layout(
        title=dict(
            # 🌟 將實時突破判斷結果顯示在標題
            text=f"{stock_id_clean} ({stock_name}) - {frequency}線趨勢: {trend_analysis['Overall_Trend']} ({first_date} ~ {last_date})",
            x=0.5, xanchor='center'
        ),
        xaxis_rangeslider_visible=False, hovermode='x unified', dragmode='drawline',
        newshape=dict(line_color='black', line_width=2),
        modebar_add=['drawline', 'drawopenpath', 'drawrect', 'drawcircle', 'eraseshape'],
        yaxis=dict(range=[yaxis_min, yaxis_max]),
        height=1200
    )

    html = fig.to_html(include_plotlyjs='cdn')
    
    # 🌟 修改回傳值，新增 trend_class
    return html, None, trend_analysis['Overall_Trend'], rebound_desc, trend_class

# ----------------- Flask 路由部分 -----------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    stock_id = request.form['stock_id'].strip()
    simple_mode = request.form.get('simple_mode') == '1'
    num_rows = request.form.get('num_rows', type=int, default=30)
    frequency = request.form.get('frequency', 'D')
    
    # 🌟 修改接收變數
    chart_html, error, trend_desc, rebound_desc, trend_class = generate_chart(stock_id, simple_mode=simple_mode, num_rows=num_rows, frequency=frequency)
    
    if error: return f"<h2>{error}</h2><a href='/'>返回</a>"
    # is_favorite 函數已被移動到前方，因此此處可以正確呼叫
    fav_status = is_favorite(stock_id) 
    
    return render_template(
        'chart.html',  
        chart_html=chart_html,  
        stock_id=stock_id,  
        stock_list=stock_id,  
        current_index=0,  
        simple_mode=simple_mode,  
        num_rows=num_rows,  
        is_favorite=fav_status,
        trend_desc=trend_desc,
        rebound_desc=rebound_desc,
        # 🌟 傳遞新的變數到前端
        trend_class=trend_class,
        frequency=frequency
    )

@app.route('/chart/<stock_id>/')
@app.route('/chart/<stock_id>')
def chart_from_list(stock_id):
    stock_id = stock_id.strip()
    simple_mode = request.args.get('simple_mode') == '1'
    num_rows = request.args.get('num_rows', type=int, default=30)
    stock_list = request.args.get('list', '')
    index = request.args.get('index', type=int, default=0)
    frequency = request.args.get('frequency', 'D')

    stock_ids = stock_list.split(',') if stock_list else [stock_id]
    index = max(0, min(index, len(stock_ids)-1))

    current_stock = stock_ids[index]
    # 🌟 修改接收變數
    chart_html, error, trend_desc, rebound_desc, trend_class = generate_chart(current_stock, simple_mode=simple_mode, num_rows=num_rows, frequency=frequency)
    
    if error: return f"<h2>{error}</h2><a href='/'>返回</a>"
    # is_favorite 函數已被移動到前方，因此此處可以正確呼叫
    fav_status = is_favorite(current_stock)

    return render_template(
        'chart.html',  
        chart_html=chart_html,  
        stock_id=current_stock,  
        stock_list=','.join(stock_ids),  
        current_index=index,  
        simple_mode=simple_mode,  
        num_rows=num_rows,  
        is_favorite=fav_status,
        trend_desc=trend_desc,
        rebound_desc=rebound_desc,
        # 🌟 傳遞新的變數到前端
        trend_class=trend_class,
        frequency=frequency
    )

# ----------------- Filter 及 Favorite 路由 -----------------
@app.route('/filter', methods=['POST'])
def filter_stocks():
    volume_min = request.form.get('volume_min', type=float, default=0)
    trend_type = request.form.get('trend_type', '')
    adr14_min = request.form.get('change_min', type=float, default=0)
    simple_mode = request.form.get('simple_mode') == '1'
    num_rows = request.form.get('num_rows', type=int, default=60)
    recent_days = request.form.get('recent_days', type=int, default=30)
    frequency = request.form.get('frequency', 'D')

    recent_date = (datetime.today() - timedelta(days=recent_days)).strftime("%Y-%m-%d")
    all_data = []
    limit = 1000
    offset = 0

    while True:
        try:
            res = requests.get(f"{SUPABASE_URL}/rest/v1/quick_view", headers=headers,
                params={
                    "latest_volume": f"gte.{int(volume_min)}",
                    "adr14": f"gte.{adr14_min}",
                    "latest_date": f"gte.{recent_date}",
                    "trend": f"eq.{trend_type}" if trend_type else None,
                    "order": "latest_date.desc", "limit": limit, "offset": offset, "select": "*"
                }, timeout=30
            )
            res.raise_for_status()
            data = res.json()
            if not data: break
            all_data.extend(data)
            if len(data) < limit: break
            offset += limit
        except Exception as e: return f"<h2>Supabase 讀取 QUICK_VIEW 失敗: {e}</h2><a href='/'>返回</a>"

    if not all_data: return "<h2>沒有符合條件的股票</h2><a href='/'>返回</a>"
    df = pd.DataFrame(all_data)
    stock_ids = [str(sid) for sid in df['stock_id']]
    count = len(df); list_param = urllib.parse.quote(','.join(stock_ids))
    
    html = (f"<h2>篩選結果（共 {count} 筆）</h2>" "<table border='1' cellpadding='6' style='margin-left:0; text-align:left;'>" "<thead><tr>" "<th>股票代號</th><th>股票名稱</th><th>成交量</th>" "<th>ADR14(%)</th><th>14天平均成交量</th><th>趨勢</th>" "</tr></thead><tbody>")
    for idx, row in df.iterrows():
        simple_param = "1" if simple_mode else "0"
        html += (f"<tr>"  
                    f"<td><a href='/chart/{row['stock_id']}?simple_mode={simple_param}&num_rows={num_rows}&list={list_param}&index={idx}&frequency={frequency}'>{row['stock_id']}</a></td>"  
                    f"<td>{row['stock_name']}</td>"  
                    f"<td>{int(row['latest_volume'])}</td>"  
                    f"<td>{row['adr14']:.2f}</td>"  
                    f"<td>{int(row['avg_volume_14'])}</td>"  
                    f"<td>{row['trend']}</td>"  
                    f"</tr>")
    html += "</tbody></table><br><a href='/'>返回</a>"
    return html

@app.route('/favorites', methods=['POST'])
def favorites_page():
    simple_mode = request.form.get('simple_mode') == '1'
    num_rows = request.form.get('num_rows', type=int, default=30)
    frequency = request.form.get('frequency', 'D')
    
    try:
        res = requests.get(f"{SUPABASE_URL}/rest/v1/{FAVORITE_TABLE}", headers=headers); res.raise_for_status(); fav_data = res.json()
    except Exception as e: return f"<h2>讀取最愛股票失敗: {e}</h2><a href='/'>返回</a>"
    if not fav_data: return "<h2>尚無最愛股票</h2><a href='/'>返回</a>"
    stock_ids = [item['stock_id'] for item in fav_data]
    try:
        res_qv = requests.get(f"{SUPABASE_URL}/rest/v1/quick_view", headers=headers, params={"stock_id": f"in.({','.join(stock_ids)})", "order": "latest_date.desc", "select": "*"})
        res_qv.raise_for_status(); qv_data = res_qv.json()
    except Exception as e: return f"<h2>讀取最愛股票快照資料失敗: {e}</h2><a href='/'>返回</a>"

    df_qv = pd.DataFrame(qv_data); count = len(df_qv); list_param = urllib.parse.quote(','.join(stock_ids))
    
    html = (f"<h2>我的最愛（共 {count} 筆）</h2>" "<form method='post' action='/favorites_clear' " "onsubmit=\"return confirm('確定要刪除所有最愛嗎？');\">" "<button type='submit' style='margin-bottom:10px;'>刪除全部最愛</button>" "</form>" "<table border='1' cellpadding='6' style='margin-left:0; text-align:left;'>" "<thead><tr>" "<th>股票代號</th><th>股票名稱</th><th>成交量</th>" "<th>ADR14(%)</th><th>14天平均成交量</th><th>趨勢</th>" "</tr></thead><tbody>")
    for idx, row in df_qv.iterrows():
        simple_param = "1" if simple_mode else "0"
        html += (f"<tr>"  
                    f"<td><a href='/chart/{row['stock_id']}?simple_mode={simple_param}&num_rows={num_rows}&list={list_param}&index={idx}&frequency={frequency}'>{row['stock_id']}</a></td>"  
                    f"<td>{row['stock_name']}</td>"  
                    f"<td>{int(row['latest_volume'])}</td>"  
                    f"<td>{row['adr14']:.2f}</td>"  
                    f"<td>{int(row['avg_volume_14'])}</td>"  
                    f"<td>{row['trend']}</td>"  
                    f"</tr>")
    html += "</tbody></table><br><a href='/'>返回</a>"
    return html

@app.route('/favorite', methods=['POST'])
def favorite_toggle():
    stock_id = request.form.get('stock_id', '').strip(); stock_name = request.form.get('stock_name', '').strip()
    if not stock_id: return jsonify({"message": "股票代號不可為空"}), 400
    try:
        res_check = requests.get(f"{SUPABASE_URL}/rest/v1/{FAVORITE_TABLE}", headers=headers, params={"stock_id": f"eq.{stock_id}"}); res_check.raise_for_status(); exists = len(res_check.json()) > 0
    except Exception as e: return jsonify({"message": f"檢查最愛失敗: {e}"}), 500

    try:
        if exists:
            res = requests.delete(f"{SUPABASE_URL}/rest/v1/{FAVORITE_TABLE}", headers=headers, params={"stock_id": f"eq.{stock_id}"}); res.raise_for_status()
            return jsonify({"message": f"{stock_name} 已從最愛移除", "favorite": False})
        else:
            payload = {"stock_id": stock_id, "stock_name": stock_name}
            res = requests.post(f"{SUPABASE_URL}/rest/v1/{FAVORITE_TABLE}", headers={**headers, "Content-Type": "application/json"}, json=payload); res.raise_for_status()
            return jsonify({"message": f"{stock_name} 已加入最愛", "favorite": True})
    except Exception as e: return jsonify({"message": f"操作最愛失敗: {e}"}), 500

@app.route('/favorites_clear', methods=['POST'])
def favorites_clear():
    try:
        res = requests.delete(f"{SUPABASE_URL}/rest/v1/{FAVORITE_TABLE}", headers=headers, params={"stock_id": "not.is.null"})  
        res.raise_for_status(); return "<script>alert('已刪除所有最愛股票'); window.location.href='/'</script>"
    except Exception as e: return f"<h2>刪除最愛失敗: {e}</h2><a href='/'>返回首頁</a>"

# ----------------- 運行程式 -----------------
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)