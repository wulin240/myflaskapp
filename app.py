from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import urllib.parse
from datetime import datetime, timedelta
import json # 確保可以處理 JSON 響應

app = Flask(__name__)

# ----------------- Supabase 設定 -----------------
# 請將這裡替換成您的 Supabase 專案資訊
SUPABASE_URL = "https://djhdpltrhlhqfxmwniki.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRqaGRwbHRyaGxocWZ4bXduaWtpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjMwNjQwODgsImV4cCI6MjA3ODY0MDA4OH0.jwSPe-HMHxv2xGCjS42O5Cjby0KtgsHEStlQWs0cyPk"
TABLE_NAME = "stock_data"
FAVORITE_TABLE = "favorites"

headers = {
    "apikey": SUPABASE_KEY.strip(),
    "Authorization": f"Bearer {SUPABASE_KEY.strip()}",
    "Content-Type": "application/json" # 新增 Content-Type 確保 POST/DELETE 正確
}

# ----------------- 輔助函數：最愛股票檢查 -----------------
def is_favorite(stock_id):
    """檢查股票是否已加入最愛"""
    try:
        # 使用 count 查詢來優化性能
        params = {"stock_id": f"eq.{stock_id}", "select": "count"}
        res = requests.get(f"{SUPABASE_URL}/rest/v1/{FAVORITE_TABLE}", headers=headers, params=params, timeout=10)
        res.raise_for_status()
        # Supabase count 回應會在 Headers 中的 Content-Range
        return int(res.headers.get("Content-Range").split('/')[-1]) > 0
    except Exception as e:
        print(f"⚠️ 檢查最愛失敗: {e}")
        return False
        
# ----------------- 抓取股票資料 -----------------
def fetch_stock_data(stock_id):
    """從 Supabase 獲取股票 OHLCV 數據"""
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
    if df_daily.empty: return df_daily
    df = df_daily.set_index('date')
    weekly_data = df.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    df_weekly = weekly_data.dropna(subset=['open']).reset_index()
    if not df_daily.empty and 'stock_name' in df_daily.columns:
        df_weekly['stock_name'] = df_daily['stock_name'].iloc[-1]
    if not df_daily.empty and 'stock_id' in df_daily.columns:
        df_weekly['stock_id'] = df_daily['stock_id'].iloc[-1]
    return df_weekly


def kline_merge(df):
    """K線合併：採用更穩健的包含關係判定邏輯，處理邊界情況。（纏論筆段預處理）"""
    if df.empty: return df
    df_raw = df.copy().set_index('date')
    processed_kline = []
    
    current_kline = {col: df_raw.iloc[0][col] for col in ['open', 'high', 'low', 'close', 'volume']}
    current_kline['Index'] = df_raw.index[0]

    for i in range(1, len(df_raw)):
        next_row = df_raw.iloc[i]
        next_kline = {col: next_row[col] for col in ['open', 'high', 'low', 'close', 'volume']}
        
        # 包含關係判定：後一根 K 線完全被前一根 K 線包含，或後一根 K 線完全包含前一根 K 線
        is_inclusion = (
            (next_row['high'] <= current_kline['high'] and next_row['low'] >= current_kline['low']) or
            (next_row['high'] >= current_kline['high'] and next_row['low'] <= current_kline['low'])
        )
        
        if is_inclusion:
            current_kline['high'] = max(current_kline['high'], next_row['high'])
            current_kline['low'] = min(current_kline['low'], next_row['low'])
            current_kline['volume'] += next_row['volume']
            # 保留方向：如果是同方向（例如都是陽線）則取最新收盤價，但這裡簡化為只取最新收盤價
            current_kline['close'] = next_row['close']
            current_kline['Index'] = df_raw.index[i]
        else:
            processed_kline.append(current_kline)
            # 建立新的 K 線段
            current_kline = next_kline
            current_kline['Index'] = df_raw.index[i]
            
    processed_kline.append(current_kline)
    
    df_merged = pd.DataFrame(processed_kline).set_index('Index').rename_axis('date').reset_index()
    df_merged['date'] = pd.to_datetime(df_merged['date'])
    return df_merged


def find_divergence(df_merged):
    """基礎分型判斷：中間 K 線高低點大於或等於兩側。"""
    df = df_merged.copy()
    df['H_prev'], df['H_next'] = df['high'].shift(1), df['high'].shift(-1)
    df['L_prev'], df['L_next'] = df['low'].shift(1), df['low'].shift(-1)
    # 頂分型：中間 high >= 左右 high
    df['Is_Top_Divergence'] = (df['high'] >= df['H_prev']) & (df['high'] >= df['H_next'])
    # 底分型：中間 low <= 左右 low
    df['Is_Bottom_Divergence'] = (df['low'] <= df['L_prev']) & (df['low'] <= df['L_next'])
    df['Is_Top_Divergence'] = df['Is_Top_Divergence'].fillna(False)
    df['Is_Bottom_Divergence'] = df['Is_Bottom_Divergence'].fillna(False)
    df['Top_Price'] = np.where(df['Is_Top_Divergence'], df['high'], np.nan)
    df['Bottom_Price'] = np.where(df['Is_Bottom_Divergence'], df['low'], np.nan)
    return df


def find_stroke_pivots(df_merged):
    """
    🌟 嚴格筆段判斷函數。篩選出符合「兩分型之間至少有一根非包含 K 線」的轉折點。
    """
    df_divergence = find_divergence(df_merged.copy())
    pivot_points = df_divergence[df_divergence['Is_Top_Divergence'] | df_divergence['Is_Bottom_Divergence']].copy()

    if pivot_points.empty: return pd.DataFrame()

    # 1: 頂分型, -1: 底分型
    pivot_points['Type'] = np.where(pivot_points['Is_Top_Divergence'], 1, -1)
    
    final_pivots_list = []
    last_pivot_index = -1 # 用來記錄在 df_merged 中的索引位置

    for idx, row in pivot_points.iterrows():
        # 獲取當前分型在 df_merged 中的實際位置
        current_index_loc = df_merged[df_merged['date'] == row['date']].index[0]
        
        if not final_pivots_list:
            # 第一個分型直接加入
            row['Pivot_Price_Calc'] = row['Top_Price'] if row['Type'] == 1 else row['Bottom_Price']
            final_pivots_list.append(row)
            last_pivot_index = current_index_loc
            continue
            
        last_pivot = final_pivots_list[-1]
        last_pivot_index_loc = df_merged[df_merged['date'] == last_pivot['date']].index[0]
        
        if row['Type'] == last_pivot['Type']:
            # 同向分型，根據價格取極值，並替換掉前一個分型
            is_new_extreme = (row['Type'] == 1 and row['Top_Price'] > last_pivot['Top_Price']) or \
                             (row['Type'] == -1 and row['Bottom_Price'] < last_pivot['Bottom_Price'])
            
            if is_new_extreme:
                # 更新前一個分型（替換）
                # 注意：這裡應該更新 final_pivots_list 裡最後一項的數據，而不是 last_pivot
                final_pivots_list[-1].update({'date': row['date'],
                                              'Top_Price': row['Top_Price'] if row['Type'] == 1 else last_pivot['Top_Price'],
                                              'Bottom_Price': row['Bottom_Price'] if row['Type'] == -1 else last_pivot['Bottom_Price'],
                                              'Pivot_Price_Calc': row['Top_Price'] if row['Type'] == 1 else row['Bottom_Price'],
                                              'Is_Top_Divergence': row['Is_Top_Divergence'],
                                              'Is_Bottom_Divergence': row['Is_Bottom_Divergence']})
                last_pivot_index = current_index_loc
        else:
            # 異向分型：檢查是否滿足嚴格筆段定義 (至少間隔一根 K 線，即 index 距離 >= 2)
            kline_count_between = current_index_loc - last_pivot_index_loc
            if kline_count_between >= 2:
                row['Pivot_Price_Calc'] = row['Top_Price'] if row['Type'] == 1 else row['Bottom_Price']
                final_pivots_list.append(row)
                last_pivot_index = current_index_loc
                
    # 由於 last_pivot 在循環中更新的是字典引用，我們需要重新構造 DataFrame 以確保數據正確
    df_filtered = pd.DataFrame(final_pivots_list)

    if df_filtered.empty: return pd.DataFrame()
        
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    df_pivot_data = df_filtered[['date', 'Type', 'Pivot_Price_Calc']].rename(columns={
        'Type': 'Pivot_Type',
        'Pivot_Price_Calc': 'Pivot_Price'
    })
    return df_pivot_data


def filter_pivots_for_stroke(df_result, df_original):
    """將分型結果合併回原始K線數據，並找出最後一個轉折點的資訊。"""
    df_original['date'] = pd.to_datetime(df_original['date'])
    
    # 處理無轉折點的情況
    if df_result.empty:
        df_original['Pivot_Type'] = 0
        df_original['Pivot_Price'] = np.nan
        return df_original, None, 0

    last_pivot_row = df_result.iloc[-1]
    last_date = last_pivot_row['date']
    last_type = last_pivot_row['Pivot_Type']
    
    df_merged = df_original.merge(df_result, on='date', how='left')
    df_merged['Pivot_Type'] = df_merged['Pivot_Type'].fillna(0).astype(int)
    df_merged['Pivot_Price'] = df_merged['Pivot_Price'].fillna(np.nan)
    
    return df_merged, last_date, last_type


def analyze_trend_by_pivots(pivot_df):
    """基於有效轉折點判斷頂底趨勢 (HH/HL)"""
    if pivot_df.empty or len(pivot_df) < 4:
        return {'Overall_Trend': "結構數據不足 (需至少四個有效轉折點)"}

    # 確保只使用最新的、有效的頂點和底點
    tops = pivot_df[pivot_df['Pivot_Type'] == 1]['Pivot_Price'].dropna()
    bottoms = pivot_df[pivot_df['Pivot_Type'] == -1]['Pivot_Price'].dropna()

    if len(tops) < 2 or len(bottoms) < 2:
        return {'Overall_Trend': "結構數據不足 (需至少兩個頂點和兩個底點)"}

    # 取最近的兩個頂點 (T2, T1) 和兩個底點 (B2, B1)
    # T2/B2 是最新的
    T2, T1 = tops.iloc[-1], tops.iloc[-2]
    B2, B1 = bottoms.iloc[-1], bottoms.iloc[-2]

    is_hh, is_hl = T2 > T1, B2 > B1 # Higher High, Higher Low
    is_lh, is_ll = T2 < T1, B2 < B1 # Lower High, Lower Low

    trend_result = "盤整/待確認"
    if is_hh and is_hl: trend_result = "✅ 上升趨勢 (Higher Highs & Higher Lows)"
    elif is_lh and is_ll: trend_result = "🔻 下降趨勢 (Lower Highs & Lower Lows)"
    elif is_hh and is_ll: trend_result = "⚠️ 擴張結構 (高點抬高, 低點降低)"
    elif is_lh and is_hl: trend_result = "⏳ 收斂結構 (高點降低, 低點抬高)"
        
    return {'Overall_Trend': trend_result}

def check_rebound_signal(df_full_processed, trend_period=90):
    """結構回調起漲信號檢查 (主要用於判斷多頭回調是否出現買點)"""
    if len(df_full_processed) < trend_period + 5:
        return False, "數據不足以判斷長線趨勢"

    df_check = df_full_processed.iloc[-trend_period:].copy()
    pivot_df = df_check[df_check['Pivot_Type'] != 0].copy()
    current = df_check.iloc[-1]
    prev = df_check.iloc[-2]

    trend_result = analyze_trend_by_pivots(pivot_df)['Overall_Trend']
    is_high_level_trend = ('上升趨勢' in trend_result)
    # 額外 MA 過濾條件：60日均線向上且收盤價在 60 日均線之上
    is_ma_aligned = (df_check['MA60'].iloc[-1] > df_check['MA60'].iloc[0]) and (current['close'] > current['MA60'])
    
    if not (is_high_level_trend and is_ma_aligned):
        return False, f"❌ 長線趨勢不符合 HH/HL 上升結構或 MA60 條件 ({trend_result})"

    bottoms = pivot_df[pivot_df['Pivot_Type'] == -1]['Pivot_Price'].dropna()
    tops = pivot_df[pivot_df['Pivot_Type'] == 1]['Pivot_Price'].dropna()
    
    if len(bottoms) < 2 or len(tops) < 1:
        return False, "結構轉折點不足，無法定位回調區間"

    T_last = tops.iloc[-1]      # 最新高點
    B_pre_T = bottoms.iloc[-2]  # 前一個低點 (前一個筆段的底部支撐)
    
    # 正在回調中 (價格從高點下來)
    is_correcting = (current['close'] < T_last)
    # 守住前一個低點 (B_pre_T) 的支撐
    is_holding_support = (current['low'] > B_pre_T)
    
    if not (is_correcting and is_holding_support):
        if current['close'] > T_last:
            return False, "✅ 已經突破前高，回調已結束，屬於新的上漲波段"
        return False, f"🚨 結構性回調失敗：低點已跌破結構支撐 B_pre_T ({B_pre_T:.2f})"

    # 檢查 K 線確認訊號：看漲吞噬 (Bullish Engulfing)
    is_bullish_engulfing = (
        (current['close'] > current['open']) and  # 當天是陽線
        (current['close'] > prev['open']) and  # 當天收盤價高於前一天開盤價
        (current['open'] < prev['close'])     # 當天開盤價低於前一天收盤價
    )
    # 檢查 K 線確認訊號：收盤站上 MA20 且突破前一根 K 線高點或形成看漲吞噬
    is_rebound_confirmed = (
        current['close'] > current['MA20']
        and (current['close'] > prev['high'] or is_bullish_engulfing)
    )

    if is_rebound_confirmed:
        return True, f"✅ **【結構回調起漲信號】**：價格在 B_pre_T 支撐上確認反轉！(支撐位: {B_pre_T:.2f})"
    else:
        return False, f"💡 **潛在起漲提示**：結構已確認為健康回調區間 ({B_pre_T:.2f} 支撐), 等待強勢 K 線確認起漲！"


import pandas as pd
import numpy as np

# ----------------- 🌟 NEW: 主力行為偵測核心函數 -----------------
def detect_smart_money_signals(df_tech, vsa_vol_multiplier=2, rsi_period=14):
    """
    主力行為偵測 - 判斷潛在的主力拉抬 (買入) 和拋售 (賣出) 訊號。
    """
    df = df_tech.copy()
    df.reset_index(drop=True, inplace=True) # 確保索引連續

    # --- 計算基礎指標 (RSI, VWAP, VOL20) ---
    df['TP'] = (df['high'] + df['low'] + df['close']) / 3
    df['VOL20'] = df['volume'].rolling(20).mean()
    df['TPV'] = df['TP'] * df['volume']
    
    # VWAP 累積計算
    df['VWAP'] = df['TPV'].cumsum() / df['volume'].cumsum()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=rsi_period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=rsi_period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # K 線形態與量能
    df['Body_Ratio'] = (df['close'] - df['open']).abs() / (df['high'] - df['low']).replace(0, 1e-6)
    is_high_volume = df['volume'] >= (df['VOL20'] * vsa_vol_multiplier)
    
    # ----------------------------------------------------
    # --- 多頭訊號 (Buy Signals) ---
    # ----------------------------------------------------
    
    # --- 1. VSA 強勢拉抬 (吸籌) ---
    is_long_bull_k = (df['close'] > df['open']) & (df['Body_Ratio'] > 0.6) # 陽線且實體飽滿
    df['Signal_VSA_Strong'] = np.where(is_long_bull_k & is_high_volume, df['low'] * 0.99, np.nan)
    
    # --- 2. 主力成本突破訊號：收盤站上 VWAP ---
    df['Signal_VWAP_Break'] = np.where(
        (df['close'] > df['VWAP']) & (df['close'].shift(1).fillna(-np.inf) <= df['VWAP'].shift(1).fillna(-np.inf)),
        df['low'] * 0.995,
        np.nan
    )
    
    # ----------------------------------------------------
    # --- 新增：空頭訊號 (Sell Signals) ---
    # ----------------------------------------------------
    
    # --- 4. VSA 恐慌拋售 (派發/出貨) ---
    is_long_bear_k = (df['close'] < df['open']) & (df['Body_Ratio'] > 0.6) # 陰線且實體飽滿
    # 標記在 K 線頂部
    df['Signal_VSA_Weak'] = np.where(is_long_bear_k & is_high_volume, df['high'] * 1.01, np.nan)

    # --- 5. 主力成本跌破訊號：收盤跌破 VWAP ---
    # 判斷今日收盤價跌破 VWAP，且昨日收盤價在 VWAP 之上 (跌破)
    df['Signal_VWAP_BreakDown'] = np.where(
        (df['close'] < df['VWAP']) & (df['close'].shift(1).fillna(np.inf) >= df['VWAP'].shift(1).fillna(np.inf)),
        df['high'] * 1.005, # 標記在 K 線頂部附近
        np.nan
    )
    
    # ----------------------------------------------------
    # --- 3. 背離訊號 (Divergence & TopDivergence) ---
    # ----------------------------------------------------
    divergence_signal = [np.nan] * len(df)
    top_divergence_signal = [np.nan] * len(df)
    
    # 找出底分型和頂分型
    df['Temp_Bottom_Pivot'] = (df['low'].shift(-1) > df['low']) & (df['low'].shift(1) > df['low'])
    df['Temp_Top_Pivot'] = (df['high'].shift(-1) < df['high']) & (df['high'].shift(1) < df['high'])
    
    bottom_pivots = df[df['Temp_Bottom_Pivot']].copy()
    top_pivots = df[df['Temp_Top_Pivot']].copy()

    # --- 底部背離 (Signal_Divergence) ---
    if len(bottom_pivots) >= 2:
        for i in range(1, len(bottom_pivots)):
            B2_idx = bottom_pivots.index[i]
            B1_idx = bottom_pivots.index[i-1]
            
            is_price_ll = df.loc[B2_idx, 'low'] < df.loc[B1_idx, 'low']
            is_rsi_hh = df.loc[B2_idx, 'RSI'] > df.loc[B1_idx, 'RSI']

            if is_price_ll and is_rsi_hh:
                divergence_signal[B2_idx] = df.loc[B2_idx, 'low'] * 0.998

    # --- 新增：頂部背離 (Signal_TopDivergence) ---
    if len(top_pivots) >= 2:
        for i in range(1, len(top_pivots)):
            T2_idx = top_pivots.index[i]
            T1_idx = top_pivots.index[i-1]
            
            # 確認價格頂頂高 (Price High Higher)
            is_price_hh = df.loc[T2_idx, 'high'] > df.loc[T1_idx, 'high']
            # 確認 RSI 頂底低 (RSI Low Lower)
            is_rsi_ll = df.loc[T2_idx, 'RSI'] < df.loc[T1_idx, 'RSI']

            if is_price_hh and is_rsi_ll:
                top_divergence_signal[T2_idx] = df.loc[T2_idx, 'high'] * 1.002
        
    df['Signal_Divergence'] = pd.Series(divergence_signal, index=df.index)
    df['Signal_TopDivergence'] = pd.Series(top_divergence_signal, index=df.index)
    
    # ----------------------------------------------------
    # --- 訊號優先級清理 (避免多空訊號衝突) ---
    # ----------------------------------------------------
    
    is_any_strong_buy = df['Signal_VSA_Strong'].notna() | df['Signal_VWAP_Break'].notna()
    is_any_strong_sell = df['Signal_VSA_Weak'].notna() | df['Signal_VWAP_BreakDown'].notna()

    # 1. 買入訊號優先：強勢買入日清除所有看跌/賣出訊號
    df.loc[is_any_strong_buy, 'Signal_VSA_Weak'] = np.nan
    df.loc[is_any_strong_buy, 'Signal_VWAP_BreakDown'] = np.nan
    df.loc[is_any_strong_buy, 'Signal_TopDivergence'] = np.nan
    
    # 2. 賣出訊號優先：強勢賣出日清除所有看漲/買入訊號
    df.loc[is_any_strong_sell, 'Signal_VSA_Strong'] = np.nan
    df.loc[is_any_strong_sell, 'Signal_VWAP_Break'] = np.nan
    df.loc[is_any_strong_sell, 'Signal_Divergence'] = np.nan
    
    # 最終回傳所有訊號欄位
    return df[['date', 
               'Signal_VSA_Strong', 'Signal_VWAP_Break', 'Signal_Divergence', 
               'Signal_VSA_Weak', 'Signal_VWAP_BreakDown', 'Signal_TopDivergence']]
# ----------------- 整合生成圖表 (含趨勢分析和訊號檢查) -----------------
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 假設以下函數已定義並可使用：
# fetch_stock_data, convert_to_weekly, kline_merge, find_stroke_pivots, 
# filter_pivots_for_stroke, detect_smart_money_signals, analyze_trend_by_pivots, 
# check_rebound_signal 

# ----------------- 整合生成圖表 (含趨勢分析和訊號檢查) -----------------
def generate_chart(stock_id_clean, start_date=None, end_date=None, simple_mode=False, num_rows=30, frequency='D'):
    """生成包含 K 線圖、纏論筆段、技術指標和主力訊號的 Plotly 圖表。"""
    
    df_original = fetch_stock_data(stock_id_clean)
    if df_original.empty: return None, f"{stock_id_clean} 無資料", "N/A", "N/A", "neutral"

    df_full = df_original.copy()
    
    if frequency == 'W': df_full = convert_to_weekly(df_full)
    
    if start_date and end_date:
        df_full = df_full[
            (df_full['date'] >= pd.to_datetime(start_date)) &
            (df_full['date'] <= pd.to_datetime(end_date))
        ]

    if df_full.empty: return None, f"{stock_id_clean} 在 {start_date} ~ {end_date} 無資料", "N/A", "N/A", "neutral"

    # --- 1. 技術指標計算 ---
    df_tech = df_full.copy()
    df_tech['TP'] = (df_tech['high'] + df_tech['low'] + df_tech['close']) / 3
    # 增加移動平均線
    for ma in [5, 10, 20, 60]: df_tech[f"MA{ma}"] = df_tech['close'].rolling(ma).mean()
    # 增加成交量均線
    df_tech['VOL5'] = df_tech['volume'].rolling(5).mean()
    df_tech['VOL20'] = df_tech['volume'].rolling(20).mean()
    # ATR (平均真實波幅) 計算
    df_tech['H-L'], df_tech['H-PC'], df_tech['L-PC'] = df_tech['high'] - df_tech['low'], abs(df_tech['high'] - df_tech['close'].shift(1)), abs(df_tech['low'] - df_tech['close'].shift(1))
    df_tech['TR'] = df_tech[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df_tech['ATR14'] = df_tech['TR'].rolling(14).mean().round(3)
    df_tech['stop_loss'] = df_tech['low'] - df_tech['ATR14'].fillna(0) # 簡化止損價
    
    # --- 2. 纏論筆段轉折點處理 ---
    df_merged = kline_merge(df_tech.copy())
    df_pivot_data = find_stroke_pivots(df_merged.copy())
    df_final, last_pivot_date, last_pivot_type = filter_pivots_for_stroke(df_pivot_data, df_tech.copy())

    # --- 🌟 主力信號偵測 ---
    df_smart_signals = detect_smart_money_signals(df_final.copy()) 
    df_final = df_final.merge(df_smart_signals, on='date', how='left')
    
    # --- 3. 趨勢分析與信號檢查 ---
    df_display = df_final.tail(num_rows).copy()
    pivot_df_full = df_final[df_final['Pivot_Type'] != 0].copy()
    
    trend_analysis = analyze_trend_by_pivots(pivot_df_full)
    is_rebound, rebound_desc = check_rebound_signal(df_final)

    trend_desc_final = trend_analysis['Overall_Trend']
    
    # 趨勢分類 (用於前端顏色顯示)
    trend_class = 'neutral'
    if '下降趨勢' in trend_desc_final or '下穿前底' in trend_desc_final:
        trend_class = 'bearish'
    elif '上升趨勢' in trend_desc_final or '上穿前高' in trend_desc_final:
        trend_class = 'bullish'
        
    # 🌟 VWAP：重新計算，僅限於 df_display 範圍 (確保顯示的 VWAP 是相對的)
    df_display['TPV_display'] = df_display['TP'] * df_display['volume']
    df_display['VWAP'] = df_display['TPV_display'].cumsum() / df_display['volume'].cumsum()
    
    # --- 4. 繪製圖表 ---
    min_price = df_display[['low', 'MA5', 'MA10', 'MA20', 'MA60', 'VWAP']].min(skipna=True).min(skipna=True)
    max_price = df_display[['high', 'MA5', 'MA10', 'MA20', 'MA60', 'VWAP']].max(skipna=True).max(skipna=True)
    
    if pd.isna(min_price) or pd.isna(max_price) or min_price == max_price:
        min_price = df_display['close'].min()
        max_price = df_display['close'].max()

    price_range = max_price - min_price
    yaxis_min = min_price - price_range * 0.2 
    yaxis_max = max_price + price_range * 0.2 

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        row_heights=[0.7, 0.15, 0.15],
        subplot_titles=(f"K線圖 ({frequency}線, 含纏論筆段)", "成交量", "ATR")
    )

    # K線圖與指標
    fig.add_trace(go.Candlestick(x=df_display['date'], open=df_display['open'], high=df_display['high'], low=df_display['low'], close=df_display['close'], increasing_line_color='red', decreasing_line_color='green', name=f'{frequency}線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_display['date'], y=df_display['stop_loss'], mode='lines', line=dict(dash='dot', color='gray'), name='止損價'), row=1, col=1)
    ma_colors = {5: 'blue', 10: 'orange', 20: 'purple', 60: 'black'}
    for ma in [5, 10, 20, 60]: fig.add_trace(go.Scatter(x=df_display['date'], y=df_display[f"MA{ma}"], mode='lines', line=dict(color=ma_colors[ma], width=1), name=f"MA{ma}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_display['date'], y=df_display['VWAP'], mode='lines', line=dict(color='magenta', width=2, dash='solid'), name='主力成本線 (VWAP)'), row=1, col=1)

    # ----------------------------------------------------
    # 🌟 買入訊號 (標記在 K 線下方)
    # ----------------------------------------------------
    
    # VSA 強勢拉抬
    fig.add_trace(go.Scatter(
        x=df_display['date'],
        y=df_display['Signal_VSA_Strong'],
        mode='markers',
        marker=dict(size=12, symbol='star-triangle-up', color='red', line=dict(width=1, color='black')),
        name='VSA 強勢拉抬',
        hovertext="主力VSA強勢拉抬",
        hoverinfo='text'
    ), row=1, col=1)

    # VWAP 成本突破
    fig.add_trace(go.Scatter(
        x=df_display['date'],
        y=df_display['Signal_VWAP_Break'],
        mode='markers',
        marker=dict(size=10, symbol='triangle-up', color='orange', line=dict(width=1, color='black')),
        name='VWAP 成本突破',
        hovertext="主力成本突破",
        hoverinfo='text'
    ), row=1, col=1)

    # 🌟 NEW: 加入 RSI 底背離吸籌信號 (調整 Y 座標)
    offset_divergence = df_display['ATR14'] * 0.2
    y_divergence_adjusted = df_display['Signal_Divergence'] - offset_divergence
    
    fig.add_trace(go.Scatter(
        x=df_display['date'],
        y=y_divergence_adjusted,
        mode='markers',
        marker=dict(size=10, symbol='diamond', color='blue', line=dict(width=1, color='black')),
        name='RSI 底背離 (吸籌)',
        hovertext="RSI底背離吸籌",
        hoverinfo='text'
    ), row=1, col=1)

    # ----------------------------------------------------
    # 🌟 賣出訊號 (標記在 K 線上方)
    # ----------------------------------------------------
    
    # VSA 恐慌拋售
    fig.add_trace(go.Scatter(
        x=df_display['date'],
        y=df_display['Signal_VSA_Weak'],
        mode='markers',
        marker=dict(size=12, symbol='star-triangle-down', color='green', line=dict(width=1, color='black')),
        name='VSA 恐慌拋售',
        hovertext="主力VSA恐慌拋售",
        hoverinfo='text'
    ), row=1, col=1)
    
    # VWAP 成本跌破
    fig.add_trace(go.Scatter(
        x=df_display['date'],
        y=df_display['Signal_VWAP_BreakDown'],
        mode='markers',
        marker=dict(size=10, symbol='triangle-down', color='purple', line=dict(width=1, color='black')),
        name='VWAP 成本跌破',
        hovertext="主力成本跌破",
        hoverinfo='text'
    ), row=1, col=1)
    
    # 🌟 NEW: 加入 RSI 頂背離派發信號 (調整 Y 座標)
    offset_top_divergence = df_display['ATR14'] * 0.2
    y_top_divergence_adjusted = df_display['Signal_TopDivergence'] + offset_top_divergence
    
    fig.add_trace(go.Scatter(
        x=df_display['date'],
        y=y_top_divergence_adjusted, # 使用調整後的座標
        mode='markers',
        marker=dict(size=10, symbol='diamond', color='green', line=dict(width=1, color='black')),
        name='RSI 頂背離 (派發)',
        hovertext="RSI頂背離派發",
        hoverinfo='text'
    ), row=1, col=1)
    
    # ----------------------------------------------------
    # 成交量 & ATR
    # ----------------------------------------------------
    vol_color = df_display.apply(lambda row: 'red' if row['close'] > row['open'] else ('green' if row['close'] < row['open'] else 'gray'), axis=1)
    fig.add_trace(go.Bar(x=df_display['date'], y=df_display['volume'] / 1000, name='成交量 (K)', marker_color=vol_color), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_display['date'], y=df_display['VOL5'] / 1000, mode='lines', line=dict(color='blue', width=1), name='VOL5 (K)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_display['date'], y=df_display['VOL20'] / 1000, mode='lines', line=dict(color='orange', width=1), name='VOL20 (K)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_display['date'], y=df_display['ATR14'], mode='lines', line=dict(color='red', width=1), name='ATR14'), row=3, col=1)
    
    # 🌟 嚴格筆段標記和折線 (實時延伸筆段) (保持不變)
    df_pivots_display_filtered = pivot_df_full[
        (pivot_df_full['date'] >= df_display['date'].min()) &
        (pivot_df_full['date'] <= df_display['date'].max())
    ].dropna(subset=['Pivot_Price']).copy()

    # 處理筆段連線的邏輯 (略，與原碼相同)
    extend_points = pd.DataFrame(columns=['date', 'Pivot_Price'])
    
    if last_pivot_date and not df_display.empty:
        start_index = df_display[df_display['date'] == last_pivot_date].index
        
        if not start_index.empty:
            start_index = start_index[0]
            df_extension = df_display.loc[start_index:].copy()
            current_trend_status = trend_analysis['Overall_Trend']
            
            # 筆段延伸的價格點
            if last_pivot_type == 1: # 上一個是頂點，現在延伸應取 Low
                df_extension['Pivot_Price_Extension'] = df_extension['low']
                # 檢查是否下穿前一個底
                if len(df_pivots_display_filtered) >= 2:
                    B_pre = df_pivots_display_filtered.iloc[-2]['Pivot_Price']
                    if df_extension['low'].min() < B_pre:
                        current_trend_status = "⚠️ **潛在趨勢反轉/持續下降 (下穿前底)**"
            elif last_pivot_type == -1: # 上一個是底點，現在延伸應取 High
                df_extension['Pivot_Price_Extension'] = df_extension['high']
                # 檢查是否上穿前一個頂
                if len(df_pivots_display_filtered) >= 2:
                    T_pre = df_pivots_display_filtered.iloc[-2]['Pivot_Price']
                    if df_extension['high'].max() > T_pre:
                        current_trend_status = "✅ **趨勢持續 (上穿前高)**"
            
            if 'Pivot_Price_Extension' in df_extension.columns:
                # 確保延伸線的第一點是最後一個轉折點的價格
                df_extension.loc[start_index, 'Pivot_Price_Extension'] = df_pivots_display_filtered.iloc[-1]['Pivot_Price']
                extend_points = df_extension[['date', 'Pivot_Price_Extension']].rename(columns={'Pivot_Price_Extension': 'Pivot_Price'})

            trend_analysis['Overall_Trend'] = current_trend_status
            trend_desc_final = current_trend_status # 更新圖表標題
            
    if not df_pivots_display_filtered.empty:
        plot_points = df_pivots_display_filtered[['date', 'Pivot_Price']].copy()
        
        # 合併延伸點
        if not extend_points.empty:
            start_date_filter = plot_points['date'].max()
            # 只取延伸點中日期大於等於最後一個轉折點日期的部分
            new_extension = extend_points[extend_points['date'] >= start_date_filter]
            plot_points = pd.concat([plot_points, new_extension], ignore_index=True).drop_duplicates(subset=['date'], keep='last')
            
        fig.add_trace(go.Scatter(
            x=plot_points['date'],
            y=plot_points['Pivot_Price'],
            mode='lines',
            line=dict(color='black', width=2, dash='solid'),
            name='筆段趨勢連線 (嚴格筆段)'
        ), row=1, col=1)

        # 標記頂點和底點
        df_top = df_pivots_display_filtered[df_pivots_display_filtered['Pivot_Type']==1]
        fig.add_trace(go.Scatter(
            x=df_top['date'], y=df_top['Pivot_Price'], mode='markers',
            marker=dict(size=8, color='black', symbol='circle', line=dict(width=1, color='black')),
            name='筆段頂點', hoverinfo='text',
            text=[f"筆段頂: {p:.2f}" for p in df_top['Pivot_Price']], uid='top_pivot_marker',
        ), row=1, col=1)
        
        df_bottom = df_pivots_display_filtered[df_pivots_display_filtered['Pivot_Type']==-1]
        fig.add_trace(go.Scatter(
            x=df_bottom['date'], y=df_bottom['Pivot_Price'], mode='markers',
            marker=dict(size=8, color='black', symbol='circle', line=dict(width=1, color='black')),
            name='筆段底點', hoverinfo='text',
            text=[f"筆段底: {p:.2f}" for p in df_bottom['Pivot_Price']], uid='bottom_pivot_marker',
        ), row=1, col=1)
        
    stock_name = df_display['stock_name'].iloc[0] if 'stock_name' in df_display.columns and not df_display.empty else stock_id_clean
    first_date = df_display['date'].iloc[0].strftime("%Y-%m-%d")
    last_date = df_display['date'].iloc[-1].strftime("%Y-%m-%d")

    fig.update_layout(
        title=dict(
            text=f"{stock_id_clean} ({stock_name}) - {frequency}線趨勢: {trend_desc_final} ({first_date} ~ {last_date})",
            x=0.5, xanchor='center'
        ),
        xaxis_rangeslider_visible=False, hovermode='x unified', dragmode='drawline',
        newshape=dict(line_color='black', line_width=2),
        modebar_add=['drawline', 'drawopenpath', 'drawrect', 'drawcircle', 'eraseshape'],
        yaxis=dict(range=[yaxis_min, yaxis_max]),
        height=1200
    )

    # 確保成交量 y 軸標籤是 K
    fig.update_yaxes(title_text="成交量 (K)", row=2, col=1)
    
    html = fig.to_html(include_plotlyjs='cdn')
    
    return html, None, trend_desc_final, rebound_desc, trend_class
# ----------------- Flask 路由部分 -----------------

# ----------------- 輔助函數部分 (確保可以訪問 SUPABASE_URL, FAVORITE_TABLE, headers, requests) -----------------

def is_favorite(stock_id):
    """
    檢查指定 stock_id 是否已在 FAVORITE_TABLE 中。
    這是為了在渲染 chart.html 時，為 is_favorite 變數提供初始值。
    """
    try:
        # 查詢 Supabase 檢查該股票 ID 是否存在
        res = requests.get(
            f"{SUPABASE_URL}/rest/v1/{FAVORITE_TABLE}", 
            headers=headers, 
            params={"stock_id": f"eq.{stock_id}", "select": "stock_id"}
        )
        res.raise_for_status()
        # 如果返回的 JSON 列表長度大於 0，則表示已加入最愛
        return len(res.json()) > 0
    except Exception:
        # 查詢失敗時，返回 False 以保險
        return False

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
    
    chart_html, error, trend_desc, rebound_desc, trend_class = generate_chart(stock_id, simple_mode=simple_mode, num_rows=num_rows, frequency=frequency)
    
    if error: return f"<h2>{error}</h2><a href='/'>返回</a>"
    
    # 🌟 使用前面定義的函數獲取最愛狀態
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
    chart_html, error, trend_desc, rebound_desc, trend_class = generate_chart(current_stock, simple_mode=simple_mode, num_rows=num_rows, frequency=frequency)
    
    if error: return f"<h2>{error}</h2><a href='/'>返回</a>"
    
    # 🌟 使用前面定義的函數獲取最愛狀態
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

@app.route('/favorites', methods=['GET', 'POST']) # 🌟 允許 GET 請求，以便通過連結訪問
def favorites_page():
    # 統一從 request.values 中獲取參數，兼容 GET 和 POST
    simple_mode = request.values.get('simple_mode') == '1'
    num_rows = request.values.get('num_rows', type=int, default=30)
    frequency = request.values.get('frequency', 'D')
    
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
    # 🌟 修正：如果 stock_name 是空的，使用 stock_id 作為備用名稱
    if not stock_name: stock_name = stock_id
    
    if not stock_id: return jsonify({"message": "股票代號不可為空"}), 400
    
    try:
        # 1. 檢查是否存在
        res_check = requests.get(f"{SUPABASE_URL}/rest/v1/{FAVORITE_TABLE}", headers=headers, params={"stock_id": f"eq.{stock_id}", "select": "stock_id"}); res_check.raise_for_status(); exists = len(res_check.json()) > 0
    except Exception as e: return jsonify({"message": f"檢查最愛失敗: {e}"}), 500

    try:
        if exists:
            # 2. 存在則執行 DELETE (移除)
            res = requests.delete(f"{SUPABASE_URL}/rest/v1/{FAVORITE_TABLE}", headers=headers, params={"stock_id": f"eq.{stock_id}"}); res.raise_for_status()
            return jsonify({"message": f"{stock_name} ({stock_id}) 已從最愛移除", "favorite": False})
        else:
            # 3. 不存在則執行 POST (新增)
            payload = {"stock_id": stock_id, "stock_name": stock_name}
            # 使用 json=payload 和正確的 Content-Type 標頭
            res = requests.post(f"{SUPABASE_URL}/rest/v1/{FAVORITE_TABLE}", headers={**headers, "Content-Type": "application/json"}, json=payload); res.raise_for_status()
            return jsonify({"message": f"{stock_name} ({stock_id}) 已加入最愛", "favorite": True})
    except Exception as e: return jsonify({"message": f"操作最愛失敗: {e}"}), 500

@app.route('/favorites_clear', methods=['POST'])
def favorites_clear():
    try:
        # 使用 neq.null 條件刪除所有記錄
        res = requests.delete(f"{SUPABASE_URL}/rest/v1/{FAVORITE_TABLE}", headers=headers, params={"stock_id": "neq.null"})  
        res.raise_for_status(); return "<script>alert('已刪除所有最愛股票'); window.location.href='/'</script>"
    except Exception as e: return f"<h2>刪除最愛失敗: {e}</h2><a href='/'>返回首頁</a>"

# ----------------- 運行程式 -----------------
if __name__ == '__main__':
    # ... (您的運行代碼) ...
    pass

# ----------------- 運行應用程式 -----------------
if __name__ == '__main__':
    # 僅在本機開發環境使用，實際部署請使用 WSGI 服務器
    app.run(debug=True, host='0.0.0.0', port=5000)