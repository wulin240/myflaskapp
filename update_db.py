#!/usr/bin/env python3
import requests
import pandas as pd
import datetime
import yfinance as yf
import re
import os
import json
import tempfile
from zoneinfo import ZoneInfo
from typing import List
from tqdm import tqdm  # 進度條

# ----------------------------
# Supabase 設定
SUPABASE_URL = "https://djhdpltrhlhqfxmwniki.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRqaGRwbHRyaGxocWZ4bXduaWtpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjMwNjQwODgsImV4cCI6MjA3ODY0MDA4OH0.jwSPe-HMHxv2xGCjS42O5Cjby0KtgsHEStlQWs0cyPk"
TABLE_NAME = "stock_data"

url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"
headers = {
    "apikey": SUPABASE_KEY.strip(),
    "Authorization": f"Bearer {SUPABASE_KEY.strip()}",
    "Content-Type": "application/json",
    "Prefer": "resolution=merge-duplicates"
}

# ----------------------------
# 讀取股票代號清單
stock_list = []
stock_name_dict = {}
with open("list.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) >= 2:
            stock_id = parts[0].strip()
            stock_name = parts[1].strip()
            stock_list.append(stock_id)
            stock_name_dict[stock_id] = stock_name

print(f"共 {len(stock_list)} 支股票")

# ----------------------------
def normalize_stock_id(stock_id: str) -> str:
    return re.sub(r'\D', '', stock_id)

def get_latest_date(stock_id: str):
    params = {"select": "date", "stock_id": f"eq.{normalize_stock_id(stock_id)}", "order": "date.desc", "limit": 1}
    try:
        res = requests.get(url, headers=headers, params=params, timeout=20)
        if res.status_code == 200 and res.json():
            last_date_str = res.json()[0]['date']
            return datetime.datetime.strptime(last_date_str, "%Y-%m-%d").date()
    except Exception as e:
        print(f"⚠️ get_latest_date {stock_id} request failed: {e}")
    return None

def sanitize_record(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if v is None:
            out[k] = None
            continue
        if isinstance(v, (str, bool)):
            out[k] = v
            continue
        if isinstance(v, float):
            if pd.isna(v):
                out[k] = None
            else:
                if k == 'volume':
                    try:
                        out[k] = int(round(v))
                    except:
                        out[k] = None
                else:
                    out[k] = float(v)
            continue
        if isinstance(v, int):
            out[k] = int(v)
            continue
        try:
            if pd.isna(v):
                out[k] = None
            else:
                out[k] = v.item() if hasattr(v, 'item') else v
        except:
            out[k] = str(v)
    return out

def upload_batch(batch: List[dict], on_conflict: str = "stock_id,date") -> bool:
    if not batch:
        return True
    sanitized = [sanitize_record(r) for r in batch]

    def try_post(records: List[dict]):
        try:
            r = requests.post(url + f"?on_conflict={on_conflict}",
                              headers=headers,
                              json=records,
                              timeout=60)
            if r.status_code in (200, 201):
                return True, r.text
            else:
                return False, r.text
        except Exception as e:
            return False, str(e)

    success, resp = try_post(sanitized)
    if success:
        print(f"✅ 已上傳 {len(sanitized)} 筆資料")
        return True

    print(f"❌ 批次上傳失敗 (嘗試分段處理)：{resp[:200]}")
    if len(sanitized) <= 16:
        bad_indices = []
        for idx, rec in enumerate(sanitized):
            ok, rtxt = try_post([rec])
            if not ok:
                bad_indices.append((idx, rec, rtxt))
        if bad_indices:
            print("⚠️ 下列單筆資料格式有問題，已跳過（顯示前三筆）：")
            for bi in bad_indices[:3]:
                print(f"  idx={bi[0]} err={bi[2]} rec={bi[1]}")
            good_records = [r for i, r in enumerate(sanitized) if i not in {bi[0] for bi in bad_indices}]
            if good_records:
                ok2, r2 = try_post(good_records)
                if ok2:
                    print(f"✅ 已上傳剩餘 {len(good_records)} 筆（跳過壞筆）")
                    return True
            return False
        else:
            return False
    else:
        mid = len(sanitized) // 2
        left = sanitized[:mid]
        right = sanitized[mid:]
        ok_left = upload_batch(left, on_conflict)
        ok_right = upload_batch(right, on_conflict)
        return ok_left and ok_right

# ----------------------------
def fetch_stock(stock: str):
    stock_num = normalize_stock_id(stock)
    stock_name = stock_name_dict.get(stock, "")

    last_date = get_latest_date(stock)
    if last_date:
        start_date_dt = last_date + datetime.timedelta(days=1)
    else:
        start_date_dt = datetime.date(2024, 1, 1)

    tz = ZoneInfo("Asia/Taipei")
    now_tw = datetime.datetime.now(tz)
    today_tw = now_tw.date()
    market_close = datetime.time(hour=13, minute=30)
    if now_tw.time() < market_close:
        end_date_dt = today_tw - datetime.timedelta(days=1)
    else:
        end_date_dt = today_tw

    if start_date_dt > end_date_dt:
        return []

    yf_start = start_date_dt.strftime("%Y-%m-%d")
    yf_end = (end_date_dt + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        df = yf.download(stock, start=yf_start, end=yf_end, progress=False, auto_adjust=False, threads=False)
    except Exception as e:
        print(f"⚠️ {stock} YF 下載失敗: {e}")
        return []

    if df is None or df.empty:
        return []

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(i) for i in col if i]) for col in df.columns.values]

    if 'Date' not in df.columns:
        df = df.reset_index()

    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if 'open' in cl and 'open' not in col_map:
            col_map['open'] = c
        elif 'high' in cl and 'high' not in col_map:
            col_map['high'] = c
        elif 'low' in cl and 'low' not in col_map:
            col_map['low'] = c
        elif 'close' in cl and 'adj' not in cl and 'close' not in col_map:
            col_map['close'] = c
        elif 'volume' in cl and 'volume' not in col_map:
            col_map['volume'] = c
    if 'close' not in col_map:
        for c in df.columns:
            cl = c.lower()
            if 'adj close' in cl and 'close' not in col_map:
                col_map['close'] = c
                break
    if len(col_map) < 5:
        print(f"⚠️ {stock} 欄位不足，抓到欄位：{list(df.columns)}, 跳過")
        return []

    needed_cols = ['Date', col_map['open'], col_map['high'], col_map['low'], col_map['close'], col_map['volume']]
    df = df.loc[:, needed_cols].copy()
    df.rename(columns={
        col_map['open']: 'open',
        col_map['high']: 'high',
        col_map['low']: 'low',
        col_map['close']: 'close',
        col_map['volume']: 'volume'
    }, inplace=True)

    df['date'] = pd.to_datetime(df['Date']).dt.strftime("%Y-%m-%d")

    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

    if df[['open', 'high', 'low', 'close']].isna().all(axis=1).all():
        return []

    df['stock_id'] = stock_num
    df['stock_name'] = stock_name

    df_out = df[['stock_id', 'stock_name', 'date', 'open', 'high', 'low', 'close', 'volume']].copy()

    records = []
    for _, row in df_out.iterrows():
        rec = {
            'stock_id': str(row['stock_id']),
            'stock_name': row['stock_name'] if pd.notna(row['stock_name']) else None,
            'date': row['date'],
            'open': None if pd.isna(row['open']) else float(row['open']),
            'high': None if pd.isna(row['high']) else float(row['high']),
            'low': None if pd.isna(row['low']) else float(row['low']),
            'close': None if pd.isna(row['close']) else float(row['close']),
            'volume': None if pd.isna(row['volume']) else int(round(float(row['volume'])))
        }
        records.append(rec)

    return records

# ----------------------------
failed_stocks = []
all_rows = []
batch_size = 500
tmp_file = os.path.join(tempfile.gettempdir(), "stock_tmp.json")

if os.path.exists(tmp_file):
    try:
        with open(tmp_file, "r", encoding="utf-8") as f:
            all_rows = json.load(f)
        print(f"從暫存檔載入 {len(all_rows)} 筆未上傳資料")
    except Exception:
        all_rows = []

# ----------------------------
# 單線程抓取，使用進度條
for stock in tqdm(stock_list, desc="抓取進度"):
    try:
        rows = fetch_stock(stock)
    except Exception as e:
        print(f"⚠️ {stock} fetch exception: {e}")
        failed_stocks.append(stock)
        continue

    if rows:
        all_rows.extend(rows)
    else:
        failed_stocks.append(stock)

    if len(all_rows) >= batch_size:
        ok = upload_batch(all_rows)
        if ok:
            all_rows = []
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        else:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(all_rows, f, ensure_ascii=False, indent=2)
            print(f"已將 {len(all_rows)} 筆資料寫入暫存檔：{tmp_file}")

# 上傳剩餘資料
if all_rows:
    if upload_batch(all_rows):
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
    else:
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, ensure_ascii=False, indent=2)
        print(f"上傳失敗，已將剩餘 {len(all_rows)} 筆寫入暫存檔：{tmp_file}")

failed_unique = sorted(set(failed_stocks))
if failed_unique:
    print(f"\n⚠️ 以下股票抓取或處理可能失敗（需人工檢查）：{failed_unique}")

print("\n✅ 單線程抓取完成！")
