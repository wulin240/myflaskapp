from flask import Flask, render_template, request
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import urllib.parse
import requests

app = Flask(__name__)

# -----------------------------
# Supabase 設定
SUPABASE_URL = "https://djhdpltrhlhqfxmwniki.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRqaGRwbHRyaGxocWZ4bXduaWtpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjMwNjQwODgsImV4cCI6MjA3ODY0MDA4OH0.jwSPe-HMHxv2xGCjS42O5Cjby0KtgsHEStlQWs0cyPk"
TABLE_NAME = "stock_data"

headers = {
    "apikey": SUPABASE_KEY.strip(),
    "Authorization": f"Bearer {SUPABASE_KEY.strip()}"
}

# -----------------------------
def fetch_stock_data(stock_id):
    """從 Supabase 抓資料"""
    stock_id_clean = stock_id.replace(".TW","").replace(".TWO","")
    params = {
        "stock_id": f"eq.{stock_id_clean}",
        "order": "date.asc"
    }
    try:
        res = requests.get(f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}", headers=headers, params=params, timeout=30)
        res.raise_for_status()
        data = res.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"⚠️ Supabase 讀取 {stock_id} 失敗: {e}")
        return pd.DataFrame()

# -----------------------------
def generate_chart(stock_id_clean, start_date=None, end_date=None, simple_mode=False, num_rows=30):
    df = fetch_stock_data(stock_id_clean)
    if df.empty:
        return None, f"{stock_id_clean} 無資料"

    # 篩選日期
    if start_date and end_date:
        df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
    df = df.tail(num_rows)
    if df.empty:
        return None, f"{stock_id_clean} 在 {start_date} ~ {end_date} 無資料"

    # 計算黑色折線
    df['line'] = df.apply(lambda row: row['high'] if row['close']>row['open'] else (row['low'] if row['close']<row['open'] else (row['open']+row['close'])/2), axis=1)

    # 均線
    for ma in [5,10,20,60]:
        if len(df) >= ma:
            df[f'MA{ma}'] = df['close'].rolling(ma).mean()
        else:
            df[f'MA{ma}'] = df['close'].expanding().mean()

    # 成交量均線
    df['VOL5'] = df['volume'].rolling(5).mean()
    df['VOL20'] = df['volume'].rolling(20).mean()

    # ATR
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L','H-PC','L-PC']].max(axis=1)
    df['ATR14'] = df['TR'].rolling(14).mean().round(3)

    # VWAP
    df['TP'] = (df['high'] + df['low'] + df['close'])/3
    df['TPV'] = df['TP']*df['volume']
    df['VWAP'] = df['TPV'].cumsum()/df['volume'].cumsum()

    # K線圖範圍
    min_price = df[['low','MA5','MA10','MA20','MA60','VWAP']].min().min()
    max_price = df[['high','MA5','MA10','MA20','MA60','VWAP']].max().max()
    price_range = max_price - min_price
    yaxis_min = min_price - price_range/4
    yaxis_max = max_price + price_range/4

    # Plotly 畫圖
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.07,
                        row_heights=[0.7,0.15,0.15],
                        subplot_titles=("K線圖","成交量","ATR"))

    # K線與均線
    if not simple_mode:
        fig.add_trace(go.Candlestick(
            x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            increasing_line_color='red', decreasing_line_color='green',
            name='K線'
        ), row=1, col=1)

        ma_colors = {5:'blue',10:'orange',20:'purple',60:'black'}
        for ma in [5,10,20,60]:
            fig.add_trace(go.Scatter(
                x=df['date'], y=df[f'MA{ma}'], mode='lines', line=dict(color=ma_colors[ma], width=1), name=f'MA{ma}'
            ), row=1, col=1)

    # 折線與VWAP
    fig.add_trace(go.Scatter(x=df['date'], y=df['line'], mode='lines', line=dict(color='black', width=1), name='折線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['VWAP'], mode='lines', line=dict(color='magenta', width=2, dash='dot'), name='VWAP'), row=1, col=1)

    # 成交量
    vol_color = df.apply(lambda row: 'red' if row['close']>row['open'] else ('green' if row['close']<row['open'] else 'yellow'), axis=1)
    fig.add_trace(go.Bar(x=df['date'], y=df['volume']/1000, name='成交量', marker_color=vol_color), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['VOL5']/1000, mode='lines', line=dict(color='blue', width=1), name='VOL5'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['VOL20']/1000, mode='lines', line=dict(color='orange', width=1), name='VOL20'), row=2, col=1)

    # ATR
    fig.add_trace(go.Scatter(x=df['date'], y=df['ATR14'], mode='lines', line=dict(color='red', width=1), name='ATR14'), row=3, col=1)

    stock_name = df['stock_name'].iloc[0] if 'stock_name' in df.columns else stock_id_clean
    first_date = df['date'].iloc[0].strftime("%Y-%m-%d")
    last_date = df['date'].iloc[-1].strftime("%Y-%m-%d")

    fig.update_layout(
        title=dict(text=f"{stock_id_clean} ({stock_name}) K線圖 ({first_date} ~ {last_date})", x=0.5, xanchor='center'),
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        dragmode='drawline',
        newshape=dict(line_color='black', line_width=2),
        modebar_add=['drawline', 'drawrect', 'drawcircle', 'eraseshape'],
        yaxis=dict(range=[yaxis_min, yaxis_max]),
        height=1200
    )

    html = fig.to_html(include_plotlyjs='cdn')
    return html, None

# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    stock_id = request.form['stock_id'].strip()
    simple_mode = request.form.get('simple_mode') == '1'
    num_rows = request.form.get('num_rows', type=int, default=30)
    chart_html, error = generate_chart(stock_id, simple_mode=simple_mode, num_rows=num_rows)
    if error:
        return f"<h2>{error}</h2><a href='/'>返回</a>"
    return render_template('chart.html', chart_html=chart_html, stock_id=stock_id, stock_list=stock_id, current_index=0, simple_mode=simple_mode, num_rows=num_rows)

# -----------------------------
@app.route('/chart/<stock_id>/')
@app.route('/chart/<stock_id>')
def chart_from_list(stock_id):
    stock_id = stock_id.strip()
    simple_mode = request.args.get('simple_mode') == '1'
    num_rows = request.args.get('num_rows', type=int, default=30)
    stock_list = request.args.get('list', '')
    index = request.args.get('index', type=int, default=0)
    stock_ids = stock_list.split(',') if stock_list else [stock_id]

    if index < 0: index = 0
    if index >= len(stock_ids): index = len(stock_ids)-1

    current_stock = stock_ids[index]
    chart_html, error = generate_chart(current_stock, simple_mode=simple_mode, num_rows=num_rows)
    if error:
        return f"<h2>{error}</h2><a href='/'>返回</a>"

    return render_template('chart.html',
                           chart_html=chart_html,
                           stock_id=current_stock,
                           stock_list=','.join(stock_ids),
                           current_index=index,
                           simple_mode=simple_mode,
                           num_rows=num_rows)

# -----------------------------
# 篩選路由 (分頁抓取 + 靈活最近資料)
@app.route('/filter', methods=['POST'])
def filter_stocks():
    import urllib.parse
    import pandas as pd
    from datetime import datetime, timedelta

    # 取得篩選條件
    volume_min = request.form.get('volume_min', type=float, default=0)
    trend_type = request.form.get('trend_type', '')
    adr14_min = request.form.get('change_min', type=float, default=0)
    simple_mode = request.form.get('simple_mode') == '1'
    num_rows = request.form.get('num_rows', type=int, default=60)  # 只控制 K 棒顯示
    recent_days = request.form.get('recent_days', type=int, default=30)  # 可調整抓最近幾天資料

    # 計算最近日期
    recent_date = (datetime.today() - timedelta(days=recent_days)).strftime("%Y-%m-%d")

    all_data = []
    limit = 1000
    offset = 0

    while True:
        try:
            res = requests.get(
                f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}",
                headers=headers,
                params={
                    "select": "stock_id,stock_name,volume,open,close,high,low,date",
                    "date": f"gte.{recent_date}",
                    "order": "stock_id.asc,date.asc",
                    "limit": limit,
                    "offset": offset
                },
                timeout=60
            )
            res.raise_for_status()
            data = res.json()
            if not data:
                break
            all_data.extend(data)
            if len(data) < limit:
                break
            offset += limit
        except Exception as e:
            return f"<h2>Supabase 讀取資料失敗: {e}</h2><a href='/'>返回</a>"

    if not all_data:
        return "<h2>Supabase 沒有資料</h2><a href='/'>返回</a>"

    df = pd.DataFrame(all_data)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(['stock_id','date'], inplace=True)

    # 計算 DailyRangePct
    df['DailyRangePct'] = ((df['close'] - df['open']).abs() / df['open']) * 100

    # 產生篩選結果
    result_rows = []

    for sid, sub in df.groupby('stock_id'):
        sub = sub.sort_values('date')
        if len(sub) < 2:
            continue

        # **第一步：最新一筆成交量判斷**
        latest = sub.iloc[-1]
        if latest['volume'] < volume_min:
            continue  # 直接跳過不處理這支股票

        # 計算最近 14 筆 ADR
        adr14_value = sub['DailyRangePct'].tail(14).mean()
        if adr14_value < adr14_min:
            continue

        prev_high = sub['high'].iloc[:-1].max()
        prev_low = sub['low'].iloc[:-1].min()
        last_high = latest['high']
        last_low = latest['low']

        if last_high > prev_high and last_low > prev_low:
            trend = "多頭"
        elif last_high < prev_high and last_low < prev_low:
            trend = "空頭"
        else:
            trend = "盤整"

        if trend_type and trend != trend_type:
            continue

        result_rows.append({
            'stock_id': sid,
            'stock_name': latest.get('stock_name',''),
            'volume': latest['volume'],
            'adr14': round(adr14_value,2),
            'trend': trend
        })

    if not result_rows:
        return "<h2>沒有符合條件的股票</h2><a href='/'>返回</a>"

    # **全部符合條件的股票都列出來，不限制數量**
    result = pd.DataFrame(result_rows).reset_index(drop=True)
    count = len(result)
    stock_ids = [row['stock_id'] for _, row in result.iterrows()]
    list_param = urllib.parse.quote(','.join(stock_ids))

    # 生成 HTML
    html = f"""
    <h2>篩選結果（共 {count} 筆）</h2>
    <table border='1' cellpadding='6' style='margin-left:0; text-align:left;'>
    <thead>
      <tr>
        <th>股票代號</th>
        <th>股票名稱</th>
        <th>成交量</th>
        <th>ADR14(%)</th>
        <th>趨勢</th>
      </tr>
    </thead>
    <tbody>
    """
    for idx, row in result.iterrows():
        simple_param = "1" if simple_mode else "0"
        html += (
            f"<tr>"
            f"<td><a href='/chart/{row['stock_id']}?simple_mode={simple_param}&num_rows={num_rows}&list={list_param}&index={idx}'>{row['stock_id']}</a></td>"
            f"<td>{row['stock_name']}</td>"
            f"<td>{int(row['volume'])}</td>"
            f"<td>{row['adr14']:.2f}</td>"
            f"<td>{row['trend']}</td>"
            f"</tr>"
        )
    html += "</tbody></table><br><a href='/'>返回</a>"

    return html


# -----------------------------
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
