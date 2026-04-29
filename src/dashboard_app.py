import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from collections import deque
import time
import threading
from datetime import datetime
import joblib
from keras.models import load_model

# --- IMPORT MODULE CỦA BẠN ---
from src.utils import TIME_FEATURES, STAT_FEATURES, MODEL_PATH, SCALER_TIME_PATH, SCALER_STAT_PATH, LABEL_ENCODER_PATH
# Giả sử file real_log_collector.py và utils.py đã nằm đúng chỗ
from src.real_log_collector import RealNetworkLogCollector
from src.mock_data_generator import MockNetworkFlowGenerator 

# ==================== GLOBAL STATE & CONFIG ====================
# Dùng Global variable để lưu trữ state giữa các lần callback
# (Lưu ý: Cách này chỉ dùng cho local single-user, production cần Redis)
class AppState:
    def __init__(self):
        self.collector = None
        self.is_monitoring = False
        self.data_buffer = deque(maxlen=1000) # Chỉ lưu 1000 flows gần nhất để hiển thị
        self.attack_history = deque(maxlen=100) # Lưu lịch sử tấn công để vẽ biểu đồ
        self.total_flows = 0
        self.malicious_count = 0
        
        # Load AI Model
        try:
            self.model = load_model(MODEL_PATH)
            self.scaler_time = joblib.load(SCALER_TIME_PATH)
            self.scaler_stat = joblib.load(SCALER_STAT_PATH)
            self.le = joblib.load(LABEL_ENCODER_PATH)
            print(">>> Model loaded successfully!")
        except Exception as e:
            print(f"!!! Error loading model: {e}")

    def predict(self, df):
        """Hàm dự đoán nhanh"""
        try:
            # Align features
            required = TIME_FEATURES + STAT_FEATURES
            for col in required:
                if col not in df.columns: df[col] = 0.0
            
            X_time = self.scaler_time.transform(df[TIME_FEATURES].values)
            X_stat = self.scaler_stat.transform(df[STAT_FEATURES].values)
            X_time = X_time.reshape(X_time.shape[0], 1, X_time.shape[1])
            
            probs = self.model.predict([X_time, X_stat], verbose=0)
            
            results = []
            for i, prob in enumerate(probs):
                idx = np.argmax(prob)
                risk = np.max(prob)
                label = self.le.inverse_transform([idx])[0]
                
                # Logic giảm nhiễu
                if risk < 0.6 and label != 'BENIGN':
                    label = 'BENIGN (Low Conf)'
                
                results.append((label, risk))
            return results
        except Exception as e:
            print(f"Prediction Error: {e}")
            return [("ERROR", 0.0)] * len(df)

state = AppState()

# ==================== DASH APP INIT ====================
# Sử dụng theme CYBORG (Giao diện tối/Hacker style)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Network Sentinel AI"

# ==================== LAYOUT COMPONENTS ====================

# 1. Header
header = dbc.Navbar(
    dbc.Container([
        html.A(
            dbc.Row([
                dbc.Col(html.I(className="bi bi-shield-lock-fill", style={"fontSize": "2rem"})),
                dbc.Col(dbc.NavbarBrand("NETWORK SENTINEL AI", className="ms-2")),
            ], align="center"),
            href="#", style={"textDecoration": "none"},
        ),
        dbc.Badge("SYSTEM READY", color="success", className="ms-auto", id="system-status-badge"),
    ]),
    color="dark", dark=True, className="mb-4"
)

# 2. Control Panel (Sidebar logic)
controls = dbc.Card([
    dbc.CardHeader("Control Panel"),
    dbc.CardBody([
        html.Label("Network Interface:"),
        dbc.Input(id="interface-input", placeholder="e.g., Wi-Fi", value="Wi-Fi", type="text", className="mb-2"),
        
        html.Label("Data Source:"),
        dcc.Dropdown(
            id="source-dropdown",
            options=[
                {'label': '🔴 Real-time Sniffer', 'value': 'real'},
                {'label': '🎲 Mock Data (Demo)', 'value': 'mock'}
            ],
            value='real',
            clearable=False,
            className="text-dark mb-3" # text-dark để chữ đen trên nền trắng của dropdown
        ),

        dbc.Row([
            dbc.Col(dbc.Button("START", id="btn-start", color="success", className="w-100"), width=6),
            dbc.Col(dbc.Button("STOP", id="btn-stop", color="danger", className="w-100"), width=6),
        ]),
        
        html.Hr(),
        html.Label("Graph Feature (Y-Axis):"),
        dcc.Dropdown(
            id="feature-dropdown",
            options=[{'label': f, 'value': f} for f in TIME_FEATURES + ['Flow Bytes/s', 'Flow Packets/s']],
            value='Flow Bytes/s',
            clearable=False,
            className="text-dark"
        )
    ])
], className="mb-4")

# 3. KPI Cards
kpi_cards = dbc.Row([
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.H4("Total Flows", className="card-title text-muted"),
            html.H2("0", id="kpi-total", className="text-light")
        ])
    ], color="secondary", outline=True), width=4),
    
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.H4("Threats Detected", className="card-title text-danger"),
            html.H2("0", id="kpi-threats", className="text-danger")
        ])
    ], color="danger", outline=True), width=4),
    
    dbc.Col(dbc.Card([
        dbc.CardBody([
            html.H4("Current Risk Level", className="card-title text-warning"),
            html.H2("0%", id="kpi-risk", className="text-warning")
        ])
    ], color="warning", outline=True), width=4),
], className="mb-4")

# 4. Main Graph
main_graph = dbc.Card([
    dbc.CardHeader("Real-time Traffic Analysis"),
    dbc.CardBody([
        dcc.Graph(id="live-graph", animate=False, config={'displayModeBar': False})
    ])
], className="mb-4")

# 5. Data Table (Filter & Details)
# Chọn các cột quan trọng để hiển thị
table_cols = ['Timestamp', 'src_ip', 'dst_ip', 'proto', 'Label', 'Risk Score', 'Flow Duration', 'Flow Bytes/s']
data_table = dbc.Card([
    dbc.CardHeader("Recent Network Packets (Filter Enabled)"),
    dbc.CardBody([
        dash_table.DataTable(
            id='traffic-table',
            columns=[{"name": i, "id": i} for i in table_cols],
            data=[],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
            style_cell={
                'backgroundColor': 'rgb(50, 50, 50)',
                'color': 'white',
                'textAlign': 'left'
            },
            # TÍNH NĂNG MẠNH MẼ CỦA DASH: FILTERING
            filter_action="native",  # Cho phép gõ filter vào header cột
            sort_action="native",    # Cho phép click header để sort
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Label} != "BENIGN"'},
                    'backgroundColor': 'rgba(255, 0, 0, 0.3)',
                    'color': 'white'
                }
            ]
        )
    ])
])

# ==================== APP LAYOUT ====================
app.layout = dbc.Container([
    header,
    dbc.Row([
        dbc.Col(controls, width=3),
        dbc.Col([kpi_cards, main_graph], width=9)
    ]),
    dbc.Row([
        dbc.Col(data_table, width=12)
    ]),
    # Interval component để trigger update mỗi 2 giây
    dcc.Interval(id='interval-component', interval=2000, n_intervals=0)
], fluid=True)


# ==================== CALLBACKS (LOGIC) ====================

@app.callback(
    [Output("kpi-total", "children"),
     Output("kpi-threats", "children"),
     Output("kpi-risk", "children"),
     Output("live-graph", "figure"),
     Output("traffic-table", "data"),
     Output("system-status-badge", "children"),
     Output("system-status-badge", "color")],
    [Input("interval-component", "n_intervals"),
     Input("btn-start", "n_clicks"),
     Input("btn-stop", "n_clicks")],
    [State("interface-input", "value"),
     State("source-dropdown", "value"),
     State("feature-dropdown", "value")]
)
def update_metrics(n, start_clicks, stop_clicks, iface_name, source_type, feature_name):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # 1. XỬ LÝ NÚT BẤM START/STOP
    if button_id == "btn-start" and not state.is_monitoring:
        state.is_monitoring = True
        state.data_buffer.clear()
        state.total_flows = 0
        state.malicious_count = 0
        
        if source_type == 'real':
            # Khởi tạo Collector thật
            state.collector = RealNetworkLogCollector(interface=iface_name)
            state.collector.start()
        else:
            # Mock generator
            state.collector = None # Sẽ xử lý trong loop

    elif button_id == "btn-stop" and state.is_monitoring:
        state.is_monitoring = False
        if state.collector:
            state.collector.stop()
            state.collector = None

    # Status Badge
    status_text = "MONITORING ACTIVE" if state.is_monitoring else "SYSTEM IDLE"
    status_color = "success" if state.is_monitoring else "secondary"

    # 2. THU THẬP DỮ LIỆU MỚI (CHỈ KHI MONITORING ACTIVE)
    if state.is_monitoring:
        new_df = pd.DataFrame()
        
        if source_type == 'real' and state.collector:
            new_df = state.collector.get_new_flows()
        elif source_type == 'mock':
            gen = MockNetworkFlowGenerator()
            # Random tấn công cho vui
            dist = {'BENIGN': 0.8, 'DDoS': 0.2}
            new_df = gen.generate_batch_flows(num_flows=5, attack_distribution=dist)

        # 3. DỰ ĐOÁN & CẬP NHẬT BUFFER
        if not new_df.empty:
            state.total_flows += len(new_df)
            
            # Predict
            predictions = state.predict(new_df)
            
            # Gán nhãn vào DataFrame
            new_df['Label'] = [p[0] for p in predictions]
            new_df['Risk Score'] = [round(p[1], 2) for p in predictions]
            new_df['Timestamp'] = datetime.now().strftime("%H:%M:%S")
            
            # Cập nhật số liệu
            malicious = new_df[new_df['Label'] != 'BENIGN']
            state.malicious_count += len(malicious)
            
            # Thêm vào buffer (deque tự động xóa cũ nếu đầy)
            for idx, row in new_df.iterrows():
                state.data_buffer.append(row.to_dict())

    # 4. CHUẨN BỊ DỮ LIỆU HIỂN THỊ
    # Chuyển buffer thành DataFrame để dễ xử lý
    df_display = pd.DataFrame(list(state.data_buffer))
    
    if df_display.empty:
        return "0", "0", "0%", go.Figure(), [], status_text, status_color

    # Tính KPI
    risk_pct = 0
    if state.total_flows > 0:
        risk_pct = (state.malicious_count / state.total_flows) * 100

    # Vẽ biểu đồ
    # Lấy 100 điểm dữ liệu mới nhất
    df_graph = df_display.tail(100)
    
    fig = go.Figure()
    
    # Line chart cho Benign
    benign_df = df_graph[df_graph['Label'] == 'BENIGN']
    fig.add_trace(go.Scatter(
        x=benign_df['Timestamp'], 
        y=benign_df[feature_name],
        mode='markers+lines',
        name='Normal Traffic',
        line=dict(color='#00cc96', width=2),
        marker=dict(size=6)
    ))
    
    # Scatter chart cho Attack (Điểm đỏ nổi bật)
    attack_df = df_graph[df_graph['Label'] != 'BENIGN']
    if not attack_df.empty:
        fig.add_trace(go.Scatter(
            x=attack_df['Timestamp'], 
            y=attack_df[feature_name],
            mode='markers',
            name='Attack Detected',
            marker=dict(color='#EF553B', size=12, symbol='x')
        ))

    fig.update_layout(
        template='plotly_dark',
        margin=dict(l=20, r=20, t=20, b=20),
        height=300,
        legend=dict(orientation="h", y=1.1),
        paper_bgcolor='rgba(0,0,0,0)', # Trong suốt để ăn theo theme Dashboard
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # Format bảng dữ liệu
    table_data = df_display.tail(50).to_dict('records') # Lấy 50 dòng cuối cho bảng
    # Đảo ngược để dòng mới nhất lên đầu
    table_data.reverse()

    return (
        f"{state.total_flows:,}", 
        f"{state.malicious_count:,}", 
        f"{risk_pct:.1f}%", 
        fig, 
        table_data, 
        status_text, 
        status_color
    )

# ==================== RUN SERVER ====================
if __name__ == "__main__":
    # debug=True để tự reload khi sửa code
    # use_reloader=False để tránh lỗi thread với Scapy
    app.run(debug=True, use_reloader=False)