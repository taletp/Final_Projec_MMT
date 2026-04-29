import streamlit as st
import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime
import joblib
from keras.models import load_model

# Import các module tự viết
from src.utils import TIME_FEATURES, STAT_FEATURES, MODEL_PATH, SCALER_TIME_PATH, SCALER_STAT_PATH, LABEL_ENCODER_PATH
from src.real_log_collector import RealNetworkLogCollector
from src.mock_data_generator import MockNetworkFlowGenerator

# --- CẤU HÌNH LOGGING ---
import logging
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

# ==================== CACHED RESOURCES ====================
# Dùng cache_resource để đảm bảo Collector chỉ khởi tạo 1 lần duy nhất (Singleton)
@st.cache_resource
def get_traffic_collector(interface_name):
    collector = RealNetworkLogCollector(interface=interface_name)
    return collector

class RealtimeMonitor:
    def __init__(self):
        """Khởi tạo hệ thống monitoring"""
        try:
            self.model = load_model(MODEL_PATH)
            self.scaler_time = joblib.load(SCALER_TIME_PATH)
            self.scaler_stat = joblib.load(SCALER_STAT_PATH)
            self.le = joblib.load(LABEL_ENCODER_PATH)
            
            # Mock generator cho chế độ Demo
            # (Giả sử bạn có class này trong src.utils hoặc file riêng)
            from src.mock_data_generator import MockNetworkFlowGenerator 
            self.data_generator = MockNetworkFlowGenerator()
        except Exception as e:
            st.error(f"Lỗi load model/scaler: {e}")
            st.stop()

        # Cấu hình mức độ nguy hiểm
        self.attack_severity = {
            'DDoS': 'HIGH', 'DoS Hulk': 'HIGH', 'DoS GoldenEye': 'HIGH',
            'Bot': 'HIGH', 'Web Attack – Brute Force': 'HIGH',
            'Web Attack – Sql Injection': 'HIGH', 'Web Attack – XSS': 'HIGH',
            'Heartbleed': 'HIGH', 'Infiltration': 'HIGH',
            'PortScan': 'MEDIUM', 'FTP-Patator': 'MEDIUM', 'SSH-Patator': 'MEDIUM',
            'DoS slowloris': 'LOW', 'DoS Slowhttptest': 'LOW',
            'BENIGN': 'SAFE'
        }

    def _align_features(self, df):
        """
        Đảm bảo DataFrame có đúng 22 cột features theo thứ tự training
        """
        required_cols = TIME_FEATURES + STAT_FEATURES
        
        # 1. Tạo các cột thiếu (điền 0)
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0
                
        # 2. Chỉ lấy đúng các cột cần thiết theo thứ tự
        return df[required_cols]

    def predict_on_flow(self, flow_data):
        """Dự đoán trên một network flow"""
        try:
            # Chuẩn hóa cột dữ liệu (Align features)
            flow_data = self._align_features(flow_data)

            # Trích xuất features
            X_time = flow_data[TIME_FEATURES].values
            X_stat = flow_data[STAT_FEATURES].values
            
            # Normalize bằng Scaler đã train
            X_time = self.scaler_time.transform(X_time)
            X_stat = self.scaler_stat.transform(X_stat)
            
            # Reshape cho LSTM (Samples, Timesteps, Features)
            # Giả sử TIME_FEATURES training shape là (None, 1, n_features)
            X_time = X_time.reshape(X_time.shape[0], 1, X_time.shape[1])
            
            # Dự đoán
            probs = self.model.predict([X_time, X_stat], verbose=0)
            
            # Xử lý kết quả
            risk_score = np.max(probs[0])
            class_idx = np.argmax(probs[0])
            attack_name = self.le.inverse_transform([class_idx])[0]
            severity = self.attack_severity.get(attack_name, 'UNKNOWN')
            
            # Logic giảm nhiễu: Nếu độ tin cậy thấp -> Benign
            if risk_score < 0.6 and attack_name != 'BENIGN':
                attack_name = 'BENIGN (Low Conf)'
                severity = 'SAFE'

            return {
                'attack_name': attack_name,
                'risk_score': float(risk_score),
                'severity': severity,
                'src_ip': flow_data.get('src_ip', 'Unknown'), # Lấy IP nếu có
                'dst_ip': flow_data.get('dst_ip', 'Unknown')
            }
        except Exception as e:
            return {
                'attack_name': 'ERROR', 'risk_score': 0.0, 
                'severity': 'ERROR', 'error': str(e)
            }

    def calculate_overall_risk(self, results):
        if not results: return 0.0
        attacks = [r for r in results if r['severity'] in ['HIGH', 'MEDIUM', 'LOW']]
        if not attacks: return 0.0
        avg_risk = np.mean([r['risk_score'] for r in attacks])
        # Công thức: Tỉ lệ attack * độ tin cậy * hệ số khuếch đại
        risk_percentage = min(100, (len(attacks) / len(results)) * 100 * avg_risk * 1.5)
        return risk_percentage

def create_monitoring_dashboard():
    st.set_page_config(page_title="Network Security Monitoring", page_icon="🛡️", layout="wide")
    st.title("🛡️ AI-Powered Network Monitoring System")
    st.markdown("---")

    # ==================== SIDEBAR CONFIG ====================
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # 1. Chọn chế độ nguồn dữ liệu
        data_source = st.radio(
            "Data Source",
            ("🛡️ Real-time Interface", "🎲 Mock Data (Demo)"),
            index=0
        )
        
        interface_name = "Wi-Fi" # Giá trị mặc định
        
        if "Real-time" in data_source:
            # Nhập tên card mạng (QUAN TRỌNG)
            interface_name = st.text_input(
                "Network Interface Name", 
                value="Wi-Fi",
                help="Dùng lệnh 'ipconfig' hoặc 'show_interfaces()' của scapy để lấy tên đúng."
            )
            st.info(f"Listening on: {interface_name}")
            
            update_interval = st.slider("Update Interval (s)", 1, 5, 2)
            
        else:
            # Cấu hình cho Mock Data
            batch_size = st.slider("Batch Size", 1, 20, 5)
            update_interval = st.slider("Update Interval (s)", 1, 5, 3)
            attack_dist_type = st.selectbox("Attack Pattern", ["Normal (80% Benign)", "Under Attack (High Risk)"])

        st.markdown("---")
        
        # Các nút điều khiển
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            start_btn = st.button("▶️ START", type="primary", use_container_width=True)
        with col_btn2:
            stop_btn = st.button("⏹️ STOP", use_container_width=True)
        
        reset_btn = st.button("🔄 Reset History", use_container_width=True)

    # ==================== SESSION STATE INIT ====================
    if 'monitoring_active' not in st.session_state: st.session_state.monitoring_active = False
    if 'start_time' not in st.session_state: st.session_state.start_time = datetime.now()
    if 'history_risk' not in st.session_state: st.session_state.history_risk = []
    if 'alerts' not in st.session_state: st.session_state.alerts = []
    if 'total_flows' not in st.session_state: st.session_state.total_flows = 0
    if 'attack_counts' not in st.session_state: st.session_state.attack_counts = 0

    # Logic nút bấm
    if start_btn:
        st.session_state.monitoring_active = True
        if "Real-time" in data_source:
            # Khởi động Collector thật
            collector = get_traffic_collector(interface_name)
            if not collector.sniff_thread or not collector.sniff_thread.is_alive():
                collector.start()
            st.toast(f"Started sniffing on {interface_name}")
            
    if stop_btn:
        st.session_state.monitoring_active = False
        if "Real-time" in data_source:
            collector = get_traffic_collector(interface_name)
            collector.stop()
            st.toast("Stopped sniffing")

    if reset_btn:
        st.session_state.history_risk = []
        st.session_state.alerts = []
        st.session_state.total_flows = 0
        st.session_state.attack_counts = 0
        st.session_state.start_time = datetime.now()
        st.experimental_rerun()

    # ==================== MAIN METRICS UI ====================
    col1, col2, col3, col4 = st.columns(4)
    
    # Tính toán thời gian chạy
    elapsed = datetime.now() - st.session_state.start_time
    elapsed_str = str(elapsed).split('.')[0]
    
    # Lấy giá trị risk mới nhất
    current_risk = st.session_state.history_risk[-1] if st.session_state.history_risk else 0
    
    # Màu trạng thái
    if current_risk > 75: state_color, state_text = "red", "CRITICAL"
    elif current_risk > 40: state_color, state_text = "orange", "WARNING"
    else: state_color, state_text = "green", "SAFE"

    col1.metric("⏱️ Monitor Duration", elapsed_str)
    col2.metric("📊 Current Risk", f"{current_risk:.1f}%", delta=None)
    col3.metric("⚡ Total Attacks", st.session_state.attack_counts)
    col4.markdown(f"#### Status: :{state_color}[{state_text}]")

    # ==================== PROCESSING ENGINE ====================
    monitor = RealtimeMonitor()
    
    # Placeholder cho UI update
    chart_place = st.empty()
    alert_place = st.empty()
    
    if st.session_state.monitoring_active:
        new_flows_df = pd.DataFrame()
        
        # 1. THU THẬP DỮ LIỆU
        if "Real-time" in data_source:
            # Lấy dữ liệu từ Collector thật
            collector = get_traffic_collector(interface_name)
            new_flows_df = collector.get_new_flows()
        else:
            # Lấy dữ liệu giả lập
            dist = {'BENIGN': 0.8} if "Normal" in attack_dist_type else {'BENIGN': 0.2, 'DDoS': 0.8}
            new_flows_df = monitor.data_generator.generate_batch_flows(batch_size, dist)

        # 2. XỬ LÝ NẾU CÓ DỮ LIỆU
        if not new_flows_df.empty:
            st.session_state.total_flows += len(new_flows_df)
            batch_results = []
            
            for idx, row in new_flows_df.iterrows():
                # Predict từng flow
                # Chuyển row thành DataFrame 1 dòng để giữ tên cột
                single_flow = new_flows_df.iloc[[idx]]
                result = monitor.predict_on_flow(single_flow)
                batch_results.append(result)
                
                # Update counters & Alerts
                if result['severity'] in ['HIGH', 'MEDIUM']:
                    st.session_state.attack_counts += 1
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    # Tạo thông báo IP nếu có
                    ip_info = ""
                    if 'src_ip' in result and isinstance(result['src_ip'], str):
                         ip_info = f" | {result['src_ip']} -> {result['dst_ip']}"

                    msg = f"[{timestamp}] 🚨 {result['attack_name']} Detected (Risk: {result['risk_score']:.2f}){ip_info}"
                    st.session_state.alerts.insert(0, msg) # Thêm vào đầu danh sách

            # Tính toán Risk tổng thể của batch này
            batch_risk = monitor.calculate_overall_risk(batch_results)
            st.session_state.history_risk.append(batch_risk)
            
            # Giới hạn lịch sử biểu đồ
            if len(st.session_state.history_risk) > 100:
                st.session_state.history_risk.pop(0)
                
            # Log ra màn hình console (optional)
            # print(f"Processed {len(new_flows_df)} flows. Risk: {batch_risk:.2f}%")
            
        else:
            # Nếu không có dữ liệu thật, lặp lại risk cũ để biểu đồ chạy tiếp
            if st.session_state.history_risk:
                st.session_state.history_risk.append(st.session_state.history_risk[-1] * 0.95) # Giảm dần risk nếu im lặng

        # 3. TỰ ĐỘNG REFRESH
        time.sleep(update_interval)
        st.rerun()

    # ==================== VISUALIZATION ====================
    
    # Biểu đồ Real-time
    with chart_place.container():
        st.subheader("📈 Network Threat Level (Real-time)")
        if st.session_state.history_risk:
            chart_data = pd.DataFrame(st.session_state.history_risk, columns=["Risk Percentage"])
            st.line_chart(chart_data, height=250)
        else:
            st.info("Waiting for traffic data...")

    # Khu vực cảnh báo & Chi tiết
    col_log, col_detail = st.columns([1, 1])
    
    with col_log:
        st.subheader("🚨 Security Alerts Log")
        alert_container = st.container(height=300)
        if st.session_state.alerts:
            for alert in st.session_state.alerts[:20]: # Hiển thị 20 alert mới nhất
                if "HIGH" in alert or "CRITICAL" in alert: # Logic màu mè
                    alert_container.error(alert)
                else:
                    alert_container.warning(alert)
        else:
            alert_container.success("No security threats detected recently.")

    with col_detail:
        st.subheader("📋 Last Captured Batch Details")
        if st.session_state.monitoring_active and 'new_flows_df' in locals() and not new_flows_df.empty:
            # Hiển thị bảng rút gọn
            display_cols = ['src_ip', 'dst_ip', 'proto', 'Flow Duration']
            # Lọc các cột tồn tại để tránh lỗi
            valid_cols = [c for c in display_cols if c in new_flows_df.columns]
            
            # Thêm cột dự đoán vào để xem
            display_df = new_flows_df[valid_cols].copy()
            if 'batch_results' in locals():
                display_df['Prediction'] = [r['attack_name'] for r in batch_results]
            
            st.dataframe(display_df.head(10), use_container_width=True)
        else:
            st.text("No active flows in current buffer.")

if __name__ == "__main__":
    create_monitoring_dashboard()