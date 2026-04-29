import pandas as pd
import numpy as np
import time
import threading
from scapy.all import sniff, IP, TCP, UDP
from collections import defaultdict
from typing import Generator, Dict, List
from src.utils import TIME_FEATURES, STAT_FEATURES

# Cấu hình Timeout cho Flow
FLOW_TIMEOUT = 5.0  # Nếu không có gói tin mới trong 5s -> Flow kết thúc

class FlowSession:
    """Class đại diện cho một luồng mạng (Network Flow) đang hoạt động"""
    def __init__(self, key):
        self.key = key
        self.start_time = time.time()
        self.last_seen = time.time()
        self.packets = []
        
        # Thống kê cơ bản
        self.fwd_pkts = 0
        self.bwd_pkts = 0
        self.fwd_len_tot = 0
        self.bwd_len_tot = 0
        self.fwd_len_list = []
        self.bwd_len_list = []
        self.iat_list = []
        self.flags = {'SYN': 0, 'ACK': 0, 'URG': 0, 'FIN': 0}

    def add_packet(self, packet, direction):
        """
        Thêm gói tin vào Flow
        direction: 'fwd' (xuôi) hoặc 'bwd' (ngược)
        """
        current_time = packet.time
        
        # Tính Inter-arrival Time (IAT)
        if self.last_seen:
            iat = float(current_time) - self.last_seen
            if iat > 0:
                self.iat_list.append(iat * 1_000_000) # Convert to microseconds
        
        self.last_seen = float(current_time)
        self.packets.append(packet)
        
        # Lấy độ dài Payload (Data)
        payload_len = len(packet[IP].payload)
        
        if direction == 'fwd':
            self.fwd_pkts += 1
            self.fwd_len_tot += payload_len
            self.fwd_len_list.append(payload_len)
        else:
            self.bwd_pkts += 1
            self.bwd_len_tot += payload_len
            self.bwd_len_list.append(payload_len)
            
        # Đếm cờ TCP (Flags)
        if packet.haslayer(TCP):
            flags = packet[TCP].flags
            if 'S' in flags: self.flags['SYN'] += 1
            if 'A' in flags: self.flags['ACK'] += 1
            if 'U' in flags: self.flags['URG'] += 1
            if 'F' in flags: self.flags['FIN'] += 1

    def to_features(self) -> Dict:
        """Chuyển đổi dữ liệu thô thành Features cho Model"""
        duration = self.last_seen - self.start_time
        if duration == 0: duration = 1e-6 # Tránh chia cho 0
        
        # Helper tính thống kê
        def get_stat(lst):
            if not lst: return 0, 0, 0, 0
            return np.mean(lst), np.std(lst), np.max(lst), np.min(lst)

        iat_mean, iat_std, iat_max, iat_min = get_stat(self.iat_list)
        fwd_mean, _, fwd_max, _ = get_stat(self.fwd_len_list)
        bwd_mean, _, bwd_max, _ = get_stat(self.bwd_len_list)
        
        # Tính Flow Bytes/s và Flow Packets/s
        tot_len = self.fwd_len_tot + self.bwd_len_tot
        tot_pkts = self.fwd_pkts + self.bwd_pkts
        
        features = {
            # === TIME FEATURES ===
            'Flow Duration': duration * 1_000_000, # Microseconds
            'Flow IAT Mean': iat_mean,
            'Flow IAT Std': iat_std,
            'Flow IAT Max': iat_max,
            'Flow IAT Min': iat_min,
            'Fwd IAT Total': sum(self.iat_list), # Giả lập đơn giản
            'Bwd IAT Total': sum(self.iat_list), # Giả lập đơn giản
            'Fwd IAT Mean': iat_mean,
            'Bwd IAT Mean': iat_mean,
            
            # === STAT FEATURES ===
            'Total Fwd Packets': self.fwd_pkts,
            'Total Backward Packets': self.bwd_pkts,
            'Total Length of Fwd Packets': self.fwd_len_tot,
            'Total Length of Bwd Packets': self.bwd_len_tot,
            'Fwd Packet Length Max': fwd_max,
            'Fwd Packet Length Mean': fwd_mean,
            'Bwd Packet Length Max': bwd_max,
            'Bwd Packet Length Mean': bwd_mean,
            'Flow Bytes/s': tot_len / duration,
            'Flow Packets/s': tot_pkts / duration,
            
            # === FLAGS ===
            'SYN Flag Count': self.flags['SYN'],
            'ACK Flag Count': self.flags['ACK'],
            'URG Flag Count': self.flags['URG']
        }
        
        # Đảm bảo đủ cột, nếu thiếu điền 0
        full_features = {}
        all_cols = TIME_FEATURES + STAT_FEATURES
        for col in all_cols:
            full_features[col] = features.get(col, 0.0)
            
        return full_features

class RealNetworkLogCollector:
    def __init__(self, interface: str):
        self.interface = interface
        self.active_flows = {}  # Key -> FlowSession
        self.finished_flows = [] # Queue chứa flows đã hoàn thành
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.sniff_thread = None

    def _get_flow_key(self, pkt):
        """Tạo Key định danh cho Flow (5-tuple)"""
        if IP not in pkt: return None
        
        src = pkt[IP].src
        dst = pkt[IP].dst
        proto = pkt[IP].proto
        sport = 0
        dport = 0
        
        if TCP in pkt:
            sport = pkt[TCP].sport
            dport = pkt[TCP].dport
        elif UDP in pkt:
            sport = pkt[UDP].sport
            dport = pkt[UDP].dport
            
        # Key 2 chiều (để gom cả gửi và nhận vào 1 flow)
        key_fwd = (src, dst, sport, dport, proto)
        key_bwd = (dst, src, dport, sport, proto)
        
        return key_fwd, key_bwd

    def _packet_callback(self, pkt):
        """Hàm xử lý từng gói tin bắt được"""
        keys = self._get_flow_key(pkt)
        if not keys: return
        
        key_fwd, key_bwd = keys
        
        with self.lock:
            current_time = time.time()
            
            # Kiểm tra xem gói tin thuộc Flow nào
            if key_fwd in self.active_flows:
                self.active_flows[key_fwd].add_packet(pkt, direction='fwd')
            elif key_bwd in self.active_flows:
                self.active_flows[key_bwd].add_packet(pkt, direction='bwd')
            else:
                # Tạo Flow mới
                new_flow = FlowSession(key_fwd)
                new_flow.add_packet(pkt, direction='fwd')
                self.active_flows[key_fwd] = new_flow
            
            # --- CƠ CHẾ TIMEOUT ---
            # Quét và đóng các flow đã hết hạn (Inactive Timeout)
            keys_to_remove = []
            for k, flow in self.active_flows.items():
                if (current_time - flow.last_seen) > FLOW_TIMEOUT:
                    # Flow kết thúc -> Đẩy vào hàng đợi kết quả
                    self.finished_flows.append(flow.to_features())
                    keys_to_remove.append(k)
            
            # Xóa flow cũ khỏi bộ nhớ
            for k in keys_to_remove:
                del self.active_flows[k]

    def _sniff_loop(self):
        """Vòng lặp bắt gói tin chạy ngầm"""
        print(f"[*] Bắt đầu hứng packets trên interface: {self.interface}")
        try:
            sniff(iface=self.interface, prn=self._packet_callback, store=False, stop_filter=lambda x: self.stop_event.is_set())
        except Exception as e:
            print(f"[!] Lỗi sniffing: {e}")

    def start(self):
        """Khởi động luồng bắt gói tin"""
        self.stop_event.clear()
        self.sniff_thread = threading.Thread(target=self._sniff_loop, daemon=True)
        self.sniff_thread.start()

    def stop(self):
        """Dừng bắt gói tin"""
        print("[*] Đang dừng hệ thống thu thập...")
        self.stop_event.set()
        if self.sniff_thread:
            self.sniff_thread.join(timeout=2)

    def get_new_flows(self) -> pd.DataFrame:
        """
        Lấy các flows mới đã hoàn thành xử lý
        Return: DataFrame chứa features
        """
        with self.lock:
            if not self.finished_flows:
                return pd.DataFrame()
            
            # Chuyển list dict thành DataFrame
            df = pd.DataFrame(self.finished_flows)
            
            # Xóa buffer sau khi đã lấy
            self.finished_flows = []
            
            return df

# ==================== DEMO CHẠY THỬ ====================
if __name__ == "__main__":
    # Thay 'Wi-Fi' bằng tên Interface thật của bạn (dùng ipconfig để xem)
    # Ví dụ Windows: "Wi-Fi" hoặc "Ethernet"
    # Ví dụ Linux: "eth0" hoặc "wlan0"
    INTERFACE_NAME = "Wi-Fi" 
    
    collector = RealNetworkLogCollector(interface=INTERFACE_NAME)
    collector.start()
    
    print(">>> Đang chạy thu thập logs. Nhấn Ctrl+C để dừng.")
    
    try:
        while True:
            time.sleep(2) # Mỗi 2 giây kiểm tra 1 lần
            
            # Lấy các flows mới
            df_flows = collector.get_new_flows()
            
            if not df_flows.empty:
                print(f"\n[+] Phát hiện {len(df_flows)} flows mới!")
                print(df_flows[['Flow Duration', 'Total Fwd Packets', 'Flow Bytes/s']].head())
                
                # Ở đây bạn sẽ gọi: model.predict(df_flows)
            else:
                print(".", end="", flush=True)
                
    except KeyboardInterrupt:
        collector.stop()
        print("\nBye bye!")