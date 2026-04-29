"""
Generator dữ liệu logs mẫu để test ứng dụng monitoring
Sẽ được thay thế bằng hệ thống thu thập logs thực tế sau này
"""
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from src.utils import TIME_FEATURES, STAT_FEATURES, LABEL_COLUMN

class MockNetworkFlowGenerator:
    def __init__(self):
        """Khởi tạo generator với các attack types"""
        self.attack_types = [
            'BENIGN',
            'DDoS',
            'DoS Hulk',
            'PortScan',
            'Bot',
            'SSH-Patator',
            'FTP-Patator',
            'Infiltration'
        ]
        
        # Cấu hình dữ liệu cho mỗi loại attack
        self.attack_profiles = {
            'BENIGN': {
                'duration': (100, 5000),
                'packet_count': (10, 500),
                'flow_rate': (1, 100),
                'flag_count': (0, 10)
            },
            'DDoS': {
                'duration': (10, 100),
                'packet_count': (1000, 5000),
                'flow_rate': (500, 2000),
                'flag_count': (100, 500)
            },
            'DoS Hulk': {
                'duration': (50, 500),
                'packet_count': (500, 2000),
                'flow_rate': (200, 1000),
                'flag_count': (50, 200)
            },
            'PortScan': {
                'duration': (500, 5000),
                'packet_count': (50, 500),
                'flow_rate': (5, 50),
                'flag_count': (10, 50)
            },
            'Bot': {
                'duration': (100, 1000),
                'packet_count': (200, 1000),
                'flow_rate': (50, 300),
                'flag_count': (20, 100)
            },
            'SSH-Patator': {
                'duration': (1000, 10000),
                'packet_count': (100, 1000),
                'flow_rate': (10, 100),
                'flag_count': (20, 100)
            },
            'FTP-Patator': {
                'duration': (500, 5000),
                'packet_count': (100, 500),
                'flow_rate': (10, 100),
                'flag_count': (10, 50)
            },
            'Infiltration': {
                'duration': (100, 5000),
                'packet_count': (50, 500),
                'flow_rate': (5, 100),
                'flag_count': (5, 50)
            }
        }
    
    def generate_single_flow(self, attack_type='BENIGN'):
        """
        Tạo một single network flow sample
        
        Args:
            attack_type: Loại attack (mặc định là BENIGN)
            
        Returns:
            dict: Một network flow đầy đủ tất cả features
        """
        profile = self.attack_profiles.get(attack_type, self.attack_profiles['BENIGN'])
        
        # Tạo dữ liệu time features
        duration = random.randint(profile['duration'][0], profile['duration'][1])
        
        flow_data = {
            'Flow Duration': duration,
            'Flow IAT Mean': random.uniform(1, 1000),
            'Flow IAT Std': random.uniform(0, 500),
            'Flow IAT Max': random.uniform(100, 10000),
            'Flow IAT Min': random.uniform(0.1, 100),
            'Fwd IAT Total': random.uniform(0, 10000),
            'Bwd IAT Total': random.uniform(0, 10000),
            'Fwd IAT Mean': random.uniform(0, 1000),
            'Bwd IAT Mean': random.uniform(0, 1000),
        }
        
        # Tạo dữ liệu stat features
        fwd_packets = random.randint(profile['packet_count'][0], profile['packet_count'][1])
        bwd_packets = random.randint(profile['packet_count'][0], profile['packet_count'][1])
        
        flow_data.update({
            'Total Fwd Packets': fwd_packets,
            'Total Backward Packets': bwd_packets,
            'Total Length of Fwd Packets': random.randint(100, 100000),
            'Total Length of Bwd Packets': random.randint(100, 100000),
            'Fwd Packet Length Max': random.randint(100, 10000),
            'Fwd Packet Length Mean': random.randint(50, 5000),
            'Bwd Packet Length Max': random.randint(100, 10000),
            'Bwd Packet Length Mean': random.randint(50, 5000),
            'Flow Bytes/s': random.uniform(0, 10000),
            'Flow Packets/s': random.uniform(0, 1000),
            'SYN Flag Count': random.randint(profile['flag_count'][0], profile['flag_count'][1]),
            'ACK Flag Count': random.randint(profile['flag_count'][0], profile['flag_count'][1]),
            'URG Flag Count': random.randint(0, profile['flag_count'][1] // 2),
            LABEL_COLUMN: attack_type
        })
        
        return flow_data
    
    def generate_batch_flows(self, num_flows=10, attack_distribution=None):
        """
        Tạo một batch network flows
        
        Args:
            num_flows: Số lượng flows cần tạo
            attack_distribution: Dict phân bố loại attacks. 
                                Nếu None, sẽ random hoặc để 80% BENIGN, 20% attack
                                
        Returns:
            DataFrame: Batch flows đầy đủ
        """
        if attack_distribution is None:
            # Mặc định: 80% BENIGN, 20% attack
            attack_distribution = {
                'BENIGN': 0.8,
                'DDoS': 0.05,
                'DoS Hulk': 0.05,
                'PortScan': 0.03,
                'Bot': 0.03,
                'SSH-Patator': 0.02,
                'FTP-Patator': 0.02
            }
        
        flows = []
        for _ in range(num_flows):
            # Chọn attack type dựa trên distribution
            attack_type = np.random.choice(
                list(attack_distribution.keys()),
                p=list(attack_distribution.values())
            )
            flows.append(self.generate_single_flow(attack_type))
        
        return pd.DataFrame(flows)
    
    def generate_stream(self, batch_size=5, interval_seconds=5):
        """
        Generator để stream dữ liệu liên tục (dùng trong ứng dụng monitoring)
        
        Args:
            batch_size: Số flows trong mỗi batch
            interval_seconds: Khoảng thời gian giữa các batch (giây)
            
        Yields:
            DataFrame: Batch flows mới
        """
        while True:
            flows = self.generate_batch_flows(batch_size)
            timestamp = datetime.now()
            flows['timestamp'] = timestamp
            yield flows
            
            # Đợi trước khi sinh batch tiếp theo
            import time
            time.sleep(interval_seconds)


# ===================== TEST ======================
if __name__ == "__main__":
    gen = MockNetworkFlowGenerator()
    
    # Test 1: Tạo single flow
    print("=" * 50)
    print("TEST 1: Tạo single flow")
    print("=" * 50)
    single_flow = gen.generate_single_flow('DDoS')
    print(f"DDoS Flow:\n{pd.DataFrame([single_flow])}\n")
    
    # Test 2: Tạo batch flows
    print("=" * 50)
    print("TEST 2: Tạo batch flows")
    print("=" * 50)
    batch = gen.generate_batch_flows(num_flows=5)
    print(f"Batch shape: {batch.shape}")
    print(f"Batch:\n{batch}\n")
    
    # Test 3: Kiểm tra features
    print("=" * 50)
    print("TEST 3: Kiểm tra TIME_FEATURES")
    print("=" * 50)
    print(f"TIME_FEATURES: {len(batch[TIME_FEATURES])}")
    print(f"STAT_FEATURES: {len(batch[STAT_FEATURES])}")
    print(f"Total: {len(TIME_FEATURES) + len(STAT_FEATURES)}")
