import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import MODEL_PATH, SCALER_TIME_PATH, SCALER_STAT_PATH, LABEL_ENCODER_PATH
from utils import TIME_FEATURES, STAT_FEATURES, LABEL_COLUMN

st.set_page_config(page_title="IDS Dashboard", layout="wide")
CHUNK_SIZE = 10000  
SAMPLE_SIZE = 100  

@st.cache_resource
def load_resources():
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        scaler_time = joblib.load(SCALER_TIME_PATH)
        scaler_stat = joblib.load(SCALER_STAT_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        return model, scaler_time, scaler_stat, label_encoder
    except Exception as e:
        st.error(f"Không thể tải file hệ thống. Bạn đã chạy train chưa? Lỗi: {e}")
        return None

resources = load_resources()

if resources:
    model, scaler_time, scaler_stat, label_encoder = resources
    
    BENIGN_LABEL_NAME = "BENIGN"
    label_classes = label_encoder.classes_
    for label in label_classes:
        if "BENIGN" in label.upper():
            BENIGN_LABEL_NAME = label
            break
    
    stats = {label: 0 for label in label_classes} 
    
    st.title("Intrusion Detection System (IDS) - Hybrid Model")
    
    st.markdown("""
    **Expected CSV Format:** The uploaded CSV should contain network traffic features including:
    - Time features: Flow Duration, Flow IAT Mean, Flow IAT Std, etc.
    - Statistical features: Total Fwd Packets, Total Backward Packets, etc.
    - Label column: Contains attack types (BENIGN, DDoS, PortScan, etc.)
    
    **Note:** The model was trained on CIC-IDS2017 dataset format.
    """)
    
    st.sidebar.header("Configuration")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file for testing (Max 1GB)", type="csv")
    
    sample_size = st.sidebar.slider("Sample Size", min_value=10, max_value=500, value=100, step=10)
    processing_speed = st.sidebar.selectbox("Processing Speed", 
                                          options=["Fast", "Normal", "Slow"], 
                                          index=1,
                                          help="Fast: 0.05s delay, Normal: 0.1s delay, Slow: 0.2s delay")
    
    speed_delays = {"Fast": 0.05, "Normal": 0.1, "Slow": 0.2}
    delay = speed_delays[processing_speed]
    SAMPLE_SIZE = sample_size
    
    if uploaded_file is not None:
        st.info(f"Processing file '{uploaded_file.name}'. Reading {SAMPLE_SIZE} random samples...")
        tab1, tab2, tab3 = st.tabs(["Real-time Monitoring", "Detailed Results", "Statistics"])
        
        logs = []
        results_data = []
        
        try:
            df_reader = pd.read_csv(uploaded_file, chunksize=CHUNK_SIZE, encoding='cp1252', 
                                   on_bad_lines='skip', low_memory=False)
            first_chunk = next(df_reader)
            first_chunk.columns = first_chunk.columns.str.strip()
            
            required_columns = TIME_FEATURES + STAT_FEATURES + [LABEL_COLUMN]
            missing_columns = [col for col in required_columns if col not in first_chunk.columns]
            
            if missing_columns:
                st.error(f"Missing required columns in CSV: {missing_columns}")
                st.stop()
            
            df_sample = first_chunk if len(first_chunk) < SAMPLE_SIZE else first_chunk.sample(SAMPLE_SIZE)
            st.success(f"Loaded {len(df_sample)} random samples from the first {CHUNK_SIZE} rows for analysis.")

            progress_bar = st.progress(0)
            status_text = st.empty()
            processed_count = 0
            
            for idx, row in df_sample.iterrows():
                processed_count += 1
                status_text.text(f"Processing sample {processed_count}/{len(df_sample)}")
                progress_bar.progress(processed_count / len(df_sample))
                
                row_df = pd.DataFrame([row], columns=df_sample.columns)
                row_df.columns = row_df.columns.str.strip()
                row_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                row_df.fillna(0, inplace=True)

                try:
                    available_time_features = [col for col in TIME_FEATURES if col in row_df.columns]
                    available_stat_features = [col for col in STAT_FEATURES if col in row_df.columns]
                    
                    if not available_time_features or not available_stat_features:
                        logs.append(f"[{time.strftime('%H:%M:%S')}] | ERROR: Missing required features")
                        continue
                    
                    X_time_raw = row_df[available_time_features].values
                    X_stat_raw = row_df[available_stat_features].values
                    
                    try:
                        X_time_scaled = scaler_time.transform(X_time_raw)
                        X_stat_scaled = scaler_stat.transform(X_stat_raw)
                    except ValueError as e:
                        logs.append(f"[{time.strftime('%H:%M:%S')}] | ERROR: Feature mismatch - {str(e)}")
                        continue
                    
                    X_time_scaled = X_time_scaled.reshape(1, 1, X_time_scaled.shape[1])
                    X_stat_scaled = X_stat_scaled.reshape(1, -1)
                    
                except Exception as e:
                    logs.insert(0, f"[{time.strftime('%H:%M:%S')}] | PROCESSING ERROR ROW {idx}: {e}")
                    continue

                try:
                    pred_probs = model.predict([X_time_scaled, X_stat_scaled], verbose=0)
                    pred_id = np.argmax(pred_probs)
                    pred_label = label_encoder.classes_[pred_id]
                    confidence = float(pred_probs[0, pred_id]) * 100
                    all_confidences = {label_encoder.classes_[i]: float(pred_probs[0, i]) * 100 
                                     for i in range(len(label_encoder.classes_))}
                except Exception as e:
                    logs.insert(0, f"[{time.strftime('%H:%M:%S')}] | PREDICTION ERROR: {e}")
                    continue
                
                status = pred_label if pred_label == BENIGN_LABEL_NAME else f"ALERT: {pred_label}"
                stats[pred_label] += 1
                
                true_label = 'N/A'
                for col in df_sample.columns:
                    if col.strip().lower() == 'label':
                        true_label = row.get(col, 'N/A')
                        if isinstance(true_label, str): 
                            true_label = true_label.strip()
                            # Fix corrupted Web Attack labels in CIC-IDS2017
                            import re
                            true_label = re.sub(r'Web Attack .*? XSS', 'Web Attack – XSS', true_label)
                        break

                result_row = {
                    'Timestamp': time.strftime('%H:%M:%S'),
                    'Row_Index': idx,
                    'Prediction': pred_label,
                    'Confidence': f"{confidence:.1f}%",
                    'True_Label': true_label,
                    'Status': 'Normal' if pred_label == BENIGN_LABEL_NAME else 'Attack',
                    'All_Confidences': all_confidences
                }
                
                for i, feature in enumerate(available_time_features):
                    result_row[f'Time_{feature}'] = float(X_time_raw[0][i])
                for i, feature in enumerate(available_stat_features):
                    result_row[f'Stat_{feature}'] = float(X_stat_raw[0][i])
                    
                results_data.append(result_row)
                logs.insert(0, f"[{time.strftime('%H:%M:%S')}] | Confidence: {confidence:.1f}% | Prediction: {status} | Actual: {true_label}")
                time.sleep(delay)

            progress_bar.empty()
            status_text.empty()
            
            with tab1:
                st.subheader("Real-time Monitoring Log")
                st.code('\n'.join(logs[:20]), language=None)
            
            with tab2:
                st.subheader("Detailed Prediction Results")
                if results_data:
                    results_df = pd.DataFrame(results_data)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Samples", len(results_df))
                    with col2:
                        st.metric("Detected Attacks", len(results_df[results_df['Status'] == 'Attack']))
                    with col3:
                        acc = (len(results_df[results_df['Prediction'] == results_df['True_Label']]) / len(results_df) * 100) if not results_df.empty else 0
                        st.metric("Accuracy", f"{acc:.1f}%")
                    
                    st.dataframe(results_df[['Timestamp', 'Row_Index', 'Prediction', 'Confidence', 'True_Label', 'Status']], 
                               use_container_width=True)
                    
                    sample_idx = st.selectbox("Select sample to view confidence details:", 
                                            range(len(results_data)), 
                                            format_func=lambda x: f"Sample {x} - {results_data[x]['Prediction']}")
                    
                    if sample_idx is not None:
                        selected_result = results_data[sample_idx]
                        st.subheader(f"Confidence Scores for Sample {sample_idx}")
                        
                        conf_df = pd.DataFrame({
                            'Class': list(selected_result['All_Confidences'].keys()),
                            'Confidence': list(selected_result['All_Confidences'].values())
                        }).sort_values('Confidence', ascending=False)
                        st.bar_chart(conf_df.set_index('Class'))
                        
                        st.subheader("Feature Values Used")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Time Features:**")
                            st.json({k.replace('Time_', ''): v for k, v in selected_result.items() if k.startswith('Time_')})
                        with col2:
                            st.write("**Statistical Features:**")
                            st.json({k.replace('Stat_', ''): v for k, v in selected_result.items() if k.startswith('Stat_')})
            
            with tab3:
                st.subheader("Detection Statistics")
                total_processed = sum(stats.values())
                benign_count = stats.get(BENIGN_LABEL_NAME, 0)
                attack_types = {k: v for k, v in stats.items() if k != BENIGN_LABEL_NAME and v > 0}
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Processed", total_processed)
                with col2:
                    st.metric("Normal Traffic", benign_count)
                with col3:
                    st.metric("Attack Traffic", sum(attack_types.values()))
                with col4:
                    st.metric("Attack Rate", f"{(sum(attack_types.values()) / total_processed * 100 if total_processed > 0 else 0):.1f}%")
                
                if attack_types:
                    st.subheader("Attack Type Distribution")
                    attack_df = pd.DataFrame({
                        'Attack Type': list(attack_types.keys()),
                        'Count': list(attack_types.values()),
                        'Percentage': [f"{v/total_processed*100:.1f}%" for v in attack_types.values()]
                    })
                    st.dataframe(attack_df, use_container_width=True)
                    st.bar_chart(attack_df.set_index('Attack Type')['Count'])
                else:
                    st.info("No attacks detected in the analyzed samples.")

        except Exception as e:
            st.error(f"Lỗi khi đọc file CSV: {e}")
            
    else:
        st.info("Please upload a CSV file to begin analysis.")
