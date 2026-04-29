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

# --- RISK SCORING CONFIGURATION ---
ATTACK_SEVERITY = {
    'DDoS': 'HIGH',
    'DoS Hulk': 'HIGH',
    'DoS GoldenEye': 'HIGH',
    'Bot': 'HIGH',
    'Web Attack \x12 Brute Force': 'HIGH',
    'Web Attack \x12 Sql Injection': 'HIGH',
    'Web Attack \x12 XSS': 'HIGH',
    'Heartbleed': 'HIGH',
    'Infiltration': 'HIGH',

    'PortScan': 'MEDIUM',
    'FTP-Patator': 'MEDIUM',
    'SSH-Patator': 'MEDIUM',
    
    'DoS slowloris': 'LOW', 
    'DoS Slowhttptest': 'LOW'
}

def calculate_risk_score_and_level(pred_label, confidence):
    """
    Calculate risk level and score based on attack type and confidence
    Following alert_system.py logic
    """
    if pred_label == 'BENIGN' or 'BENIGN' in str(pred_label).upper():
        return 0, "LOW", "green"
    
    normalized_label = str(pred_label).strip().replace('\x12', '–').replace('\x0b', '')
    
    severity = ATTACK_SEVERITY.get(normalized_label, None)
    
    if severity is None:
        for attack_type, sev in ATTACK_SEVERITY.items():
            if attack_type.lower() in normalized_label.lower() or normalized_label.lower() in attack_type.lower():
                severity = sev
                break
    
    if severity is None:
        severity = 'LOW'
    
    score = confidence / 100.0
    
    if severity == 'HIGH' and score > 0.7:
        risk_level = "CRITICAL"
        color = "red"
        risk_score = score * 100
    
    elif severity == 'MEDIUM' or score > 0.9:
        risk_level = "HIGH"
        color = "orange"
        risk_score = score * 80
    
    else:
        risk_level = "MEDIUM"
        color = "yellow"
        risk_score = score * 50
    
    return risk_score, risk_level, color


st.set_page_config(page_title="IDS Dashboard", layout="wide")
CHUNK_SIZE = 1000000   
SAMPLE_SIZE = 100

# Initialize session state for caching
if 'cached_results' not in st.session_state:
    st.session_state.cached_results = None
if 'cached_critical_alerts' not in st.session_state:
    st.session_state.cached_critical_alerts = None
if 'cached_stats' not in st.session_state:
    st.session_state.cached_stats = None  

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
    SAMPLE_SIZE = sample_size
    
    if uploaded_file is not None:
        st.info(f"Processing file '{uploaded_file.name}'. Reading {SAMPLE_SIZE} random samples...")
        
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.cached_results is not None and hasattr(st.session_state, 'last_file_key') and st.session_state.last_file_key == file_key:
            results_data = st.session_state.cached_results
            critical_alerts = st.session_state.cached_critical_alerts
            stats = st.session_state.cached_stats
            st.info("Using cached analysis results")
            skip_processing = True
        else:
            skip_processing = False
            results_data = []
            critical_alerts = []
            stats = {label: 0 for label in label_classes}
        
        tab1, tab2, tab3, tab4 = st.tabs(["Alerts & Risks", "Detailed Results", "Statistics", "Risk Analysis"])
        
        logs = []
        
        try:
            if not skip_processing:
                df_reader = pd.read_csv(uploaded_file, chunksize=CHUNK_SIZE, encoding='cp1252', 
                                       on_bad_lines='skip', low_memory=False)
                first_chunk = next(df_reader)
                
                first_chunk.columns = first_chunk.columns.str.strip()
                
                required_columns = TIME_FEATURES + STAT_FEATURES + [LABEL_COLUMN]
                missing_columns = [col for col in required_columns if col not in first_chunk.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns in CSV: {missing_columns}")
                    st.error("Please ensure your CSV file contains all the required features.")
                    st.stop()
                
                if len(first_chunk) < SAMPLE_SIZE:
                    df_sample = first_chunk.copy()
                else:
                    df_sample = first_chunk.sample(SAMPLE_SIZE, random_state=42).copy()
                    
                st.success(f"Loaded {len(df_sample)} samples. Starting analysis...")

                available_time_features = [col for col in TIME_FEATURES if col in df_sample.columns]
                available_stat_features = [col for col in STAT_FEATURES if col in df_sample.columns]
                
                if not available_time_features or not available_stat_features:
                    st.error("Missing required features in CSV file")
                    st.stop()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                info_placeholder = st.empty()
                
                processed_count = 0
                update_interval = max(1, len(df_sample) // 10)
                
                for batch_start in range(0, len(df_sample), 32):
                    batch_end = min(batch_start + 32, len(df_sample))
                    batch_indices = df_sample.index[batch_start:batch_end]
                    
                    for idx in batch_indices:
                        row = df_sample.loc[idx]
                        processed_count += 1
                        
                        if processed_count % update_interval == 0 or processed_count == len(df_sample):
                            progress_bar.progress(processed_count / len(df_sample))
                            status_text.text(f"Processing: {processed_count}/{len(df_sample)} samples")
                        
                        try:
                            X_time_raw = pd.to_numeric(row[available_time_features], errors='coerce').values.reshape(1, -1)
                            X_stat_raw = pd.to_numeric(row[available_stat_features], errors='coerce').values.reshape(1, -1)
                            
                            X_time_raw = np.nan_to_num(X_time_raw, nan=0.0, posinf=0.0, neginf=0.0)
                            X_stat_raw = np.nan_to_num(X_stat_raw, nan=0.0, posinf=0.0, neginf=0.0)
                            
                            X_time_scaled = scaler_time.transform(X_time_raw)
                            X_stat_scaled = scaler_stat.transform(X_stat_raw)
                            
                            X_time_scaled = X_time_scaled.reshape(1, 1, -1)
                            X_stat_scaled = X_stat_scaled.reshape(1, -1)
                            
                        except Exception as e:
                            logs.append(f"[{time.strftime('%H:%M:%S')}] | ERROR: Row {idx} - {str(e)[:50]}")
                            continue

                        try:
                            pred_probs = model.predict([X_time_scaled, X_stat_scaled], verbose=0)
                            pred_id = np.argmax(pred_probs)
                            pred_label = label_encoder.classes_[pred_id]
                            confidence = float(pred_probs[0, pred_id]) * 100
                            all_confidences = {label_encoder.classes_[i]: float(pred_probs[0, i]) * 100 
                                             for i in range(len(label_encoder.classes_))}
                        except Exception as e:
                            logs.append(f"[{time.strftime('%H:%M:%S')}] | PREDICTION ERROR: {str(e)[:50]}")
                            continue
                        
                        status = pred_label
                        if pred_label != BENIGN_LABEL_NAME:
                            status = f"ALERT: {pred_label}"
                        
                        risk_score, risk_level, risk_color = calculate_risk_score_and_level(pred_label, confidence)
                        
                        if risk_level == "CRITICAL":
                            critical_alerts.append({
                                'timestamp': time.strftime('%H:%M:%S'),
                                'attack_type': pred_label,
                                'confidence': confidence,
                                'risk_score': risk_score,
                                'row_index': idx
                            })
                        
                        stats[pred_label] += 1
                        
                        true_label = 'N/A'
                        if LABEL_COLUMN in row.index:
                            true_label = str(row[LABEL_COLUMN]).strip()
                        
                        result_row = {
                            'Timestamp': time.strftime('%H:%M:%S'),
                            'Row_Index': idx,
                            'Prediction': pred_label,
                            'Confidence': f"{confidence:.1f}%",
                            'True_Label': true_label,
                            'Status': 'Normal' if pred_label == BENIGN_LABEL_NAME else 'Attack',
                            'Risk_Score': f"{risk_score:.1f}",
                            'Risk_Level': risk_level,
                            'All_Confidences': all_confidences
                        }
                        
                        for i, feature in enumerate(available_time_features):
                            result_row[f'Time_{feature}'] = float(X_time_raw[0, i])
                        for i, feature in enumerate(available_stat_features):
                            result_row[f'Stat_{feature}'] = float(X_stat_raw[0, i])
                        
                        results_data.append(result_row)
                        msg = f"[{time.strftime('%H:%M:%S')}] | Confidence: {confidence:.1f}% | {status} | Actual: {true_label}"
                        logs.insert(0, msg)

                progress_bar.empty()
                status_text.empty()
                info_placeholder.empty()
                
                st.session_state.cached_results = results_data
                st.session_state.cached_critical_alerts = critical_alerts
                st.session_state.cached_stats = stats
                st.session_state.last_file_key = file_key
            
            st.success(f"Analysis completed: {len(results_data)} samples processed successfully")
            
            if critical_alerts:
                st.error(f"SYSTEM ALERT: {len(critical_alerts)} CRITICAL threats detected!")
                alert_cols = st.columns(len(critical_alerts) if len(critical_alerts) <= 5 else 5)
                for i, alert in enumerate(critical_alerts[:5]):
                    with alert_cols[i % 5]:
                        st.error(f"""
                        **CRITICAL ALERT**
                        Type: {alert['attack_type']}
                        Confidence: {alert['confidence']:.1f}%
                        Risk Score: {alert['risk_score']:.1f}
                        Time: {alert['timestamp']}
                        """)
            
            with tab1:
                st.subheader("Early Warning System - Priority Alerts")
                
                col1, col2, col3, col4 = st.columns(4)
                critical_count = len(critical_alerts)
                high_count = len([r for r in results_data if r['Risk_Level'] == 'HIGH'])
                medium_count = len([r for r in results_data if r['Risk_Level'] == 'MEDIUM'])
                low_count = len([r for r in results_data if r['Risk_Level'] == 'LOW'])
                
                with col1:
                    st.error(f"**CRITICAL**\n{critical_count}")
                with col2:
                    st.warning(f"**HIGH**\n{high_count}")
                with col3:
                    st.info(f"**MEDIUM**\n{medium_count}")
                with col4:
                    st.success(f"**LOW**\n{low_count}")
                
                st.divider()
                
                if results_data:
                    results_df = pd.DataFrame(results_data)
                    
                    risk_filter = st.multiselect(
                        "Filter by Risk Level:",
                        options=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'],
                        default=['CRITICAL', 'HIGH']
                    )
                    
                    filtered_df = results_df[results_df['Risk_Level'].isin(risk_filter)]
                    
                    if not filtered_df.empty:
                        st.dataframe(
                            filtered_df[['Timestamp', 'Prediction', 'Confidence', 'Risk_Level', 'Risk_Score', 'True_Label']],
                            use_container_width=True
                        )
                    else:
                        st.info("No alerts with selected risk levels")
                
                if critical_alerts:
                    st.subheader("Critical Threats Timeline")
                    alert_df = pd.DataFrame(critical_alerts)
                    st.bar_chart(alert_df.groupby('attack_type').size())
            
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

            with tab4:
                st.subheader("Risk Analysis and Scoring")
                
                if results_data:
                    results_df = pd.DataFrame(results_data)
                    
                    # Convert Risk_Score to float for analysis
                    results_df['Risk_Score_float'] = results_df['Risk_Score'].astype(float)
                    
                    # Risk score distribution
                    st.subheader("Risk Score Distribution")
                    risk_scores = results_df['Risk_Score_float']
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Avg Risk Score", f"{risk_scores.mean():.1f}")
                    with col2:
                        st.metric("Max Risk Score", f"{risk_scores.max():.1f}")
                    with col3:
                        st.metric("Min Risk Score", f"{risk_scores.min():.1f}")
                    with col4:
                        st.metric("Median Risk Score", f"{risk_scores.median():.1f}")
                    with col5:
                        st.metric("Std Dev", f"{risk_scores.std():.1f}")
                    
                    # Risk distribution chart
                    st.bar_chart(results_df['Risk_Level'].value_counts())
                    
                    # Attack severity heatmap
                    st.subheader("Attack Type Risk Analysis")
                    attack_risk = results_df.groupby('Prediction').agg({
                        'Risk_Score_float': ['mean', 'max', 'count']
                    }).round(1)
                    attack_risk.columns = ['Avg Risk', 'Max Risk', 'Count']
                    st.dataframe(attack_risk.sort_values('Avg Risk', ascending=False))
                    
                    # Risk timeline
                    st.subheader("Risk Score Timeline")
                    st.line_chart(results_df.set_index('Timestamp')['Risk_Score_float'])
                    
                    # DEBUG: Show all unique predicted labels
                    st.subheader("DEBUG: Predicted Attack Types")
                    unique_predictions = results_df['Prediction'].unique()
                    st.write("**Unique predicted labels:**")
                    for pred in unique_predictions:
                        count = len(results_df[results_df['Prediction'] == pred])
                        severity = ATTACK_SEVERITY.get(pred, 'UNKNOWN')
                        st.write(f"- `{pred}` (Count: {count}, Severity: {severity})")

        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            
    else:
        st.info("Please upload a CSV file to begin analysis.")
