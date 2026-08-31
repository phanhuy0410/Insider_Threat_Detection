#################################################################
# PIPELINE: SINGLE-STAGE TCN+LSTM (UNBALANCED)
# TÍNH NĂNG: Cửa sổ trượt 3D -> Scaling -> Train Multiclass gốc
#################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import time
import warnings
from tqdm import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.backend import clear_session
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import class_weight

warnings.filterwarnings('ignore')

print("="*75)
print("🚀 KHỞI ĐỘNG PIPELINE: BiLSTM BASELINE (ĐỒNG BỘ TIME-SPLIT & SMART SCALE)")
print("="*75)

# ==============================================================================
# 1️⃣ ĐỌC DỮ LIỆU & LỌC ĐẶC TRƯNG GỐC 
# ==============================================================================
print("\n[1/5] Đang đọc Parquet, làm sạch dữ liệu và phân loại đặc trưng...")
file_path = '/kaggle/input/datasets/phanthanhhoang/cert-r42-session/session_r4.2.parquet' # Cập nhật path nếu cần
df = pd.read_parquet(file_path)
df['insider'] = df['insider'].astype(int)

# --- 🔥 SỬA LỖI TẠI ĐÂY: QUY ĐỔI EPOCH TIME (GIÂY) SANG DATETIME ---
try:
    # Đảm bảo cột là dạng số trước khi convert
    df['starttime'] = pd.to_numeric(df['starttime'], errors='coerce')
    # Chuyển đổi từ số giây (unit='s') sang định dạng chuẩn
    df['starttime'] = pd.to_datetime(df['starttime'], unit='s')
except Exception as e:
    print(f"[!] Cảnh báo khi chuyển đổi thời gian: {e}")

exclude_cols = ['insider', 'starttime', 'endtime', 'sessionid', 'user', 'day', 'week']
drop_cols = exclude_cols

feature_cols = [c for c in df.columns if c not in drop_cols]

# Xử lý các cột dạng chuỗi thành số
orig_cat_cols = df[feature_cols].select_dtypes(include=["object", "string", "category"]).columns.tolist()
for col in orig_cat_cols:
    df[col] = df[col].astype('category').cat.codes

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# Khai báo nhóm cột Phân loại (Categorical/Binary)
cat_binary_cols_manual = [
    'isworkhour', 'isafterhour', 'isweekend', 'isweekendafterhour', 'pc', 'n_logon',
    'ITAdmin', 'role', 'b_unit', 'f_unit', 'dept', 'team', 'start_with', 'end_with'
]
cat_binary_cols = [c for c in cat_binary_cols_manual if c in feature_cols]
numeric_cols = [c for c in feature_cols if c not in cat_binary_cols]

print(f"-> Đã xác định {len(feature_cols)} đặc trưng hợp lệ (Trong đó: {len(numeric_cols)} Liên tục, {len(cat_binary_cols)} Phân loại).")

# ==============================================================================
# 2️⃣ CHIA DỮ LIỆU THEO THỜI GIAN (TIME-BASED SPLIT)
# ==============================================================================
print("\n[2/5] Đang cắt chia dữ liệu theo Trục thời gian (64% - 16% - 20%)...")
df = df.sort_values(by='starttime').reset_index(drop=True)

n_total = len(df)
train_end = int(n_total * 0.64)  
val_end   = int(n_total * 0.80)  

df_train = df.iloc[:train_end].copy()
df_val   = df.iloc[train_end:val_end].copy()
df_test  = df.iloc[val_end:].copy()

# --- BÁO CÁO KIỂM TOÁN (AUDIT REPORT) 1: THỜI GIAN & PHÂN BỔ NHÃN TRƯỚC WINDOW ---
print("\n" + "-"*60)
print("📊 BÁO CÁO KIỂM TOÁN TẬP DỮ LIỆU (TRƯỚC SLIDING WINDOW)")
print("-"*60)
print(f"Tập TRAIN ({len(df_train):,} mẫu):")
print(f"  > Thời gian: {df_train['starttime'].min()}  -->  {df_train['starttime'].max()}")
print(f"  > Phân bổ nhãn: {df_train['insider'].value_counts().to_dict()}")

print(f"\nTập VAL ({len(df_val):,} mẫu):")
print(f"  > Thời gian: {df_val['starttime'].min()}  -->  {df_val['starttime'].max()}")
print(f"  > Phân bổ nhãn: {df_val['insider'].value_counts().to_dict()}")

print(f"\nTập TEST ({len(df_test):,} mẫu):")
print(f"  > Thời gian: {df_test['starttime'].min()}  -->  {df_test['starttime'].max()}")
print(f"  > Phân bổ nhãn: {df_test['insider'].value_counts().to_dict()}")
print("-"*60)

del df; gc.collect()

# ==============================================================================
# TẠO TENSOR 3D BẰNG SLIDING WINDOW (Cho Deep Learning)
# ==============================================================================
print("\n[4/5] Đang cuộn dữ liệu thành Tensor 3D...")

def create_sliding_windows_3D(df_subset, window_size=3):
    df_subset = df_subset.sort_values(by=['user', 'starttime']).reset_index(drop=True)
    X_list, y_list = [], []

    for user, group in tqdm(df_subset.groupby('user'), leave=False):
        group_len = len(group)
        if group_len < window_size: continue
            
        features = group[feature_cols].values
        labels   = group['insider'].values

        for i in range(group_len - window_size + 1):
            window_features = features[i : i + window_size] 
            window_label = np.max(labels[i : i + window_size])
            X_list.append(window_features)
            y_list.append(window_label)
            
    return np.array(X_list), np.array(y_list)

WINDOW_SIZE = 5
X_train, y_train = create_sliding_windows_3D(df_train, WINDOW_SIZE)
X_val, y_val = create_sliding_windows_3D(df_val, WINDOW_SIZE)
X_test, y_test = create_sliding_windows_3D(df_test, WINDOW_SIZE)

# --- BÁO CÁO KIỂM TOÁN (AUDIT REPORT) 2: SHAPE 3D & PHÂN BỔ NHÃN SAU WINDOW ---
print("\n" + "-"*60)
print("📦 BÁO CÁO KIỂM TOÁN TENSOR 3D (SAU SLIDING WINDOW)")
print("-"*60)
print(f"Tập TRAIN Tensor: {X_train.shape}")
print(f"  > Giải thích: {X_train.shape[0]:,} (Mẫu) x {X_train.shape[1]} (Time steps) x {X_train.shape[2]} (Đặc trưng gốc/bước)")
print(f"  > Phân bổ nhãn: {dict(zip(*np.unique(y_train, return_counts=True)))}")

print(f"\nTập VAL Tensor: {X_val.shape}")
print(f"  > Phân bổ nhãn: {dict(zip(*np.unique(y_val, return_counts=True)))}")

print(f"\nTập TEST Tensor: {X_test.shape}")
print(f"  > Phân bổ nhãn: {dict(zip(*np.unique(y_test, return_counts=True)))}")
print("-"*60)

# =====================================================
# 5️⃣ XÂY DỰNG VÀ HUẤN LUYỆN LSTM ĐA LỚP
# =====================================================
print(f"\n[TRAINING] Huấn luyện Baseline TCN+LSTM Multiclass (Train: {len(y_train)} mẫu gốc)...")

T = X_train.shape[1]
F = X_train.shape[2]

model = Sequential([
    # ---------------------------------------------------------
    # KHỐI 1: TCN (Temporal Convolutional Network)
    # Nhiệm vụ: Rút trích đặc trưng cục bộ (Local Feature Extraction)
    # ---------------------------------------------------------
    Conv1D(filters=64, kernel_size=2, padding='causal', dilation_rate=1, activation='relu', input_shape=(T,F)),
    BatchNormalization(),
    Dropout(0.3),
    
    Conv1D(filters=64, kernel_size=2, padding='causal', dilation_rate=2, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    # ---------------------------------------------------------
    # KHỐI 2: LSTM (Long Short-Term Memory)
    # Nhiệm vụ: Học trình tự thời gian và ngữ cảnh nhân quả
    # Lớp Conv1D tự động nhả ra mảng 3D, đưa thẳng vào LSTM được luôn!
    # ---------------------------------------------------------
    LSTM(64, return_sequences=False), # return_sequences=False vì ta chỉ cần kết luận cuối cùng của chuỗi
    Dropout(0.3),

    # ---------------------------------------------------------
    # KHỐI 3: CLASSIFICATION HEAD (Đầu ra Phân loại Đa lớp)
    # ---------------------------------------------------------
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005), 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

start_time = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100, 
    batch_size=256, 
    callbacks=[early_stop],
    verbose=1
)
print(f"✅ Hoàn thành Huấn luyện TCN-LSTM! Thời gian: {time.time() - start_time:.2f} giây.")

# =====================================================
# 6️⃣ INFERENCE VÀ ĐÁNH GIÁ (ARGMAX CƠ BẢN)
# =====================================================
print("\n[INFERENCE] Đang dự đoán trên tập Test...")
#prob_test = model.predict(X_test)

# 1. Rút xác suất tập Val để tính Threshold động
val_probs = model.predict(X_val_scaled)
benign_mask_val = (y_val == 0)
benign_probs_val = val_probs[benign_mask_val]

# Tính FP Budget
T1_THRESH_S1 = np.percentile(benign_probs_val[:, 1], 99.99)  
T1_THRESH_S2 = np.percentile(benign_probs_val[:, 2], 98.5) 
T1_THRESH_S3 = np.percentile(benign_probs_val[:, 3], 99.5)

print("\n" + "="*50)
print("🎯 TÍNH TOÁN NGƯỠNG TỪ TẬP VALIDATION")
print("="*50)
print(f"-> Threshold Scen 1: {T1_THRESH_S1:.4f}")
print(f"-> Threshold Scen 2: {T1_THRESH_S2:.4f}")
print(f"-> Threshold Scen 3: {T1_THRESH_S3:.4f}")

# 2. Áp dụng lên Test
test_probs = model.predict(X_test_scaled, batch_size=1024)
final_preds_test = np.zeros(len(test_probs), dtype=int)

for i in range(len(test_probs)):
    candidates = []
    if test_probs[i, 1] >= T1_THRESH_S1: candidates.append((1, test_probs[i, 1]))
    if test_probs[i, 2] >= T1_THRESH_S2: candidates.append((2, test_probs[i, 2]))
    if test_probs[i, 3] >= T1_THRESH_S3: candidates.append((3, test_probs[i, 3]))
    if candidates:
        final_preds_test[i] = max(candidates, key=lambda item: item[1])[0]

classes_names = ['Benign', 'Scen1', 'Scen2', 'Scen3']

print("\n========== CLASSIFICATION REPORT (TCN + LSTM) ==========")
print(classification_report(y_test, final_preds, digits=4, zero_division=0))

cm = confusion_matrix(y_test, final_preds, normalize="true")
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues",
            xticklabels=["Benign","Scenario 1","Scenario 2","Scenario 3"], 
            yticklabels=["Benign","Scenario 1","Scenario 2","Scenario 3"])
plt.title("Confusion Matrix - TCN+LSTM", fontweight='bold')
plt.xlabel("Predict")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# --- BIỂU ĐỒ LEARNING CURVE ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('HYBRID ARCHITECTURE: TCN + LSTM', fontsize=14, fontweight='bold', y=1.05)

axes[0].plot(history.history['loss'], label='Train Loss', color='blue')
axes[0].plot(history.history['val_loss'], label='Validation Loss', color='darkorange')
axes[0].set_title('TCN+LSTM: Loss', fontweight='bold')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].legend(); axes[0].grid(True, linestyle=':', alpha=0.7)

axes[1].plot(history.history['accuracy'], label='Train Accuracy', color='blue')
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', color='darkorange')
axes[1].set_title('TCN+LSTM: Accuracy', fontweight='bold')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy')
axes[1].legend(); axes[1].grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()