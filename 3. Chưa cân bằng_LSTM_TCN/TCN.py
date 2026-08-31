#################################################################
# PIPELINE: MULTICLASS BASELINE (UNBALANCED DATA)
# TÍNH NĂNG: Sliding Window 3D -> Scaling -> Dễ dàng swap Model -> Full Metrics
#################################################################

import os
import time
import warnings
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, GRU, LSTM, Bidirectional, Conv1D, Flatten, Dropout, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

warnings.filterwarnings('ignore')

# Khởi động đồng hồ bấm giờ toàn hệ thống
GLOBAL_START_TIME = time.time()

# ==============================================================================
# 1️⃣ ĐỌC DỮ LIỆU & LỌC ĐẶC TRƯNG GỐC 
# ==============================================================================
print("⏳ [1/6] Đang đọc và cấu trúc dữ liệu gốc...")
file_path = '/kaggle/input/datasets/phanthanhhoang/cert-r42-session/session_r4.2.parquet'
df = pd.read_parquet(file_path)
df['insider'] = df['insider'].astype(int)

# Sort chuẩn theo User và Thời gian
df = df.sort_values(by=['user', 'starttime']).reset_index(drop=True)

# Xác định danh sách feature (Bỏ qua các cột định danh)
exclude_cols = ['insider', 'starttime', 'endtime', 'sessionid', 'user', 'day', 'week']
feature_cols = [col for col in df.columns if col not in exclude_cols]

# ==============================================================================
# 2️⃣ TIME + USER SPLIT 
# ==============================================================================
def split_user_time_multiclass(df, label_col='insider'):
    print("\n⏳ [2/6] Phân chia tập dữ liệu (Time + User Split)...")
    df = df.sort_values(by='starttime').reset_index(drop=True)
    
    # Chia 80/20 theo thời gian
    test_idx = int(len(df) * 0.80)
    df_past = df.iloc[:test_idx].copy()
    df_future = df.iloc[test_idx:].copy()
    
    # Loại bỏ Insider đã rò rỉ trong Train ra khỏi Test
    known_insiders = set(df_past[df_past[label_col] != 0]['user'])
    df_test = df_future[~df_future['user'].isin(known_insiders)].copy()
    
    # Chia Train/Val từ Past
    val_idx = int(len(df_past) * 0.80)
    df_train = df_past.iloc[:val_idx].copy()
    df_val   = df_past.iloc[val_idx:].copy()
    
    # Sort lại toàn bộ
    df_train = df_train.sort_values(by=['user', 'starttime']).reset_index(drop=True)
    df_val   = df_val.sort_values(by=['user', 'starttime']).reset_index(drop=True)
    df_test  = df_test.sort_values(by=['user', 'starttime']).reset_index(drop=True)
    
    print("-" * 50)
    print(f" -> TRAIN : {len(df_train):,} mẫu")
    print(f" -> VAL   : {len(df_val):,} mẫu")
    print(f" -> TEST  : {len(df_test):,} mẫu")
    print("-" * 50)
    return df_train, df_val, df_test

df_train, df_val, df_test = split_user_time_multiclass(df, label_col='insider')

# ==============================================================================
# 3️⃣ XÓA ZERO VARIANCE & SCALING
# ==============================================================================
print("\n⏳ [3/6] Xử lý đặc trưng & Chuẩn hóa (Scaling)...")

# Dọn cột Zero Variance
dead_cols = [c for c in feature_cols if df_train[c].nunique() <= 1]
for d in [df_train, df_val, df_test]:
    d.drop(columns=dead_cols, inplace=True, errors='ignore')
feature_cols = [c for c in feature_cols if c not in dead_cols]

# Chuẩn hóa (Trừ các cột Categorical)
categorical_cols = ['pc', 'start_with', 'end_with', 'ses_start', 'ses_end', 'role', 'f_unit', 'dept', 'team', 'ITAdmin']
cat_cols_to_keep = [c for c in feature_cols if c in categorical_cols]
num_cols_to_scale = [c for c in feature_cols if c not in cat_cols_to_keep]

scaler = StandardScaler()
df_train[num_cols_to_scale] = scaler.fit_transform(df_train[num_cols_to_scale])
df_val[num_cols_to_scale]   = scaler.transform(df_val[num_cols_to_scale])
df_test[num_cols_to_scale]  = scaler.transform(df_test[num_cols_to_scale])

# ==============================================================================
# 4️⃣ TẠO TENSOR 3D (SLIDING WINDOW)
# ==============================================================================
print("\n⏳ [4/6] Cuộn dữ liệu thành Tensor 3D (Sliding Window)...")

def create_sliding_windows_3D(df_subset, window_size=5):
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

# ==============================================================================
# 5️⃣ KHỞI TẠO VÀ HUẤN LUYỆN MÔ HÌNH (UNBALANCED)
# ==============================================================================
print("\n⏳ [5/6] Khởi tạo và Huấn luyện mô hình...")

# 🔥 BẠN MUỐN ĐỔI 1 TRONG 10 MÔ HÌNH THÌ CHỈ CẦN SỬA ĐOẠN RUỘT NÀY 🔥
def build_my_model(T, F):
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, padding='causal', dilation_rate=1, activation='relu', input_shape=(T,F)), BatchNormalization(), Dropout(0.3),
        Conv1D(filters=64, kernel_size=2, padding='causal', dilation_rate=2, activation='relu'), BatchNormalization(), Dropout(0.3),
        tf.keras.layers.Flatten(),
        Dense(32, activation='relu'), BatchNormalization(),
        Dense(4, activation='softmax') 
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Gọi hàm tạo model
model = build_my_model(T=X_train.shape[1], F=X_train.shape[2])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# BẮT ĐẦU TRAIN (Không truyền class_weight)
train_start = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100, 
    batch_size=256,
    callbacks=[early_stop],
    verbose=1
)
print(f"✅ Hoàn thành Huấn luyện! Thời gian train: {(time.time() - train_start)/60:.2f} phút.")

# ==============================================================================
# 6️⃣ INFERENCE & BÁO CÁO TOÀN DIỆN (CLASSIFICATION, CM, AUC, ROC)
# ==============================================================================
from sklearn.metrics import roc_auc_score # Đảm bảo import hàm tính AUC

print("\n⏳ [6/6] Đang chấm điểm và xuất biểu đồ trên tập Test...")
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

# ---------------------------------------------------------
# 🏆 TÍNH TOÁN VÀ IN BÁO CÁO (MACRO AUC + CLASSIFICATION REPORT)
# ---------------------------------------------------------
# Chuyển đổi nhãn Test thành dạng One-Hot để tính AUC Đa lớp
y_test_ohe = tf.keras.utils.to_categorical(y_test, num_classes=4)

# Tính Macro AUC (OVR - One vs Rest)
macro_auc = roc_auc_score(y_test_ohe, prob_test, multi_class='ovr', average='macro')

print("\n" + "="*50)
print(f"🏆 BÁO CÁO PHÂN LOẠI (UNBALANCED)")
print("="*50)
print(f"🔥 MACRO AUC TỔNG THỂ: {macro_auc:.4f} 🔥\n") # In Macro AUC ở đây
print(classification_report(y_test, preds_main, target_names=classes_names, digits=4, zero_division=0))

# TỔNG KẾT THỜI GIAN
total_time = time.time() - GLOBAL_START_TIME
print(f"⏱️ TỔNG THỜI GIAN CHẠY PIPELINE: {total_time/60:.2f} phút.")

# ---------------------------------------------------------
# 🎨 VẼ BIỂU ĐỒ 1: CONFUSION MATRIX (CHỈ HIỂN THỊ %)
# ---------------------------------------------------------
cm = confusion_matrix(y_test, preds_main)
plt.figure(figsize=(8, 6))
row_sums = cm.sum(axis=1)[:, np.newaxis]
cm_percentages = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums!=0)

annot_labels = np.empty_like(cm).astype(str)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annot_labels[i, j] = f"{cm_percentages[i, j]:.2%}"

sns.heatmap(cm_percentages, annot=annot_labels, fmt="", cmap="Blues", 
            xticklabels=classes_names, yticklabels=classes_names)
plt.title("Confusion Matrix - TCN", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 🎨 VẼ BIỂU ĐỒ 2: LEARNING CURVES (LOSS & ACCURACY)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Train Loss', color='blue', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
axes[0].set_title('Loss Curve', fontweight='bold')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].legend(); axes[0].grid(True, linestyle=':', alpha=0.7)

axes[1].plot(history.history['accuracy'], label='Train Accuracy', color='blue', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
axes[1].set_title('Accuracy Curve', fontweight='bold')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy')
axes[1].legend(); axes[1].grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 🎨 VẼ BIỂU ĐỒ 3: ROC CURVE (ĐA LỚP - OVR)
# ---------------------------------------------------------
plt.figure(figsize=(9, 7))
colors = ['dodgerblue', 'crimson', 'forestgreen', 'darkorange']

for i in range(4):
    fpr, tpr, _ = roc_curve(y_test_ohe[:, i], prob_test[:, i])
    roc_auc = auc(fpr, tpr)
    if not np.isnan(roc_auc):
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'ROC curve - {classes_names[i]} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
# Bổ sung luôn Macro AUC vào tiêu đề biểu đồ ROC cho trực quan
plt.title(f'Biểu đồ ROC - TCN', fontsize=15, fontweight='bold', pad=15)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()