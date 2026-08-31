###################
#  LSTM UNBALANCE
##################
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
import warnings
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import class_weight

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 1. INIT & TIME CONVERT
# ------------------------------------------------------------------------------
print("\n[1/4] Đang đọc file Parquet và chuẩn bị không gian...")
file_path = '/kaggle/input/datasets/phanthanhhoang/cert-r42-session/session_r4.2.parquet'
df = pd.read_parquet(file_path)
df['insider'] = df['insider'].astype(int)
df = df.sort_values(by=['user', 'starttime']).reset_index(drop=True)

exclude_cols = ['insider', 'starttime', 'endtime', 'sessionid', 'user', 'day', 'week']
drop_cols = exclude_cols

df['starttime'] = pd.to_numeric(df['starttime'], errors='coerce')
df['starttime'] = pd.to_datetime(df['starttime'], unit='s')

# ------------------------------------------------------------------------------
# 2. HANDLE CATEGORICAL (ONE-HOT CHO DEEP LEARNING)
# ------------------------------------------------------------------------------
feature_cols = [c for c in df.columns if c not in drop_cols]
cat_cols = df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# ------------------------------------------------------------------------------
# 3. DEFINE FEATURE TYPES
# ------------------------------------------------------------------------------
keep_org_cols = [
    'isworkhour', 'isafterhour', 'isweekend', 'isweekendafterhour',
    'pc', 'n_logon', 'n_days', 'n_concurrent_sessions',
    'start_with', 'end_with', 'ses_start', 'ses_end',
    'role', 'b_unit', 'f_unit', 'dept', 'team', 'ITAdmin',
    'O', 'C', 'E', 'A', 'N'
]
cat_binary_cols = [c for c in keep_org_cols if c in df.columns]

# Khởi tạo cột Numeric (Bỏ các cột nhị phân/hạng mục ra)
numeric_cols = [c for c in df.columns if c not in (cat_binary_cols + drop_cols)]

# ------------------------------------------------------------------------------
# 4. TIME SPLIT (CHIA TRƯỚC KHI ONE-HOT ĐỂ CHỐNG LEAKAGE)
# ------------------------------------------------------------------------------
print("\n[3/5] TIME SPLIT & ONE-HOT ENCODING (LEAKAGE-FREE)...")

df = df.sort_values(by='starttime').reset_index(drop=True)

n = len(df)
train_end = int(n * 0.64)
val_end = int(n * 0.80)

df_train = df.iloc[:train_end].copy()
df_val   = df.iloc[train_end:val_end].copy()
df_test  = df.iloc[val_end:].copy()

# --- ONE-HOT ĐỘC LẬP TỪNG TẬP ---
cat_cols = df_train.select_dtypes(include=["object", "category"]).columns.tolist()

df_train = pd.get_dummies(df_train, columns=cat_cols, drop_first=True)
df_val   = pd.get_dummies(df_val, columns=cat_cols, drop_first=True)
df_test  = pd.get_dummies(df_test, columns=cat_cols, drop_first=True)

# 🔥 BÍ KÍP ALIGN: Khớp cột Val và Test theo đúng form của Train (thiếu thì điền 0)
df_train, df_val = df_train.align(df_val, join='left', axis=1, fill_value=0)
df_train, df_test = df_train.align(df_test, join='left', axis=1, fill_value=0)

# Cứu hộ dọn dẹp các giá trị lỗi có thể sinh ra
for d in [df_train, df_val, df_test]:
    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    d.fillna(0, inplace=True)

# ------------------------------------------------------------------------------
# 5. PREPROCESSING SÂU: LỌC RÁC, TRỊ SKEW & SCALING
# ------------------------------------------------------------------------------
print("\n[4/5] PREPROCESSING...")

# --- 5.0 TỰ ĐỘNG PHÂN LOẠI CỘT BẰNG CODE CỦA BẠN ---
# Loại bỏ các cột định danh ra khỏi quá trình duyệt
skip_cols = drop_cols + ['starttime', 'endtime', 'user', 'sessionid']
scan_cols = [c for c in df_train.columns if c not in skip_cols]

# Nhận diện cột Binary 
# binary_cols = [c for c in scan_cols if set(df_train[c].dropna().unique()) <= {0, 1, 0.0, 1.0, True, False}]
binary_cols = [c for c in scan_cols if df_train[c].nunique() <= 5]

# Cột Numeric là phần còn lại
numeric_cols = [c for c in scan_cols if c not in binary_cols]

print(f"  -> Nhận diện tự động: {len(binary_cols)} cột Binary | {len(numeric_cols)} cột Numeric")

# --- 5.1 XÓA ZERO VARIANCE ---
valid_numeric = [c for c in numeric_cols if df_train[c].std() > 0]
dead_cols = list(set(numeric_cols) - set(valid_numeric))

for d in [df_train, df_val, df_test]:
    d.drop(columns=dead_cols, inplace=True, errors='ignore')
numeric_cols = valid_numeric

# --- 5.2 XÓA CỘT SPARSE (BẢO VỆ TIỀN TỐ) ---
sparse_threshold = 0.99
potential_sparse = [c for c in numeric_cols if (df_train[c] == 0).mean() > sparse_threshold]
protected_prefixes = ('n_', 'file_', 'email_', 'http_', 'usb', 'logon')
sparse_to_drop = [c for c in potential_sparse if not c.startswith(protected_prefixes)]

for d in [df_train, df_val, df_test]:
    d.drop(columns=sparse_to_drop, inplace=True, errors='ignore')
numeric_cols = [c for c in numeric_cols if c not in sparse_to_drop]

# --- 5.3 SKEW TREATMENT (NUNIQUE > 10) ---
# 🔥 BÍ KÍP REFINE SKEW CỦA BẠN 
skewed_cols = [c for c in numeric_cols if (df_train[c].nunique() > 10) and (abs(df_train[c].skew()) > 3)]

def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))

for col in skewed_cols:
    df_train[col] = signed_log(df_train[col])
    df_val[col]   = signed_log(df_val[col])
    df_test[col]  = signed_log(df_test[col])

# --- 5.4 STANDARD SCALER CHO TOÀN BỘ NUMERIC ---
if len(numeric_cols) > 0:
    sc = StandardScaler()
    df_train[numeric_cols] = sc.fit_transform(df_train[numeric_cols])
    df_val[numeric_cols]   = sc.transform(df_val[numeric_cols])
    df_test[numeric_cols]  = sc.transform(df_test[numeric_cols])

# ------------------------------------------------------------------------------
# 6. CHỐT HẠ FEATURE
# ------------------------------------------------------------------------------
# Chốt lại danh sách cột theo đúng thứ tự đang nằm trong df_train
feature_cols = [c for c in df_train.columns if c not in skip_cols]

print(f"✅ Đã chốt hạ danh sách đầu vào cho Model: {len(feature_cols)} features hợp lệ.")

# ==============================================================================
# 4️⃣ TẠO TENSOR 3D BẰNG SLIDING WINDOW (Cho Deep Learning)
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
from tensorflow.keras.layers import Masking
print(f"\n[TRAINING] Huấn luyện Baseline LSTM Multiclass (Train: {len(y_train)} mẫu gốc)...")
T = X_train.shape[1]
F = X_train.shape[2]

model = Sequential([
    # Tầng 1: 128 units để hấp thụ trọn vẹn 120 features
    LSTM(128, return_sequences=True, input_shape=(T, F)),
    Dropout(0.3), # Tăng chút Dropout vì mạng lớn hơn
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    #Dense(64, activation='relu'),
    #BatchNormalization(),
    # Tầng Dense 2: Nén thông tin trước khi phân loại
    Dense(32, activation='relu'),
    BatchNormalization(),
    # Đầu ra 4 nodes tương ứng 4 lớp, dùng hàm Softmax
    Dense(4, activation='softmax')
])

# Vì nhãn là số nguyên (0,1,2,3), ta dùng sparse_categorical_crossentropy
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
    epochs=100, # Để 30 epoch, EarlyStopping sẽ tự ngắt nếu hội tụ sớm
    batch_size=256, # Batch size lớn để train nhanh trên dữ liệu khổng lồ
    callbacks=[early_stop],
    verbose=1
)
print(f"✅ Hoàn thành Huấn luyện! Thời gian: {time.time() - start_time:.2f} giây.")

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

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, final_preds, digits=4, zero_division=0))

cm = confusion_matrix(y_test, final_preds, normalize="true")
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues",
            xticklabels=["Benign","Scenario 1","Scenario 2","Scenario 3"], 
            yticklabels=["Benign","Scenario 1","Scenario 2","Scenario 3"])
plt.title("Confusion Matrix - Unbalance LSTM")
plt.xlabel("Predict")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# --- BIỂU ĐỒ LEARNING CURVE ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Train Loss', color='blue')
axes[0].plot(history.history['val_loss'], label='Validation Loss', color='darkorange')
axes[0].set_title('LSTM: Loss', fontweight='bold')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].legend(); axes[0].grid(True, linestyle=':', alpha=0.7)

axes[1].plot(history.history['accuracy'], label='Train Accuracy', color='blue')
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', color='darkorange')
axes[1].set_title('LSTM: Accuracy', fontweight='bold')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy')
axes[1].legend(); axes[1].grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()
