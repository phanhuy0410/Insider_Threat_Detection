#################################################################
# PIPELINE BASELINE: MULTICLASS LIGHTGBM (CHƯA CÂN BẰNG)
#################################################################

import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import gc

warnings.filterwarnings('ignore')

print("="*75)
print("🚀 KHỞI ĐỘNG PIPELINE: BASELINE CHƯA CÂN BẰNG (UNBALANCED)")
print("="*75)

# ==============================================================================
# 1️⃣ ĐỌC DỮ LIỆU & LỌC CỘT
# ==============================================================================
file_path = '/kaggle/input/datasets/phanthanhhoang/cert-r42-session/session_r4.2.parquet'
df = pd.read_parquet(file_path)
df['insider'] = df['insider'].astype(int)

# Sort chuẩn theo User và Thời gian
df = df.sort_values(by=['user', 'starttime']).reset_index(drop=True)

# Xác định danh sách feature (Bỏ qua các cột định danh)
exclude_cols = ['insider', 'starttime', 'endtime', 'sessionid', 'user', 'day', 'week',]

feature_cols = [col for col in df.columns if col not in (exclude_cols)]

#feature_cols = [col for col in df.columns if col not in exclude_cols]

# ==============================================================================
# 2. TIME + USER SPLIT 
# ==============================================================================
def split_user_time_multiclass(df, label_col='insider'):
    print("\n SPLIT USER + TIME")
    # ------------------------------------------------------------------
    # 1. SORT TIME
    # ------------------------------------------------------------------
    df = df.sort_values(by='starttime').reset_index(drop=True)
    n = len(df)
    test_idx = int(n * 0.80)
    df_past = df.iloc[:test_idx].copy()
    df_future = df.iloc[test_idx:].copy()
    print(f"  -> Past: {len(df_past):,} | Future (Test): {len(df_future):,}")
    # ------------------------------------------------------------------
    # 2. LOẠI INSIDER TRONG FUTURE
    # ------------------------------------------------------------------
    known_insiders = set(df_past[df_past[label_col] != 0]['user'])
    df_test = df_future[~df_future['user'].isin(known_insiders)].copy()
    print(f"  -> Loại {len(known_insiders)} insider đã xuất hiện trong train khỏi TEST")
    # ------------------------------------------------------------------
    # 3. SPLIT TRAIN / VAL (TRONG PAST)
    # ------------------------------------------------------------------
    n_past = len(df_past)
    val_idx = int(n_past * 0.80)
    df_train = df_past.iloc[:val_idx].copy()
    df_val   = df_past.iloc[val_idx:].copy()
    # ------------------------------------------------------------------
    # 4. SORT LẠI THEO USER + TIME
    # ------------------------------------------------------------------
    df_train = df_train.sort_values(by=['user', 'starttime']).reset_index(drop=True)
    df_val   = df_val.sort_values(by=['user', 'starttime']).reset_index(drop=True)
    df_test  = df_test.sort_values(by=['user', 'starttime']).reset_index(drop=True)
    # ------------------------------------------------------------------
    # 5. REPORT
    # ------------------------------------------------------------------
    print("\n📊 DATA REPORT")
    print("-" * 50)
    print(f"TRAIN: {len(df_train):,} samples | {df_train['user'].nunique()} users")
    print(df_train[label_col].value_counts())
    print(f"\nVAL: {len(df_val):,} samples | {df_val['user'].nunique()} users")
    print(df_val[label_col].value_counts())
    print(f"\nTEST: {len(df_test):,} samples | {df_test['user'].nunique()} users")
    print(df_test[label_col].value_counts())
    print("-" * 50)
    return df_train, df_val, df_test
df_train, df_val, df_test = split_user_time_multiclass(df, label_col='insider')

# ==============================================================================
# 2️⃣ XÓA CỘT ZERO VARIANCE (DỌN RÁC)
# ==============================================================================
print("\n Đang dọn dẹp các cột Zero Variance...")

# Tìm các cột không thay đổi giá trị trên tập Train
dead_cols = [c for c in feature_cols if df_train[c].nunique() <= 1]
print(f" -> Đã tìm thấy {len(dead_cols)} cột variance = 0.")

for d in [df_train, df_val, df_test]:
    d.drop(columns=dead_cols, inplace=True, errors='ignore')

# Cập nhật lại danh sách feature hợp lệ (Vẫn giữ đúng thứ tự cột ban đầu)
feature_cols = [c for c in feature_cols if c not in dead_cols]

# ==============================================================================
# 3️⃣ PHÂN LOẠI & SCALING CÓ CHỌN LỌC
# ==============================================================================
print("\nĐang phân loại và Scale các cột Numeric...")

categorical_cols = [
    'pc', 'start_with', 'end_with', 'ses_start', 'ses_end',
    'role', 'f_unit', 'dept', 'team', 'ITAdmin'
]
# Đảm bảo các cột đang tồn tại trong data
cat_cols_to_keep = [c for c in feature_cols if c in categorical_cols]

# Các cột còn lại sẽ Scale
num_cols_to_scale = [c for c in feature_cols if c not in cat_cols_to_keep]

print(f" -> {len(cat_cols_to_keep)} cột Categorical.")
print(f" -> {len(num_cols_to_scale)} cột Numeric.")

scaler = StandardScaler()

# Chỉ Scale đúng nhóm num_cols_to_scale
df_train[num_cols_to_scale] = scaler.fit_transform(df_train[num_cols_to_scale])
df_val[num_cols_to_scale]   = scaler.transform(df_val[num_cols_to_scale])
df_test[num_cols_to_scale]  = scaler.transform(df_test[num_cols_to_scale])

# ==============================================================================
# 4️⃣ TẠO MẢNG 2D BẰNG SLIDING WINDOW (ÉP PHẲNG)
# ==============================================================================
print("\n[4/4] Cuộn dữ liệu thành mảng 2D cho LightGBM...")

def create_sliding_windows_2D(df_subset, window_size=3):
    df_subset = df_subset.sort_values(by=['user', 'starttime']).reset_index(drop=True)
    X_list, y_list = [], []
    for user, group in tqdm(df_subset.groupby('user'), leave=False):
        group_len = len(group)
        if group_len < window_size: continue
        features = group[feature_cols].values
        labels   = group['insider'].values
        for i in range(group_len - window_size + 1):
            window_features = features[i : i + window_size].flatten() 
            window_label = np.max(labels[i : i + window_size])
            X_list.append(window_features)
            y_list.append(window_label)
    return np.array(X_list), np.array(y_list)

WINDOW_SIZE = 5
X_train, y_train = create_sliding_windows_2D(df_train, WINDOW_SIZE)
X_val, y_val     = create_sliding_windows_2D(df_val, WINDOW_SIZE)
X_test, y_test   = create_sliding_windows_2D(df_test, WINDOW_SIZE)

print("\n" + "-"*60)
print("📦 BÁO CÁO MẢNG 2D (SAU SLIDING WINDOW)")
print("-" * 60)
print(f"Tập TRAIN: {X_train.shape[0]:,} mẫu x {X_train.shape[1]} đặc trưng (Window {WINDOW_SIZE} x {len(feature_cols)} gốc)")
print(f"  > Nhãn: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"Tập VAL: {X_val.shape[0]:,} mẫu")
print(f"Tập TEST: {X_test.shape[0]:,} mẫu")

del df_train, df_val, df_test

# ==============================================================================
# 4️⃣ HUẤN LUYỆN LIGHTGBM (KHÔNG DÙNG TRỌNG SỐ)
# ==============================================================================
print("\n[4/4] Đang huấn luyện Mô hình LightGBM Đa lớp...")

dtrain = lgb.Dataset(X_train, label=y_train)
dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

params = { 
    'objective': 'multiclass', 
    'num_class': 4, 
    'metric': ['multi_logloss', 'multi_error'], 
    'learning_rate': 0.01, 
    'max_depth': 5, 
    'num_leaves': 32, 
    'min_data_in_leaf': 100, 
    'feature_fraction': 0.7, 
    'bagging_fraction': 0.8, 
    'bagging_freq': 5, 
    'random_state': 42, 
    'verbosity': -1 
} 

evals_result = {} 
start_train = time.time() 

model = lgb.train( 
    params, dtrain, 
    num_boost_round=1000, 
    valid_sets=[dtrain, dval], 
    valid_names=['train', 'valid'], 
    callbacks=[
        lgb.early_stopping(100, first_metric_only=True, verbose=False), 
        lgb.record_evaluation(evals_result)
    ] 
)
print(f"✅ Huấn luyện hoàn tất sau {time.time() - start_train:.2f} giây!")

# ==============================================================================
# 🔴 ĐÁNH GIÁ TRÊN TẬP TEST
# ==============================================================================
print("\n" + "="*50)
print("🏆 BÁO CÁO ĐÁNH GIÁ (BASELINE CHƯA CÂN BẰNG)")
print("="*50)

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
# VẼ BIỂU ĐỒ & CONFUSION MATRIX
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5)) 
fig.suptitle('BASELINE: MÔ HÌNH CHƯA CÂN BẰNG DỮ LIỆU', fontsize=16, fontweight='bold', y=1.05)

# Loss
epochs = range(1, len(evals_result['train']['multi_logloss']) + 1)
axes[0].plot(epochs, evals_result['train']['multi_logloss'], label='Train Loss', color='blue')
axes[0].plot(epochs, evals_result['valid']['multi_logloss'], label='Val Loss', color='orange')
axes[0].set_title('LightGBM - Logloss', fontweight='bold')
axes[0].legend(); axes[0].grid(True, linestyle=':', alpha=0.6)

# Accuracy
acc_train = [1.0 - err for err in evals_result['train']['multi_error']]
acc_val = [1.0 - err for err in evals_result['valid']['multi_error']]
axes[1].plot(epochs, acc_train, label='Train Acc', color='blue')
axes[1].plot(epochs, acc_val, label='Val Acc', color='orange')
axes[1].set_title('LightGBM - Accuracy', fontweight='bold')
axes[1].legend(); axes[1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# VẼ BIỂU ĐỒ CONFUSION MATRIX (TÁCH RIÊNG)
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
# Tính toán Confusion Matrix
cm = confusion_matrix(y_test, final_preds)
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# Vẽ Heatmap
sns.heatmap(cmn, annot=True, fmt='.2%', cmap='Blues', xticklabels=classes_names, yticklabels=classes_names)
plt.title('LightGBM - Confusion Matrix', fontweight='bold', pad=15)
plt.xlabel('Predict')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# ==============================================================================
# 📈 TÍNH TOÁN ROC-AUC VÀ VẼ BIỂU ĐỒ
# ==============================================================================
print("\n" + "="*50)
print("📈 TỔNG HỢP CHỈ SỐ AUC (AREA UNDER CURVE)")
print("="*50)

# Binarize nhãn thực tế cho bài toán đa lớp (One-vs-Rest)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
test_aucs = {}

for i, name in enumerate(classes_names):
    # Sử dụng xác suất dự đoán của từng lớp (probs[:, i])
    test_aucs[name] = roc_auc_score(y_test_bin[:, i], probs[:, i])
    print(f"-> Final Test AUC for {name}: {test_aucs[name]:.4f}")

print(classification_report(y_test, final_preds, target_names=classes_names, digits=4, zero_division=0))
final_test_auc_macro = roc_auc_score(y_test_bin, probs, multi_class='ovr', average='macro')
print(f"\n🌟 OVERALL TEST AUC (MACRO): {final_test_auc_macro:.4f}")

# Tính toán Tỉ lệ Dương tính thật (TPR) và Dương tính giả (FPR)
n_classes = y_test_bin.shape[1]
fpr, tpr, roc_auc_dict = dict(), dict(), dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probs[:, i])
    roc_auc_dict[i] = auc(fpr[i], tpr[i])

# Vẽ biểu đồ ROC
plt.figure(figsize=(9, 7))
class_colors = ['dodgerblue', 'crimson', 'forestgreen', 'darkorange']

for i, color in zip(range(n_classes), class_colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2.5,
             label=f'ROC {classes_names[i]} (AUC = {roc_auc_dict[i]:.4f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([-0.05, 1.05]); plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('LightGBM - ROC Curve', fontsize=15, fontweight='bold', pad=15)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()