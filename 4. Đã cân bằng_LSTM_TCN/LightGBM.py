import os
import gc
import random
import warnings
import numpy as np
import pandas as pd
import time
import lightgbm as lgb
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)
warnings.filterwarnings('ignore')

file_path = '/kaggle/input/datasets/phanthanhhoang/cert-r42-session/session_r4.2.parquet'
df = pd.read_parquet(file_path)
df['insider'] = df['insider'].astype(int)
df = df.sort_values(by=['user', 'starttime']).reset_index(drop=True)

eps = 1e-5
if 'duration' in df.columns:
    df['duration_quantile'] = pd.qcut(df['duration'], q=[0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1], labels=False, duplicates='drop')
if 'n_allact' in df.columns:
    if 'n_http' in df.columns: df['http_ratio'] = df['n_http'] / (df['n_allact'] + eps)
    if 'n_email' in df.columns: df['email_ratio'] = df['n_email'] / (df['n_allact'] + eps)
    if 'n_file' in df.columns: df['file_ratio'] = df['n_file'] / (df['n_allact'] + eps)
    if 'n_usb' in df.columns: df['usb_ratio'] = df['n_usb'] / (df['n_allact'] + eps)
if 'n_http' in df.columns:
    if 'http_n_leakf' in df.columns: df['http_leakf_ratio'] = df['http_n_leakf'] / (df['n_http'] + eps)
    if 'http_n_jobf' in df.columns: df['http_jobf_ratio'] = df['http_n_jobf'] / (df['n_http'] + eps)
    if 'http_n_hackf' in df.columns: df['http_hackf_ratio'] = df['http_n_hackf'] / (df['n_http'] + eps)

new_features = ['duration_quantile', 'http_ratio', 'email_ratio', 'file_ratio', 'http_leakf_ratio', 'http_jobf_ratio', 'http_hackf_ratio', 'usb_ratio']
valid_new_features = [col for col in new_features if col in df.columns]
df[valid_new_features] = df[valid_new_features].fillna(0)

activity_cols = ['n_http', 'n_email', 'n_file']
split_index = int(len(df) * 0.80)
for col in activity_cols:
    if col in df.columns:
        mean_col = df.groupby('user')[col].transform(lambda x: x.expanding().mean().shift(1))
        std_col = df.groupby('user')[col].transform(lambda x: x.expanding().std().shift(1))
        global_mean = df[col].iloc[:split_index].mean()
        global_std = df[col].iloc[:split_index].std()
        mean_col = mean_col.fillna(global_mean)
        std_col = std_col.fillna(global_std).replace(0, 1.0)
        df[f'{col}_zscore'] = (df[col] - mean_col) / std_col

# ==============================================================================
# CHIA TẬP DỮ LIỆU
# ==============================================================================
def split_data(df, label_col='insider'):
    df = df.sort_values(by='starttime').reset_index(drop=True)
    test_idx = int(len(df) * 0.80)
    df_past = df.iloc[:test_idx].copy()
    df_future = df.iloc[test_idx:].copy()
    
    known_insiders = set(df_past[df_past[label_col] != 0]['user'])
    df_test = df_future[~df_future['user'].isin(known_insiders)].copy()
    
    val_idx = int(len(df_past) * 0.80)
    df_train = df_past.iloc[:val_idx].copy()
    df_val = df_past.iloc[val_idx:].copy()
    
    return (df_train.sort_values(by=['user', 'starttime']).reset_index(drop=True),
            df_val.sort_values(by=['user', 'starttime']).reset_index(drop=True),
            df_test.sort_values(by=['user', 'starttime']).reset_index(drop=True))

df_train, df_val, df_test = split_data(df)

exclude_cols = ['insider', 'starttime', 'endtime', 'sessionid', 'user', 'day', 'week']
feature_cols = [col for col in df_train.columns if col not in exclude_cols]

variance_cols = [c for c in feature_cols if df_train[c].nunique() <= 1]
for d in [df_train, df_val, df_test]:
    d.drop(columns=variance_cols, inplace=True, errors='ignore')
feature_cols = [c for c in feature_cols if c not in variance_cols]

categorical_cols = ['pc', 'start_with', 'end_with', 'role', 'b_unit', 'f_unit', 'dept', 'team', 'ITAdmin']
active_cat_cols = [c for c in feature_cols if c in categorical_cols]
for col in active_cat_cols:
    freq_map = {k: max(v, 0.01) for k, v in df_train[col].value_counts(normalize=True).to_dict().items()}
    df_train[col] = df_train[col].map(freq_map).fillna(0.01)
    df_val[col] = df_val[col].map(freq_map).fillna(0.01)
    df_test[col] = df_test[col].map(freq_map).fillna(0.01)

def create_sliding_windows(df_subset, window_size=5, stride_benign=1, stride_attack=1):
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

window_size = 5
X_train_seq, y_train = create_sliding_windows(df_train, window_size, stride_benign=1, stride_attack=1)
X_val_seq, y_val = create_sliding_windows(df_val, window_size, stride_benign=1, stride_attack=1)
X_test_seq, y_test = create_sliding_windows(df_test, window_size, stride_benign=1, stride_attack=1)

del df_train, df_val, df_test
gc.collect()

n_samples, n_steps, n_features = X_train_seq.shape
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_seq.reshape(-1, n_features)).reshape(-1, n_steps, n_features)
X_val_scaled = scaler.transform(X_val_seq.reshape(-1, n_features)).reshape(-1, n_steps, n_features)
X_test_scaled = scaler.transform(X_test_seq.reshape(-1, n_features)).reshape(-1, n_steps, n_features)

X_train_flat = X_train_scaled.reshape(len(X_train_scaled), n_steps * n_features)
X_val_flat = X_val_scaled.reshape(len(X_val_scaled), n_steps * n_features)
X_test_flat = X_test_scaled.reshape(len(X_test_scaled), n_steps * n_features)

benign_indices = np.where(y_train == 0)[0]
attack_indices = np.where(y_train != 0)[0]
X_benign_flat = X_train_flat[benign_indices]

n_clusters = 100 
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=2048)
cluster_distances = kmeans.fit_transform(X_benign_flat)
target_benign_count = 20000
selected_benign_indices = []

for cluster_id, count in pd.Series(kmeans.labels_).value_counts().items():
    n_draw = int(np.round((count / len(benign_indices)) * target_benign_count))
    if n_draw <= 0: continue
    idx_in_cluster = np.where(kmeans.labels_ == cluster_id)[0]
    dist = cluster_distances[idx_in_cluster, cluster_id]
    sorted_idx = idx_in_cluster[np.argsort(dist)]
    n_cluster = len(sorted_idx)
    near_end = int(n_cluster * 0.30)
    mid_end = int(n_cluster * 0.70)
    near_idx = sorted_idx[:near_end]
    mid_idx = sorted_idx[near_end:mid_end]
    far_idx = sorted_idx[mid_end:int(n_cluster * 0.95)]
    n_near = int(n_draw * 0.50)
    n_mid = int(n_draw * 0.30)
    n_far = n_draw - n_near - n_mid
    selected = []
    if len(near_idx) > 0: selected.extend(near_idx[:min(n_near, len(near_idx))])
    if len(mid_idx) > 0 and n_mid > 0: 
        start = max(0, len(mid_idx)//2 - n_mid//2)
        selected.extend(mid_idx[start:min(len(mid_idx), start + n_mid)])
    if len(far_idx) > 0 and n_far > 0: 
        selected.extend(far_idx[-min(n_far, len(far_idx)):])
        
    selected_benign_indices.extend(selected)

final_benign_indices = benign_indices[np.array(selected_benign_indices)]

if len(final_benign_indices) > target_benign_count:
    final_benign_indices = np.random.choice(final_benign_indices, target_benign_count, replace=False)
elif len(final_benign_indices) < target_benign_count:
    remaining_benign = np.setdiff1d(benign_indices, final_benign_indices)
    shortage = target_benign_count - len(final_benign_indices)
    extra_benign = np.random.choice(remaining_benign, shortage, replace=False)
    final_benign_indices = np.concatenate([final_benign_indices, extra_benign])

X_undersampled_flat = np.concatenate([X_train_flat[final_benign_indices], X_train_flat[attack_indices]])
y_undersampled = np.concatenate([y_train[final_benign_indices], y_train[attack_indices]])

smote = SMOTE(sampling_strategy={1: 10000, 2: 10000, 3: 10000}, random_state=42, k_neighbors=5)
X_balanced_flat, y_balanced = smote.fit_resample(X_undersampled_flat, y_undersampled)

y_test_encoded = np.eye(4)[y_test]

# ==============================================================================
# HUẤN LUYỆN MÔ HÌNH
# ==============================================================================
lgb_model = lgb.LGBMClassifier(
    objective='multiclass', 
    num_class=4, 
    n_estimators=1000, 
    learning_rate=0.01,
    max_depth=8, 
    num_leaves=32, 
    min_child_samples=100, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    subsample_freq=5,
    random_state=42, 
    n_jobs=-1, 
    verbose=-1,
    device_type='gpu',
    max_bin=63,            
    gpu_use_dp=False
)

callbacks = [
    lgb.early_stopping(stopping_rounds=100, verbose=False),
    lgb.log_evaluation(period=50)
]

start_time = time.time()
lgb_model.fit(
    X_balanced_flat, y_balanced,
    eval_set=[(X_balanced_flat, y_balanced), (X_val_flat, y_val)],
    eval_names=['Train', 'Validation'],
    eval_metric=['multi_logloss', 'multi_error'],
    callbacks=callbacks
)
train_time = time.time() - start_time
print(f"Tổng thời gian huấn luyện: {train_time/60:.2f} phút")

# ==============================================================================
# 6. ĐÁNH GIÁ VÀ THRESHOLDING
# ==============================================================================
val_predictions = lgb_model.predict_proba(X_val_flat)
benign_val_preds = val_predictions[y_val == 0]
thresh_s1 = np.percentile(benign_val_preds[:, 1], 99.99)  
thresh_s2 = np.percentile(benign_val_preds[:, 2], 99) 
thresh_s3 = np.percentile(benign_val_preds[:, 3], 99.5)

test_predictions = lgb_model.predict_proba(X_test_flat)
y_pred = np.zeros(len(test_predictions), dtype=int)

for i in range(len(test_predictions)):
    candidates = []
    if test_predictions[i, 1] >= thresh_s1: candidates.append((1, test_predictions[i, 1]))
    if test_predictions[i, 2] >= thresh_s2: candidates.append((2, test_predictions[i, 2]))
    if test_predictions[i, 3] >= thresh_s3: candidates.append((3, test_predictions[i, 3]))
    if candidates:
        y_pred[i] = max(candidates, key=lambda item: item[1])[0]

classes = ['Benign', 'Scen1', 'Scen2', 'Scen3']
print(classification_report(y_test, y_pred, target_names=classes, digits=4, zero_division=0))

# ==============================================================================
# 7. VẼ BIỂU ĐỒ
# ==============================================================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
row_sums = cm.sum(axis=1)[:, np.newaxis]
cm_percentages = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums!=0)

annot_labels = np.empty_like(cm).astype(str)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annot_labels[i, j] = f"{cm_percentages[i, j]:.2%}"

sns.heatmap(cm_percentages, annot=annot_labels, fmt="", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix - LightGBM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

evals_result = lgb_model.evals_result_
train_loss = evals_result['Train']['multi_logloss']
val_loss = evals_result['Validation']['multi_logloss']
train_acc = [1 - x for x in evals_result['Train']['multi_error']]
val_acc = [1 - x for x in evals_result['Validation']['multi_error']]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(train_loss, color='blue', lw=2, label='Train Loss')
ax1.plot(val_loss, color='orange', lw=2, label='Val Loss')
ax1.set_title('LightGBM: Loss Curve')
ax1.set_xlabel('Boosting Rounds')
ax1.set_ylabel('Multi-Logloss')
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.plot(train_acc, color='blue', lw=2, label='Train Accuracy')
ax2.plot(val_acc, color='orange', lw=2, label='Val Accuracy')
ax2.set_title('LightGBM: Accuracy Curve')
ax2.set_xlabel('Boosting Rounds')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 7))
colors = ['dodgerblue', 'crimson', 'forestgreen', 'darkorange']
for i in range(4):
    fpr, tpr, _ = roc_curve(y_test_encoded[:, i], test_predictions[:, i])
    roc_auc_val = auc(fpr, tpr)
    if not np.isnan(roc_auc_val):
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{classes[i]} (AUC = {roc_auc_val:.4f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - LightGBM')
plt.legend(loc="lower right")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()