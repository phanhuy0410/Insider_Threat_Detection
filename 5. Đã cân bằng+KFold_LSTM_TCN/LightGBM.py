import os
import gc
import time
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, label_binarize
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
        
        df[f'{col}_zscore'] = (df[col] - mean_col.fillna(global_mean)) / std_col.fillna(global_std).replace(0, 1.0)

def split_dev_test(df, label_col='insider'):
    df = df.sort_values(by='starttime').reset_index(drop=True)
    test_idx = int(len(df) * 0.80)
    df_dev = df.iloc[:test_idx].copy()
    df_test = df.iloc[test_idx:].copy()
    
    known_insiders = set(df_dev[df_dev[label_col] != 0]['user'])
    df_test = df_test[~df_test['user'].isin(known_insiders)].copy()
    
    return df_dev.reset_index(drop=True), df_test.reset_index(drop=True)

df_dev, df_test = split_dev_test(df)

exclude_cols = ['insider', 'starttime', 'endtime', 'sessionid', 'user', 'day', 'week']
feature_cols = [col for col in df_dev.columns if col not in exclude_cols]

variance_cols = [c for c in feature_cols if df_dev[c].nunique() <= 1]
for d in [df_dev, df_test]:
    d.drop(columns=variance_cols, inplace=True, errors='ignore')
feature_cols = [c for c in feature_cols if c not in variance_cols]

categorical_cols = ['pc', 'start_with', 'end_with', 'role', 'b_unit', 'f_unit', 'dept', 'team', 'ITAdmin']
active_cat_cols = [c for c in feature_cols if c in categorical_cols]

for col in active_cat_cols:
    freq_map = {k: max(v, 0.01) for k, v in df_dev[col].value_counts(normalize=True).to_dict().items()}
    df_dev[col] = df_dev[col].map(freq_map).fillna(0.01)
    df_test[col] = df_test[col].map(freq_map).fillna(0.01)

def create_sliding_windows(df_subset, window_size=5, stride_benign=1, stride_attack=1):
    df_subset = df_subset.sort_values(by=['user', 'starttime']).reset_index(drop=True)
    X_seq, y_seq, w_seq = [], [], []
    
    for user, group in tqdm(df_subset.groupby('user'), leave=False):
        if len(group) < window_size: continue
        features = group[feature_cols].values
        labels = group['insider'].values
        weeks = group['week'].values
        
        for i in range(len(group) - window_size + 1):
            window_labels = labels[i : i + window_size]
            max_label = np.max(window_labels)
            stride = stride_attack if max_label > 0 else stride_benign
            
            if i % stride == 0:
                X_seq.append(features[i : i + window_size])
                y_seq.append(max_label)
                w_seq.append(weeks[i + window_size - 1])
                
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int8), np.array(w_seq)

window_size = 5
X_dev_seq, y_dev, w_dev = create_sliding_windows(df_dev, window_size, stride_benign=1, stride_attack=1)
X_test_seq, y_test, _ = create_sliding_windows(df_test, window_size, stride_benign=1, stride_attack=1)
del df_dev, df_test, df
gc.collect()

n_samples, n_steps, n_features = X_dev_seq.shape

time_folds = [
    {"train_weeks": list(range(0, 30)), "val_weeks": list(range(31, 35))},
    {"train_weeks": list(range(0, 35)), "val_weeks": list(range(36, 40))},
    {"train_weeks": list(range(0, 40)), "val_weeks": list(range(41, 45))},
    {"train_weeks": list(range(0, 45)), "val_weeks": list(range(46, 50))},
    {"train_weeks": list(range(0, 50)), "val_weeks": list(range(51, 56))}
]

oof_probs = np.zeros((len(y_dev), 4))
test_probs_accum = np.zeros((len(y_test), 4))
all_evals = []
feature_importances_accum = np.zeros(n_steps * n_features)

for fold_idx, fold_cfg in enumerate(time_folds, 1):
    print(f"\n" + "="*50)
    print(f"🚀 ĐANG CHẠY FOLD {fold_idx}/5")
    print("="*50)

    train_mask = np.isin(w_dev, fold_cfg["train_weeks"])
    val_mask = np.isin(w_dev, fold_cfg["val_weeks"])
    
    X_train = X_dev_seq[train_mask]
    y_train = y_dev[train_mask]
    X_val = X_dev_seq[val_mask]
    y_val = y_dev[val_mask]
    
    # --------------------------------------------------------------------------
    # Chuẩn hóa
    # --------------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(-1, n_steps, n_features)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(-1, n_steps, n_features)
    X_test_scaled = scaler.transform(X_test_seq.reshape(-1, n_features)).reshape(-1, n_steps, n_features)
    
    X_train_flat = X_train_scaled.reshape(len(X_train), n_steps * n_features)
    X_val_flat = X_val_scaled.reshape(len(X_val), n_steps * n_features)
    X_test_flat = X_test_scaled.reshape(len(X_test_seq), n_steps * n_features)

    # --------------------------------------------------------------------------
    # K-Means Undersampling
    # --------------------------------------------------------------------------
    benign_idx = np.where(y_train == 0)[0]
    attack_idx = np.where(y_train != 0)[0]
    
    kmeans = MiniBatchKMeans(n_clusters=100, random_state=42, batch_size=2048)
    distances = kmeans.fit_transform(X_train_flat[benign_idx])
    
    target_benign = 20000
    chosen_benign_idx = []
    
    for cluster_id, count in pd.Series(kmeans.labels_).value_counts().items():
        n_draw = int(np.round((count / len(benign_idx)) * target_benign))
        if n_draw <= 0: continue
        
        idx_in_cluster = np.where(kmeans.labels_ == cluster_id)[0]
        dist = distances[idx_in_cluster, cluster_id]
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
            
        chosen_benign_idx.extend(selected)

    final_benign_idx = benign_idx[np.array(chosen_benign_idx)]
    
    if len(final_benign_idx) > target_benign:
        final_benign_idx = np.random.choice(final_benign_idx, target_benign, replace=False)
    elif len(final_benign_idx) < target_benign:
        remaining_benign = np.setdiff1d(benign_idx, final_benign_idx)
        shortage = target_benign - len(final_benign_idx)
        extra_benign = np.random.choice(remaining_benign, shortage, replace=False)
        final_benign_idx = np.concatenate([final_benign_idx, extra_benign])

    X_undersampled_flat = np.concatenate([X_train_flat[final_benign_idx], X_train_flat[attack_idx]])
    y_undersampled = np.concatenate([y_train[final_benign_idx], y_train[attack_idx]])

    smote = SMOTE(sampling_strategy={1: 10000, 2: 10000, 3: 10000}, random_state=42, k_neighbors=5)
    X_balanced_flat, y_balanced = smote.fit_resample(X_undersampled_flat, y_undersampled)

    # --------------------------------------------------------------------------
    # Huấn luyện LightGBM
    # --------------------------------------------------------------------------
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
    print(f"✅ Tổng thời gian huấn luyện Fold {fold_idx}: {train_time/60:.2f} phút")

    all_evals.append(lgb_model.evals_result_)
    test_probs_accum += lgb_model.predict_proba(X_test_flat) / len(time_folds    
    gc.collect()

# ==============================================================================
# ĐÁNH GIÁ
# ==============================================================================
classes = ['Benign', 'Scen1', 'Scen2', 'Scen3']
benign_oof_probs = oof_probs[y_dev == 0]

thresh_s1 = np.percentile(benign_oof_probs[:, 1], 99.999)
thresh_s2 = np.percentile(benign_oof_probs[:, 2], 99.5)
thresh_s3 = np.percentile(benign_oof_probs[:, 3], 99.995)

# Báo cáo Test
test_preds = np.zeros(len(test_probs_accum), dtype=int) 
for i in range(len(test_probs_accum)):
    candidates = []
    if test_probs_accum[i, 1] >= thresh_s1: candidates.append((1, test_probs_accum[i, 1]))
    if test_probs_accum[i, 2] >= thresh_s2: candidates.append((2, test_probs_accum[i, 2]))
    if test_probs_accum[i, 3] >= thresh_s3: candidates.append((3, test_probs_accum[i, 3]))
    if candidates:
        test_preds[i] = max(candidates, key=lambda item: item[1])[0]

print("\n🚀 BÁO CÁO CUỐI CÙNG TRÊN TẬP TEST ĐỘC LẬP")     
print(classification_report(y_test, test_preds, target_names=classes, digits=4, zero_division=0))

# ==============================================================================
# 5. XUẤT CÁC BIỂU ĐỒ
# ==============================================================================
# 1. Confusion Matrix
cm_test = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(8, 6))
cmn_test = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
sns.heatmap(cmn_test, annot=True, fmt='.2%', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix - LightGBM Test')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# 2. Biểu đồ Loss / Accuracy K-Fold Average
max_epochs = max([len(e['Train']['multi_logloss']) for e in all_evals])

train_loss_pad = np.array([np.pad(e['Train']['multi_logloss'], (0, max_epochs - len(e['Train']['multi_logloss'])), 'edge') for e in all_evals])
val_loss_pad = np.array([np.pad(e['Validation']['multi_logloss'], (0, max_epochs - len(e['Validation']['multi_logloss'])), 'edge') for e in all_evals])
train_acc_pad = np.array([np.pad([1-x for x in e['Train']['multi_error']], (0, max_epochs - len(e['Train']['multi_error'])), 'edge') for e in all_evals])
val_acc_pad = np.array([np.pad([1-x for x in e['Validation']['multi_error']], (0, max_epochs - len(e['Validation']['multi_error'])), 'edge') for e in all_evals])

mean_train_loss, mean_val_loss = np.mean(train_loss_pad, axis=0), np.mean(val_loss_pad, axis=0)
mean_train_acc, mean_val_acc = np.mean(train_acc_pad, axis=0), np.mean(val_acc_pad, axis=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(mean_train_loss, linestyle='-', color='blue', lw=2, label='Train Loss')
ax1.plot(mean_val_loss, linestyle='-', color='orange', lw=2, label='Val Loss')
ax1.set_title('LightGBM K-Fold: Loss Curve')
ax1.set_xlabel('Boosting Rounds')
ax1.set_ylabel('Multi-Logloss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(mean_train_acc, linestyle='-', color='blue', lw=2, label='Train Accuracy')
ax2.plot(mean_val_acc, linestyle='-', color='orange', lw=2, label='Val Accuracy')
ax2.set_title('LightGBM K-Fold: Accuracy Curve')
ax2.set_xlabel('Boosting Rounds')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. Biểu đồ ROC Curve
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
plt.figure(figsize=(10, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 

for i, color in enumerate(colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], test_probs_accum[:, i])
    roc_auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2.5, label=f'ROC {classes[i]} (AUC = {roc_auc_val:.4f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.title('Multiclass ROC Curve - LightGBM (Ensemble)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()