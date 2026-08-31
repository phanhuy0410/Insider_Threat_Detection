import os
import gc
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalFocalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, label_binarize
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.backend.clear_session()

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
all_histories = []

for fold_idx, fold_cfg in enumerate(time_folds, 1):
    train_mask = np.isin(w_dev, fold_cfg["train_weeks"])
    val_mask = np.isin(w_dev, fold_cfg["val_weeks"])
    
    X_train = X_dev_seq[train_mask]
    y_train = y_dev[train_mask]
    X_val = X_dev_seq[val_mask]
    y_val = y_dev[val_mask]
    
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train.reshape(-1, n_features))
    X_val_flat = scaler.transform(X_val.reshape(-1, n_features))
    X_test_flat = scaler.transform(X_test_seq.reshape(-1, n_features))
    
    X_train_scaled = X_train_flat.reshape(-1, n_steps, n_features)
    X_val_scaled = X_val_flat.reshape(-1, n_steps, n_features)
    X_test_scaled = X_test_flat.reshape(-1, n_steps, n_features)

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

    X_train_final = X_balanced_flat.reshape(-1, n_steps, n_features)
    y_train_encoded = tf.keras.utils.to_categorical(y_balanced, num_classes=4)
    y_val_encoded = tf.keras.utils.to_categorical(y_val, num_classes=4)

    inputs = Input(shape=(n_steps, n_features))
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(4, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
    	optimizer=Adam(learning_rate=0.0001), 
    	loss=CategoricalFocalCrossentropy(gamma=2.0),
    	metrics=['accuracy']
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=0)
    
    start_train_time = time.time()
    history = model.fit(
        X_train_final, y_train_encoded, 
        validation_data=(X_val_scaled, y_val_encoded),
        epochs=100, batch_size=256, callbacks=[early_stopping, rlr], verbose=0
    )
    train_time = (time.time() - start_train_time) / 60
    print(f"✅ Đã huấn luyện xong Fold {fold_idx} trong {train_time:.2f} phút.")

    all_histories.append(history.history)
    oof_probs[val_mask] = model.predict(X_val_scaled, verbose=0)
    test_probs_accum += model.predict(X_test_scaled, verbose=0) / len(time_folds)
    
    gc.collect()

classes = ['Benign', 'Scen1', 'Scen2', 'Scen3']
benign_oof_probs = oof_probs[y_dev == 0]

thresh_s1 = np.percentile(benign_oof_probs[:, 1], 99.999)
thresh_s2 = np.percentile(benign_oof_probs[:, 2], 99.5)
thresh_s3 = np.percentile(benign_oof_probs[:, 3], 99.995)

test_preds = np.zeros(len(test_probs_accum), dtype=int) 
for i in range(len(test_probs_accum)):
    candidates = []
    if test_probs_accum[i, 1] >= thresh_s1: candidates.append((1, test_probs_accum[i, 1]))
    if test_probs_accum[i, 2] >= thresh_s2: candidates.append((2, test_probs_accum[i, 2]))
    if test_probs_accum[i, 3] >= thresh_s3: candidates.append((3, test_probs_accum[i, 3]))
    if candidates:
        test_preds[i] = max(candidates, key=lambda item: item[1])[0]
        
print(classification_report(y_test, test_preds, target_names=classes, digits=4, zero_division=0))

cm_test = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(8, 6))
cmn_test = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
sns.heatmap(cmn_test, annot=True, fmt='.2%', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix - Test')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

max_epochs = max([len(h['loss']) for h in all_histories])
train_loss = np.mean([np.pad(h['loss'], (0, max_epochs - len(h['loss'])), 'edge') for h in all_histories], axis=0)
val_loss = np.mean([np.pad(h['val_loss'], (0, max_epochs - len(h['val_loss'])), 'edge') for h in all_histories], axis=0)
train_acc = np.mean([np.pad(h['accuracy'], (0, max_epochs - len(h['accuracy'])), 'edge') for h in all_histories], axis=0)
val_acc = np.mean([np.pad(h['val_accuracy'], (0, max_epochs - len(h['val_accuracy'])), 'edge') for h in all_histories], axis=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(train_loss, color='blue', lw=2, label='Train Loss')
ax1.plot(val_loss, color='orange', lw=2, label='Val Loss')
ax1.set_title('Loss')
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.plot(train_acc, color='blue', lw=2, label='Train Accuracy')
ax2.plot(val_acc, color='orange', lw=2, label='Val Accuracy')
ax2.set_title('Accuracy')
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

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
plt.title('Multiclass ROC Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()