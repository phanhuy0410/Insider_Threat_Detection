import pandas as pd
import gc
import os, re
import time
import random
import numpy as np
import pickle
import multiprocessing as mp
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

TENSOR_DIR = "/kaggle/input/datasets/doanthimo80/tensor-data-static"
PATH_SESSION = '/kaggle/input/datasets/phanthanhhoang/cert-r42-session/session_r4.2.parquet'
classes_names = ['Benign', 'Scen1', 'Scen2', 'Scen3']

# ==============================================================================
# KHÔI PHỤC USER MAPPING
# ==============================================================================
print("\nĐang khôi phục lại bộ danh sách User Mapping từ tập Session gốc...")
df_sess_rec = pd.read_parquet(PATH_SESSION)
df_sess_rec = df_sess_rec.sort_values(by='starttime').reset_index(drop=True)

test_idx_rec = int(len(df_sess_rec) * 0.80)
df_sess_dev_rec = df_sess_rec.iloc[:test_idx_rec].copy()
df_sess_test_raw_rec = df_sess_rec.iloc[test_idx_rec:].copy()

known_insiders_dev_rec = set(df_sess_dev_rec[df_sess_dev_rec['insider'] != 0]['user'])
df_sess_test_rec = df_sess_test_raw_rec[~df_sess_test_raw_rec['user'].isin(known_insiders_dev_rec)].copy()

unique_users_dev = sorted(df_sess_dev_rec['user'].unique().tolist())
user_mapping_dev = {u: i for i, u in enumerate(unique_users_dev)}
unique_users_test = sorted(df_sess_test_rec['user'].unique().tolist())
user_mapping_test = {u: i for i, u in enumerate(unique_users_test)}

del df_sess_rec, df_sess_dev_rec, df_sess_test_raw_rec, df_sess_test_rec, known_insiders_dev_rec; gc.collect()

# ==============================================================================
# ĐỌC VÀ XỬ LÝ DỮ LIỆU
# ==============================================================================
def load_pass1_session(chunk_dir, prefix):
    X_s, y, uid, w_list, day_list, counts = [], [], [], [], [], []
    files = [os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.startswith(f"{prefix}_chunk")]
    files.sort(key=lambda x: int(re.search(r'chunk_(\d+)', x).group(1)))
    for f in files:
        with np.load(f) as data:
            if len(data['y']) == 0: continue
            X_s.append(data['X_sess']); y.append(data['y']); uid.append(data['user_id'])
            w_list.append(data['w']); day_list.append(data['day']); counts.append(len(data['y']))
    return np.concatenate(X_s), np.concatenate(y), np.concatenate(uid), np.concatenate(w_list), np.concatenate(day_list), counts, files

def load_userday_lookup(file_path, user_mapping):
    df = pd.read_parquet(file_path)
    df['user_id'] = df['user'].map(user_mapping)
    df = df.dropna(subset=['user_id'])
    df['user_id'] = df['user_id'].astype(np.int16)
    meta_cols = ['user', 'insider', 'user_id']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    keys = df['user_id'].tolist()
    values = df[feature_cols].to_numpy(dtype=np.float32)
    lookup_dict = dict(zip(keys, values))
    return lookup_dict, feature_cols

def pool_3d_to_2d(X_3d):
    X_3d = np.nan_to_num(X_3d, nan=0.0, posinf=0.0, neginf=0.0)
    X_mean = np.mean(X_3d, axis=1)
    X_max = np.max(X_3d, axis=1)
    X_std = np.std(X_3d, axis=1) 
    return np.hstack([X_mean, X_max, X_std]).astype(np.float32)

def train_single_fold_lgbm(fold_idx, train_weeks, val_weeks, X_dev_sess_full, y_dev_full, uid_dev_full, w_dev_full, day_dev_full, chunk_files, chunk_counts, user_lookup_dev, F_user, user_lookup_test, chunk_files_test, len_y_test):
    import numpy as np
    import lightgbm as lgb
    np.random.seed(SEED)
    
    print(f"\n{'='*20} KHỞI ĐỘNG TIẾN TRÌNH FOLD {fold_idx + 1} {'='*20}")
    
    # 1. Split Train / Val
    train_mask = np.isin(w_dev_full, train_weeks)
    val_mask = np.isin(w_dev_full, val_weeks)
    idx_train_raw = np.where(train_mask)[0]
    idx_val = np.where(val_mask)[0]
    if len(idx_val) == 0: return
    
    X_tr_sess_flat = X_dev_sess_full[idx_train_raw].reshape(len(idx_train_raw), -1)
    X_tr_sess_flat = np.nan_to_num(X_tr_sess_flat, nan=0.0, posinf=0.0, neginf=0.0)
    y_train_temp = y_dev_full[idx_train_raw]
    idx_benign = np.where(y_train_temp == 0)[0]
    idx_attack = np.where(y_train_temp != 0)[0]
    
    kmeans = MiniBatchKMeans(n_clusters=100, random_state=SEED, batch_size=2048)
    distances = kmeans.fit_transform(X_tr_sess_flat[idx_benign])
    
    target_benign = 20000
    chosen_benign_idx = []
    for cluster_id, count in pd.Series(kmeans.labels_).value_counts().items():
        n_draw = int(np.round((count / len(idx_benign)) * target_benign))
        if n_draw <= 0: continue
        idx_in_cluster = np.where(kmeans.labels_ == cluster_id)[0]
        dist = distances[idx_in_cluster, cluster_id]
        sorted_idx = idx_in_cluster[np.argsort(dist)]
        chosen_benign_idx.extend(sorted_idx[:n_draw])
    
    final_benign_idx = idx_benign[np.array(chosen_benign_idx)]
    if len(final_benign_idx) > target_benign: final_benign_idx = np.random.choice(final_benign_idx, target_benign, replace=False)
    elif len(final_benign_idx) < target_benign: final_benign_idx = np.concatenate([final_benign_idx, np.random.choice(np.setdiff1d(idx_benign, final_benign_idx), target_benign - len(final_benign_idx), replace=False)])
    
    keep_indices = np.concatenate([final_benign_idx, idx_attack])
    keep_indices.sort()
    keep_global_idx = idx_train_raw[keep_indices]
    
    y_train = y_dev_full[keep_global_idx]
    uid_train = uid_dev_full[keep_global_idx]
    day_train = day_dev_full[keep_global_idx]
    X_train_sess = X_dev_sess_full[keep_global_idx]
    
    y_val = y_dev_full[idx_val]
    uid_val = uid_dev_full[idx_val]
    day_val = day_dev_full[idx_val]
    X_val_sess = X_dev_sess_full[idx_val]
    
    X_train_user = np.array([user_lookup_dev.get(u, np.zeros(F_user)) for u in uid_train], dtype=np.float32)
    X_val_user = np.array([user_lookup_dev.get(u, np.zeros(F_user)) for u in uid_val], dtype=np.float32)
    
    X_train_concat = np.hstack([pool_3d_to_2d(X_train_sess), X_train_user])
    X_val_concat = np.hstack([pool_3d_to_2d(X_val_sess), X_val_user])
    
    del X_train_sess, X_val_sess; gc.collect()

    X_train_concat = np.nan_to_num(X_train_concat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X_val_concat = np.nan_to_num(X_val_concat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_concat)
    X_val_scaled = scaler.transform(X_val_concat)
    del X_train_concat, X_val_concat; gc.collect()
    
    smote = SMOTE(sampling_strategy={1: 10000, 2: 10000, 3: 10000}, random_state=SEED, k_neighbors=5)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    del X_train_scaled; gc.collect()

    lgb_clf = lgb.LGBMClassifier(
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
    
    evals_result = {}
    TRAIN_START = time.time()
    lgb_clf.fit(
        X_train_bal, y_train_bal,
        eval_set=[(X_train_bal, y_train_bal), (X_val_scaled, y_val)],
        eval_names=['train', 'val'],
        eval_metric=['multi_logloss', 'multi_error'], 
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False), lgb.record_evaluation(evals_result)]
    )
    TRAIN_DURATION = (time.time() - TRAIN_START) / 60
    print(f"Xong Fold {fold_idx + 1} trong {TRAIN_DURATION:.2f} phút.")
    
    fold_history = {
        'loss': evals_result['train']['multi_logloss'],
        'val_loss': evals_result['val']['multi_logloss'],
        'accuracy': [1 - x for x in evals_result['train']['multi_error']],
        'val_accuracy': [1 - x for x in evals_result['val']['multi_error']]
    }
    
    # 7. Inference Val & Test
    val_probs = lgb_clf.predict_proba(X_val_scaled)
    y_val_ohe = pd.get_dummies(y_val).values
    if y_val_ohe.shape[1] == 4:
        fold_auc = roc_auc_score(y_val_ohe, val_probs, multi_class='ovr', average='macro')
    else: fold_auc = 0.0
    print(f"MACRO AUC CỦA FOLD {fold_idx + 1}: {fold_auc:.4f}")

    test_probs_list = []
    for f in chunk_files_test: 
        with np.load(f) as data:
            if len(data['y']) == 0: continue
            batch_sess, batch_uid = data['X_sess'], data['user_id']
            
            batch_concat = np.hstack([
                pool_3d_to_2d(batch_sess), 
                np.array([user_lookup_test.get(u, np.zeros(F_user)) for u in batch_uid], dtype=np.float32)
            ])
            batch_concat = np.nan_to_num(batch_concat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            batch_scaled = scaler.transform(batch_concat)
            test_probs_list.append(lgb_clf.predict_proba(batch_scaled))
    
    test_probs_fold = np.concatenate(test_probs_list, axis=0)
    
    # 8. Lưu kết quả tạm thời
    np.savez(
        os.path.join(TMP_RESULT_DIR, f"fold_{fold_idx}.npz"),
        val_probs = val_probs, y_val = y_val, test_probs = test_probs_fold, fold_auc = np.array([fold_auc]) 
    )
    with open(os.path.join(TMP_RESULT_DIR, f"history_{fold_idx}.pkl"), 'wb') as f:
        pickle.dump(fold_history, f)
        
    print(f"TIẾN TRÌNH FOLD {fold_idx + 1} HOÀN TẤT. TRẢ LẠI 100% RAM!")
    return

if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    PREFIX = "dev" 
    X_dev_sess_full, y_dev_full, uid_dev_full, w_dev_full, day_dev_full, chunk_counts, chunk_files = load_pass1_session(TENSOR_DIR, PREFIX)
    user_lookup_dev, feature_cols_ud = load_userday_lookup(f"{TENSOR_DIR}/userday_clean_dev.parquet", user_mapping_dev)
    F_user = len(feature_cols_ud) 
    user_lookup_test, _ = load_userday_lookup(f"{TENSOR_DIR}/userday_clean_test.parquet", user_mapping_test)
    _, y_test, _, _, _, _, chunk_files_test = load_pass1_session(TENSOR_DIR, "test")
    
    TMP_RESULT_DIR = "./temp_results"
    os.makedirs(TMP_RESULT_DIR, exist_ok=True)
    
    time_folds = [
        {"train_weeks": list(range(0, 30)), "val_weeks": list(range(31, 35))},
        {"train_weeks": list(range(0, 35)), "val_weeks": list(range(36, 40))},
        {"train_weeks": list(range(0, 40)), "val_weeks": list(range(41, 45))},
        {"train_weeks": list(range(0, 45)), "val_weeks": list(range(46, 50))},
        {"train_weeks": list(range(0, 50)), "val_weeks": list(range(51, 56))},
    ]
    
    for fold_idx, fold_cfg in enumerate(time_folds):
        print(f"\nĐang nạp lệnh cho Fold {fold_idx + 1}")
        p = mp.Process(
            target=train_single_fold_lgbm, 
            args=(fold_idx, fold_cfg["train_weeks"], fold_cfg["val_weeks"], X_dev_sess_full, y_dev_full, uid_dev_full, w_dev_full, day_dev_full, chunk_files, chunk_counts, user_lookup_dev, F_user, user_lookup_test, chunk_files_test, len(y_test))
        )
        p.start()
        p.join()  
        if p.exitcode != 0: break 

    oof_probs_list, oof_y_list, fold_aucs, training_histories = [], [], [], []
    test_probs_accum = np.zeros((len(y_test), 4)) 

    for fold_idx in range(len(time_folds)):
        file_path = os.path.join(TMP_RESULT_DIR, f"fold_{fold_idx}.npz")
        hist_path = os.path.join(TMP_RESULT_DIR, f"history_{fold_idx}.pkl")
        if os.path.exists(file_path):
            with np.load(file_path) as data:
                oof_probs_list.append(data['val_probs']); oof_y_list.append(data['y_val'])
                fold_aucs.append(data['fold_auc'][0]); test_probs_accum += (data['test_probs'] / len(time_folds))
        if os.path.exists(hist_path):
            with open(hist_path, 'rb') as f: training_histories.append(pickle.load(f))

    print(f"TRUNG BÌNH MACRO AUC SAU {len(fold_aucs)} FOLD: {np.mean(fold_aucs):.4f}")
    
    oof_probs_all = np.vstack(oof_probs_list)
    oof_y_all = np.concatenate(oof_y_list)
    
    np.savez(
        '/kaggle/working/bi_view_predictions_lgbm.npz',
        oof_probs = oof_probs_all,
        oof_y = oof_y_all,
        test_probs = test_probs_accum,
        y_test = y_test,
        fold_aucs = fold_aucs
    )
    with open('/kaggle/working/bi_view_histories_lgbm.pkl', 'wb') as f:
        pickle.dump(training_histories, f)
        
    print("Đã lưu xong 2 file: bi_view_predictions_lgbm.npz và bi_view_histories_lgbm.pkl!")

    # Tune Threshold
    oof_benign_probs = oof_probs_all[(oof_y_all == 0)]
    THRESH_S1 = np.percentile(oof_benign_probs[:, 1], 99.995)
    THRESH_S2 = np.percentile(oof_benign_probs[:, 2], 99.5)  
    THRESH_S3 = np.percentile(oof_benign_probs[:, 3], 99.99)
    
    test_preds = np.zeros(len(test_probs_accum), dtype=int)
    for i in range(len(test_probs_accum)):
        candidates = []
        if test_probs_accum[i, 1] >= THRESH_S1: candidates.append((1, test_probs_accum[i, 1]))
        if test_probs_accum[i, 2] >= THRESH_S2: candidates.append((2, test_probs_accum[i, 2]))
        if test_probs_accum[i, 3] >= THRESH_S3: candidates.append((3, test_probs_accum[i, 3]))
        if candidates: test_preds[i] = max(candidates, key=lambda item: item[1])[0]
    
    y_test_ohe = pd.get_dummies(y_test).values
    if y_test_ohe.shape[1] < 4:
        full = np.zeros((len(y_test), 4))
        for idx, val in enumerate(y_test): full[idx, val] = 1
        y_test_ohe = full
        
    test_auc = roc_auc_score(y_test_ohe, test_probs_accum, multi_class='ovr', average='macro')
    print(f"\nMACRO AUC ENSEMBLE TEST SET: {test_auc:.4f} \n")
    print(classification_report(y_test, test_preds, target_names=classes_names, digits=4, zero_division=0))

    # 1. BIỂU ĐỒ LOSS VÀ ACCURACY
    def plot_training_history_average(histories):
        if not histories: return
        max_epochs = max([len(h['loss']) for h in histories])
        avg_history = {}
        for key in ['loss', 'val_loss', 'accuracy', 'val_accuracy']:
            padded_folds = [np.pad(h[key], (0, max_epochs - len(h[key])), mode='edge') for h in histories]
            avg_history[key] = np.mean(padded_folds, axis=0)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(avg_history['loss'], label='Train Loss', color='blue', lw=2)
        axes[0].plot(avg_history['val_loss'], label='Validation Loss', color='orange', lw=2)
        axes[0].set_title(f'Trung bình Loss (LightGBM Bi-View - {len(histories)} Folds)', fontweight='bold')
        axes[0].set_xlabel('Boosting Round'); axes[0].set_ylabel('Loss'); axes[0].legend(); axes[0].grid(True, linestyle=':', alpha=0.6)
        
        # Vẽ Accuracy
        axes[1].plot(avg_history['accuracy'], label='Train Accuracy', color='blue', lw=2)
        axes[1].plot(avg_history['val_accuracy'], label='Validation Accuracy', color='orange', lw=2)
        axes[1].set_title(f'Trung bình Accuracy (LightGBM Bi-View - {len(histories)} Folds)', fontweight='bold')
        axes[1].set_xlabel('Boosting Round'); axes[1].set_ylabel('Accuracy'); axes[1].legend(); axes[1].grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout(); plt.show()

    plot_training_history_average(training_histories)

    # 2. CONFUSION MATRIX
    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(8, 6))
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_percentages = np.where(row_sums == 0, 0, cm.astype('float') / row_sums)
    annot_labels = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): annot_labels[i, j] = f"{cm_percentages[i, j]:.2%}"
    sns.heatmap(cm_percentages, annot=annot_labels, fmt="", cmap="Blues", xticklabels=classes_names, yticklabels=classes_names)
    plt.title("Confusion Matrix - LightGBM (Bi-View Session+User)", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Actual', fontsize=12); plt.xlabel('Predicted', fontsize=12); plt.tight_layout(); plt.show()

    # 3. ROC CURVE
    plt.figure(figsize=(9, 7))
    colors = ['dodgerblue', 'crimson', 'forestgreen', 'darkorange']
    for i in range(4):
        fpr, tpr, _ = roc_curve(y_test_ohe[:, i], test_probs_accum[:, i])
        roc_auc_val = auc(fpr, tpr)
        if not np.isnan(roc_auc_val):
            plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'ROC curve - {classes_names[i]} (AUC = {roc_auc_val:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance')
    plt.xlim([-0.05, 1.05]); plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold'); plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title(f'Biểu đồ ROC Đa lớp - LightGBM Bi-View (Macro AUC = {test_auc:.4f})', fontsize=15, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=11); plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout(); plt.show()