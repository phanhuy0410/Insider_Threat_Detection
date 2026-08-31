import pandas as pd
import gc
import os, re
import time
import random
import numpy as np
import lightgbm as lgb
import pickle
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.cluster import MiniBatchKMeans
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, auc, f1_score
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------
# 1. CỐ ĐỊNH SEED VÀ CẤU HÌNH ĐƯỜNG DẪN
# ------------------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
print(f"🔒 Hệ thống đã khóa cứng tính ngẫu nhiên với SEED = {SEED}")

TENSOR_DIR = "/kaggle/input/datasets/doanthimo80/tensor-data-static"
PATH_SESSION = '/kaggle/input/datasets/phanthanhhoang/cert-r42-session/session_r4.2.parquet'

TENSOR_DIR_R52 = "/kaggle/input/datasets/doanthimo80/tensor-data-r52-new" 
PATH_SESSION_R52 = '/kaggle/input/datasets/phanthanhhoang/r5-2-session/sessionr5.2.csv'
PATH_USERDAY_R52_DEV = f"{TENSOR_DIR_R52}/userday_clean_dev_r52.parquet"
PATH_USERDAY_R52_TEST = f"{TENSOR_DIR_R52}/userday_clean_test_r52.parquet"

classes_names = ['Benign', 'Scen1', 'Scen2', 'Scen3']

sample_data = np.load(f"{TENSOR_DIR}/dev_chunk_0.npz")
_, T_sess, F_sess_actual = sample_data['X_sess'].shape
del sample_data; gc.collect()

# ==============================================================================
# 🎛️ BẢNG ĐIỀU KHIỂN FUZZY THÍ NGHIỆM (70 CỘT)
# ==============================================================================
FUZZY_SELECTED_FEATURES = [
    'job_site_ratio', 'peer_usb_z', 'peer_file_z', 'peer_email_z', 'peer_http_z', 
    'leak_site_ratio', 'hack_site_ratio', 'email_ratio', 'duration_quantile', 
    'afterhour_ratio', 'http_hackf_mean_url_depth', 'file_ratio', 'http_mean_url_depth', 
    'weekend_afterhour_ratio', 'http_hackf_mean_http_c_nwords', 'http_socnetf_mean_url_depth',
    'n_concurrent_sessions', 'http_ratio', 'email_mean_n_atts', 'usb_ratio', 'ses_start',
    'http_hackf_mean_http_c_len', 'weekend_ratio', 'email_mean_n_bccdes', 'duration', 
    'email_mean_n_exdes', 'n_email_zscore', 'http_leakf_ratio', 'n_file_zscore', 'n_usb',
    'ses_end', 'http_otherf_mean_url_depth', 'email_mean_email_text_nwords', 'n_http_zscore',
    'http_n_otherf', 'email_n-exbccmail1', 'http_n_jobf', 'n_allact', 'http_hackf_ratio',
    'email_mean_email_size', 'http_jobf_ratio', 'file_txtf_n-disk2', 'n_http', 'n_email',
    'isafterhour', 'isworkhour', 'http_socnetf_mean_url_len', 'file_n-disk2', 'email_mean_email_text_slen',
    'file_n-disk1', 'http_n_socnetf', 'email_mean_n_des', 'http_socnetf_mean_http_c_len', 'http_mean_url_len',
    'http_jobf_mean_url_depth', 'email_n-Xemail1', 'http_n_cloudf', 'http_mean_http_c_len', 'http_otherf_mean_url_len',
    'n_days', 'http_socnetf_mean_http_c_nwords', 'http_mean_http_c_nwords', 'http_hackf_mean_url_len', 
    'http_leakf_mean_http_c_nwords', 'http_leakf_mean_http_c_len', 'file_phof_mean_file_len', 'http_jobf_mean_url_len', 
    'http_leakf_mean_url_depth', 'file_docf_n-disk1', 'file_mean_file_len'
]

# ==============================================================================
# 🔄 KHÔI PHỤC USER MAPPING
# ==============================================================================
print("🔄 Đang khôi phục User Mapping R4.2 (Chỉ lấy Dev để Train/Val)...")
df_sess_rec = pd.read_parquet(PATH_SESSION, columns=['user', 'starttime'])
df_sess_rec = df_sess_rec.sort_values(by='starttime').reset_index(drop=True)
test_idx_rec = int(len(df_sess_rec) * 0.80)
df_sess_dev_rec = df_sess_rec.iloc[:test_idx_rec].copy()

unique_users_dev = sorted(df_sess_dev_rec['user'].unique().tolist())
user_mapping_dev = {u: i for i, u in enumerate(unique_users_dev)}
del df_sess_rec, df_sess_dev_rec; gc.collect()

print("🔄 Đang khôi phục User Mapping R5.2 (Chia 80/20 lấy Test R5.2)...")
df_sess_r52 = pd.read_csv(PATH_SESSION_R52, usecols=['user', 'starttime', 'insider'])
df_sess_r52 = df_sess_r52.sort_values(by='starttime').reset_index(drop=True)
test_idx_r52 = int(len(df_sess_r52) * 0.80)
df_sess_dev_r52 = df_sess_r52.iloc[:test_idx_r52].copy()
df_sess_test_raw_r52 = df_sess_r52.iloc[test_idx_r52:].copy()

known_insiders_dev_r52 = set(df_sess_dev_r52[df_sess_dev_r52['insider'] != 0]['user'])
df_sess_test_r52 = df_sess_test_raw_r52[~df_sess_test_raw_r52['user'].isin(known_insiders_dev_r52)].copy()

unique_users_dev_r52 = sorted(df_sess_dev_r52['user'].unique().tolist())
user_mapping_dev_r52 = {u: i for i, u in enumerate(unique_users_dev_r52)}
unique_users_test_r52 = sorted(df_sess_test_r52['user'].unique().tolist())
user_mapping_test_r52 = {u: i for i, u in enumerate(unique_users_test_r52)}
del df_sess_r52, df_sess_dev_r52, df_sess_test_raw_r52, df_sess_test_r52; gc.collect()

print(f"✅ User R4.2 Dev: {len(user_mapping_dev)} | User R5.2 Dev: {len(user_mapping_dev_r52)} | Test R5.2: {len(user_mapping_test_r52)}")

# ==============================================================================
# 🛡️ HÀM ĐỌC DỮ LIỆU
# ==============================================================================
def load_pass1_session(chunk_dir, prefix):
    X_s, y, uid, w_list, day_list, counts = [], [], [], [], [], []
    files = [os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.startswith(f"{prefix}_chunk")]
    if len(files) == 0: raise FileNotFoundError(f"❌ Không tìm thấy file {prefix} tại {chunk_dir}")
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
    return dict(zip(keys, values)), feature_cols

class NumpyFuzzyMembershipLayer:
    def __init__(self, num_rules=3, concat_original=True):
        if num_rules != 3:
            raise ValueError("Kiến trúc phân vị Q25, Q50, Q75 yêu cầu bắt buộc num_rules=3")
        self.num_rules = num_rules
        self.concat_original = concat_original
        self.indices = None
        self.centers = None
        self.sigmas = None

    def fit(self, X, indices=None):
        self.indices = indices if indices is not None else list(range(X.shape[1]))
        if not self.indices:
            return self
        X_fuz = X[:, self.indices]
        fuz_dim = X_fuz.shape[1]
        
        self.centers = np.zeros((fuz_dim, self.num_rules), dtype=np.float32)
        self.sigmas = np.zeros((fuz_dim, self.num_rules), dtype=np.float32)
        
        for i in range(fuz_dim):
            col_data = X_fuz[:, i]
            q25, q50, q75 = np.percentile(col_data, [25, 50, 75])
            self.centers[i] = [q25, q50, q75]
            # Hệ số 1.1774 = sqrt(2 * ln(2)), chuẩn của Gaussian Fuzzy Set
            s1 = max((q50 - q25) / 1.1774, 1e-3) 
            s2 = max((q75 - q25) / (2 * 1.1774), 1e-3) 
            s3 = max((q75 - q50) / 1.1774, 1e-3)
            self.sigmas[i] = [s1, s2, s3]
        return self

    def transform(self, X):
        if not self.indices:
            return X
            
        X_fuz = X[:, self.indices]
        B = X_fuz.shape[0]
        
        # Mở rộng X thành shape: (Batch, fuz_dim, num_rules)
        x_expanded = np.repeat(X_fuz[:, :, np.newaxis], self.num_rules, axis=2)
        
        # Mở rộng centers và sigmas
        centers_expanded = self.centers[np.newaxis, :, :]
        sigmas_expanded = self.sigmas[np.newaxis, :, :]
        
        # Tính khoảng cách Gaussian
        diff = (x_expanded - centers_expanded) / sigmas_expanded
        memberships = np.exp(-0.5 * (diff ** 2))
        
        # Flatten thành shape (Batch, fuz_dim * num_rules)
        reshaped = memberships.reshape(B, -1)
        
        if self.concat_original:
            return np.concatenate([X, reshaped], axis=1)
        return reshaped

# ==============================================================================
# 🌀 PIPELINE HUẤN LUYỆN LIGHTGBM: TRAIN MIX | VAL MIX | TEST R5.2
# ==============================================================================
def train_single_fold_lightgbm(fold_idx, train_weeks, val_weeks, 
                               X_mix, y_mix, uid_mix, w_mix, 
                               global_user_lookup, F_user, 
                               chunk_files_r52_test, OFFSET, flat_fuzzy_indices, TMP_RESULT_DIR): 
    
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED); np.random.seed(SEED)
    print(f"\n{'='*20} KHỞI ĐỘNG TIẾN TRÌNH FOLD {fold_idx + 1} (LIGHTGBM) {'='*20}")
    
    train_mask = np.isin(w_mix, train_weeks)
    idx_train_raw = np.where(train_mask)[0]
    
    val_mask = np.isin(w_mix, val_weeks)
    idx_val_mix = np.where(val_mask)[0]
    if len(idx_val_mix) == 0: return
    
    # --- KMEANS DOWNSAMPLING ---
    X_tr_sess_flat = X_mix[idx_train_raw].reshape(len(idx_train_raw), -1)
    X_tr_sess_flat = np.nan_to_num(X_tr_sess_flat, nan=0.0, posinf=65000.0, neginf=-65000.0)
    y_train_temp = y_mix[idx_train_raw]
    idx_benign = np.where(y_train_temp == 0)[0]
    idx_attack = np.where(y_train_temp != 0)[0]
    
    kmeans = MiniBatchKMeans(n_clusters=100, random_state=SEED, batch_size=2048)
    distances = kmeans.fit_transform(X_tr_sess_flat[idx_benign])
    
    target_benign = 20000; chosen_benign_idx = []
    for cluster_id, count in pd.Series(kmeans.labels_).value_counts().items():
        n_draw = int(np.round((count / len(idx_benign)) * target_benign))
        if n_draw <= 0: continue
        idx_in_cluster = np.where(kmeans.labels_ == cluster_id)[0]
        dist = distances[idx_in_cluster, cluster_id]
        sorted_idx = idx_in_cluster[np.argsort(dist)]
        n_cluster = len(sorted_idx)
        near_end, mid_end = int(n_cluster * 0.30), int(n_cluster * 0.70)
        n_near, n_mid = int(n_draw * 0.50), int(n_draw * 0.30)
        n_far = n_draw - n_near - n_mid
        selected = []
        if len(sorted_idx[:near_end]) > 0: selected.extend(sorted_idx[:near_end][:min(n_near, len(sorted_idx[:near_end]))])
        if len(sorted_idx[near_end:mid_end]) > 0 and n_mid > 0: selected.extend(sorted_idx[near_end:mid_end][max(0, len(sorted_idx[near_end:mid_end])//2 - n_mid//2):][:n_mid])
        if len(sorted_idx[mid_end:]) > 0 and n_far > 0: selected.extend(sorted_idx[mid_end:][-min(n_far, len(sorted_idx[mid_end:])):])
        chosen_benign_idx.extend(selected)
    
    final_benign_idx = idx_benign[np.array(chosen_benign_idx)]
    if len(final_benign_idx) > target_benign: final_benign_idx = np.random.choice(final_benign_idx, target_benign, replace=False)
    elif len(final_benign_idx) < target_benign: final_benign_idx = np.concatenate([final_benign_idx, np.random.choice(np.setdiff1d(idx_benign, final_benign_idx), target_benign - len(final_benign_idx), replace=False)])
    
    keep_indices = np.concatenate([final_benign_idx, idx_attack])
    keep_indices.sort()
    keep_global_idx = idx_train_raw[keep_indices]
    
    X_train_sess = X_mix[keep_global_idx]; y_train = y_mix[keep_global_idx]; uid_train = uid_mix[keep_global_idx]
    X_val_sess = X_mix[idx_val_mix]; y_val = y_mix[idx_val_mix]; uid_val = uid_mix[idx_val_mix]
    
    X_train_user = np.array([global_user_lookup.get(u, np.zeros(F_user)) for u in uid_train], dtype=np.float32)
    X_val_user = np.array([global_user_lookup.get(u, np.zeros(F_user)) for u in uid_val], dtype=np.float32)
    
    # --- CHUẨN BỊ DỮ LIỆU BẢNG VÀ ÁP DỤNG FUZZY ---
    X_train_sess_flat = X_train_sess.reshape(X_train_sess.shape[0], -1)
    X_val_sess_flat = X_val_sess.reshape(X_val_sess.shape[0], -1)

    X_train_full = np.concatenate([X_train_sess_flat, X_train_user], axis=1)
    X_val_full = np.concatenate([X_val_sess_flat, X_val_user], axis=1)

    X_train_full = np.nan_to_num(X_train_full, nan=0.0, posinf=65000.0, neginf=-65000.0)
    X_val_full = np.nan_to_num(X_val_full, nan=0.0, posinf=65000.0, neginf=-65000.0)

    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(X_train_full).astype(np.float32)
    X_val_full = scaler.transform(X_val_full).astype(np.float32)
    del X_train_sess, X_val_sess, X_train_sess_flat, X_val_sess_flat, X_train_user, X_val_user; gc.collect()

    fuzzy_transform = NumpyFuzzyMembershipLayer(num_rules=3, concat_original=True)
    if flat_fuzzy_indices:
        fuzzy_transform.fit(X_train_full, flat_fuzzy_indices)
        X_train_full = fuzzy_transform.transform(X_train_full)
        X_val_full = fuzzy_transform.transform(X_val_full)

    # --- SMOTE CÂN BẰNG DỮ LIỆU ---
    print(" -> Đang chạy SMOTE cân bằng nhãn...")
    smote = SMOTE(sampling_strategy={1: 10000, 2: 10000, 3: 10000}, random_state=SEED, k_neighbors=5)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_full, y_train)
    del X_train_full; gc.collect()

    # =========================================================================
    # 🚀 HUẤN LUYỆN LIGHTGBM
    # =========================================================================
    print(" -> Bắt đầu train LightGBM...")
    clf = lgb.LGBMClassifier(
        objective='multiclass', 
        num_class=4, 
        n_estimators=1500, 
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

    fit_start_time = time.time()
    
    clf.fit(
        X_train_bal, y_train_bal,
        eval_set=[(X_train_bal, y_train_bal), (X_val_full, y_val)],
        eval_names=['train', 'val'],
        eval_metric=['multi_logloss', 'multi_error'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )
    
    fit_duration_minutes = (time.time() - fit_start_time) / 60.0
    print(f" ⏱️ THỜI GIAN TRAIN LÕI FOLD {fold_idx + 1}: {fit_duration_minutes:.2f} phút")
    
    train_err = clf.evals_result_['train']['multi_error']
    val_err = clf.evals_result_['val']['multi_error']
    clf_history = {
        'loss': clf.evals_result_['train']['multi_logloss'],
        'val_loss': clf.evals_result_['val']['multi_logloss'],
        'accuracy': [1.0 - e for e in train_err],
        'val_accuracy': [1.0 - e for e in val_err]
    }

    val_probs = clf.predict_proba(X_val_full)
    y_val_ohe = np.eye(4)[y_val]
    fold_auc = roc_auc_score(y_val_ohe, val_probs, multi_class='ovr', average='macro')
    print(f" 🔥 MACRO AUC FOLD {fold_idx + 1} (VAL MIX): {fold_auc:.4f}")

    # ---------------------------------------------------------
    # 🎯 CHỈ INFERENCE TẬP TEST R5.2 (Áp dụng cấu trúc Fuzzy)
    # ---------------------------------------------------------
    print(" -> Đang quét Inference tập Test R5.2...")
    prob_r52_list = []
    for f in chunk_files_r52_test: 
        with np.load(f) as data:
            if len(data['y']) == 0: continue
            mask = data['y'] != 4
            if not np.any(mask): continue
            batch_sess = data['X_sess'][mask]
            batch_uid = data['user_id'][mask] + OFFSET 

        batch_sess_flat = batch_sess.reshape(batch_sess.shape[0], -1)
        batch_user_raw = np.array([global_user_lookup.get(u, np.zeros(F_user)) for u in batch_uid], dtype=np.float32)
        
        batch_full = np.concatenate([batch_sess_flat, batch_user_raw], axis=1)
        batch_full = np.nan_to_num(batch_full, nan=0.0, posinf=65000.0, neginf=-65000.0)
        batch_full_s = scaler.transform(batch_full).astype(np.float32)
        
        if flat_fuzzy_indices: 
            batch_full_s = fuzzy_transform.transform(batch_full_s)
            
        prob_r52_list.append(clf.predict_proba(batch_full_s))

    r52_probs_fold = np.concatenate(prob_r52_list, axis=0)
    del prob_r52_list, batch_full, batch_full_s; gc.collect()

    np.savez(
        os.path.join(TMP_RESULT_DIR, f"fold_{fold_idx}.npz"),
        val_probs = val_probs,        
        y_val = y_val,                
        r52_probs = r52_probs_fold, 
        fold_auc = np.array([fold_auc]),
        train_time = np.array([fit_duration_minutes])
    )
    
    with open(os.path.join(TMP_RESULT_DIR, f"history_{fold_idx}.pkl"), 'wb') as f:
        pickle.dump(clf_history, f)
        
    print(f" ✅ TIẾN TRÌNH FOLD {fold_idx + 1} HOÀN TẤT!")
    return

# ==============================================================================
# 🚀 VÙNG AN TOÀN MAIN PROCESS
# ==============================================================================
if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    print("\n" + "="*75)
    print("🚀 PIPELINE 4 (LIGHTGBM + NEURO-FUZZY 70): TRAIN MIX | VAL MIX | TEST R5.2")
    print("="*75)
    
    global_start_time = time.time()
    
    X_dev_42, y_dev_42, uid_dev_42, w_dev_42, day_dev_42, _, _ = load_pass1_session(TENSOR_DIR, "dev")
    user_lookup_42_dev, feature_cols_ud = load_userday_lookup(f"{TENSOR_DIR}/userday_clean_dev.parquet", user_mapping_dev)
    F_user = len(feature_cols_ud) 
    
    print("📥 Nạp dữ liệu R5.2 để Mix Train/Val và làm Test Test R5.2...")
    user_lookup_52_dev, _ = load_userday_lookup(PATH_USERDAY_R52_DEV, user_mapping_dev_r52)
    user_lookup_52_test, _ = load_userday_lookup(PATH_USERDAY_R52_TEST, user_mapping_test_r52)
    
    X_dev_52, y_dev_52_full, uid_dev_52, w_dev_52, day_dev_52, _, _ = load_pass1_session(TENSOR_DIR_R52, "dev_r52")
    _, y_r52_full, _, _, _, _, chunk_files_r52_test = load_pass1_session(TENSOR_DIR_R52, "test_r52")

    mask_dev_52 = y_dev_52_full != 4
    X_dev_52 = X_dev_52[mask_dev_52]
    y_dev_52 = y_dev_52_full[mask_dev_52]
    uid_dev_52 = uid_dev_52[mask_dev_52]
    w_dev_52 = w_dev_52[mask_dev_52]
    y_r52 = y_r52_full[y_r52_full != 4]

    max_id_r42 = max(user_mapping_dev.values()) if user_mapping_dev else 0
    OFFSET = max_id_r42 + 1
    print(f" -> Tự động tính OFFSET = {OFFSET}")

    uid_dev_52 = uid_dev_52 + OFFSET 
    user_lookup_52_dev = {k + OFFSET: v for k, v in user_lookup_52_dev.items()}
    user_lookup_52_test = {k + OFFSET: v for k, v in user_lookup_52_test.items()}

    X_mix_sess_full = np.concatenate([X_dev_42, X_dev_52], axis=0)
    y_mix_full = np.concatenate([y_dev_42, y_dev_52], axis=0)
    uid_mix_full = np.concatenate([uid_dev_42, uid_dev_52], axis=0)
    w_mix_full = np.concatenate([w_dev_42, w_dev_52], axis=0)
    
    global_user_lookup = {**user_lookup_42_dev, **user_lookup_52_dev, **user_lookup_52_test}
    del X_dev_52, X_dev_42; gc.collect()

    print("\n -> Đang dò tìm Tọa độ 70 cột Fuzzy trên Ma trận 2D Flatten...")
    df_tmp = pd.read_parquet(PATH_SESSION)
    df_tmp = df_tmp.sort_values(by=['user', 'starttime']).reset_index(drop=True)
    df_dev_tmp = df_tmp.iloc[:int(len(df_tmp) * 0.80)]
    
    exclude_cols = ['insider', 'starttime', 'endtime', 'sessionid', 'user', 'day', 'week']
    feature_cols_tmp = [col for col in df_dev_tmp.columns if col not in exclude_cols]
    dead_cols = [c for c in feature_cols_tmp if df_dev_tmp[c].nunique() <= 1]
    valid_original_cols = [c for c in feature_cols_tmp if c not in dead_cols]
    
    virtual_cols = []
    if 'duration' in df_tmp.columns: virtual_cols.append('duration_quantile')
    virtual_cols.extend(['http_ratio', 'email_ratio', 'file_ratio', 'usb_ratio', 'http_leakf_ratio', 'http_jobf_ratio', 'http_hackf_ratio'])
    virtual_cols.extend(['n_http_zscore', 'n_email_zscore', 'n_file_zscore', 'n_usb_zscore', 'http_n_jobf_zscore', 'http_n_hackf_zscore'])
    
    session_feature_cols = valid_original_cols + virtual_cols
    known_dead_cols = ['http_n_jobf_zscore', 'n_usb_zscore', 'http_n_hackf_zscore']
    session_feature_cols = [c for c in session_feature_cols if c not in known_dead_cols]
    
    if len(session_feature_cols) < F_sess_actual: session_feature_cols.extend([f"Unknown_{i}" for i in range(F_sess_actual - len(session_feature_cols))])
    elif len(session_feature_cols) > F_sess_actual: session_feature_cols = session_feature_cols[:F_sess_actual]
    del df_tmp, df_dev_tmp; gc.collect()

    flat_fuzzy_indices = []
    sess_match = 0; user_match = 0
    
    for col in FUZZY_SELECTED_FEATURES:
        if col in session_feature_cols:
            idx = session_feature_cols.index(col)
            if idx < F_sess_actual:
                for t in range(5): flat_fuzzy_indices.append(t * F_sess_actual + idx)
                sess_match += 1
        elif col in feature_cols_ud:
            idx = feature_cols_ud.index(col)
            flat_fuzzy_indices.append((5 * F_sess_actual) + idx)
            user_match += 1

    TMP_RESULT_DIR = "./temp_results_lgbm_p4_fuzzy_70"
    os.makedirs(TMP_RESULT_DIR, exist_ok=True)
    
    time_folds = [
        {"train_weeks": list(range(0, 30)), "val_weeks": list(range(31, 35))},
        {"train_weeks": list(range(0, 35)), "val_weeks": list(range(36, 40))},
        {"train_weeks": list(range(0, 40)), "val_weeks": list(range(41, 45))},
        {"train_weeks": list(range(0, 45)), "val_weeks": list(range(46, 50))},
        {"train_weeks": list(range(0, 50)), "val_weeks": list(range(51, 56))},
    ]
    
    for fold_idx, fold_cfg in enumerate(time_folds):
        p = mp.Process(
            target=train_single_fold_lightgbm, 
            args=(fold_idx, fold_cfg["train_weeks"], fold_cfg["val_weeks"], 
                  X_mix_sess_full, y_mix_full, uid_mix_full, w_mix_full, 
                  global_user_lookup, F_user, 
                  chunk_files_r52_test, OFFSET, flat_fuzzy_indices, TMP_RESULT_DIR)
        )
        p.start(); p.join()

    print("\n" + "="*75)
    print("📥 ĐANG THU THẬP KẾT QUẢ ĐÁNH GIÁ VÀ LỊCH SỬ HUẤN LUYỆN...")
    val_mix_probs_list, val_mix_y_list, training_histories = [], [], []
    r52_probs_accum = np.zeros((len(y_r52), 4))  

    for fold_idx in range(len(time_folds)):
        file_path = os.path.join(TMP_RESULT_DIR, f"fold_{fold_idx}.npz")
        hist_path = os.path.join(TMP_RESULT_DIR, f"history_{fold_idx}.pkl")
        
        if os.path.exists(file_path):
            with np.load(file_path) as data:
                val_mix_probs_list.append(data['val_probs']); val_mix_y_list.append(data['y_val'])
                r52_probs_accum += (data['r52_probs'] / len(time_folds))
                
        if os.path.exists(hist_path):
            with open(hist_path, 'rb') as f:
                training_histories.append(pickle.load(f))

    val_mix_probs_all = np.vstack(val_mix_probs_list)
    val_mix_y_all = np.concatenate(val_mix_y_list)

    OFFLINE_SAVE_PATH = '/kaggle/working/pipeline4_lgbm_fuzzy70_valmix_test_r52.npz'
    np.savez_compressed(
        OFFLINE_SAVE_PATH,
        oof_probs_all=val_mix_probs_all,
        oof_y_all=val_mix_y_all,
        r52_probs_accum=r52_probs_accum,
        y_r52=y_r52
    )
    print(f"📦 ĐÃ LƯU TOÀN BỘ KẾT QUẢ DỰ ĐOÁN TẠI: {OFFLINE_SAVE_PATH}")

    # ==============================================================================
    # 💾 ĐÁNH GIÁ TÌM NGƯỠNG TRÊN TẬP VAL MIX (BẰNG TỐI ƯU HÓA F1-SCORE)
    # ==============================================================================
    oof_benign_probs = val_mix_probs_all[(val_mix_y_all == 0)]
    THRESH_S1 = np.percentile(oof_benign_probs[:, 1], 99.995)
    THRESH_S2 = np.percentile(oof_benign_probs[:, 2], 99.5)  
    THRESH_S3 = np.percentile(oof_benign_probs[:, 3], 99.99)

    print(f"   - Scen1: {THRESH_S1:.2f}")
    print(f"   - Scen2: {THRESH_S2:.2f}")
    print(f"   - Scen3: {THRESH_S3:.2f}")
    
    # ==============================================================================
    # 🌍 ÁP DỤNG NGƯỠNG CHO TEST R5.2 
    # ==============================================================================
    r52_preds = np.zeros(len(r52_probs_accum), dtype=int)
    for i in range(len(r52_probs_accum)):
        candidates = []
        if r52_probs_accum[i, 1] >= THRESH_S1: candidates.append((1, r52_probs_accum[i, 1]))
        if r52_probs_accum[i, 2] >= THRESH_S2: candidates.append((2, r52_probs_accum[i, 2]))
        if r52_probs_accum[i, 3] >= THRESH_S3: candidates.append((3, r52_probs_accum[i, 3]))
        if candidates: r52_preds[i] = max(candidates, key=lambda item: item[1])[0]
    
    r52_y_ohe = label_binarize(y_r52, classes=[0, 1, 2, 3])
    r52_auc = roc_auc_score(r52_y_ohe, r52_probs_accum, multi_class='ovr', average='macro')
    print(f"\n🔥 MACRO AUC TẬP TEST R5.2: {r52_auc:.4f} 🔥\n")
    print(classification_report(y_r52, r52_preds, target_names=classes_names, digits=4, zero_division=0))
    conf_matrix_r52 = confusion_matrix(y_r52, r52_preds)

    # ==============================================================================
    # 🎨 XUẤT CÁC BIỂU ĐỒ TRỰC QUAN
    # ==============================================================================
    def plot_confusion_matrix_percent(cm, classes, title='Confusion Matrix'):
        plt.figure(figsize=(8, 6))
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        row_sums_safe = np.where(row_sums == 0, 1, row_sums) 
        cm_percentages = cm.astype('float') / row_sums_safe
        sns.heatmap(cm_percentages, annot=True, fmt='.2%', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(title, fontsize=15, fontweight='bold', pad=15)
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        plt.show()

    print("\n -> Đang xuất biểu đồ Confusion Matrix cho Test R5.2...")
    plot_confusion_matrix_percent(conf_matrix_r52, classes_names, title="Ma Trận Nhầm Lẫn Test R5.2 (LGBM Fuzzy Mix)")

    def plot_training_history_average(histories):
        if not histories: return
        max_epochs = max([len(h['loss']) for h in histories])
        print(f" -> Đang tính trung bình 5 Fold (Kéo giãn các Fold ngắn cho bằng {max_epochs} Iterations)...")
        
        avg_history = {}
        for key in histories[0].keys():
            padded_folds = []
            for h in histories:
                arr = h[key]
                pad_length = max_epochs - len(arr)
                padded_arr = np.pad(arr, (0, pad_length), mode='edge')
                padded_folds.append(padded_arr)
            avg_history[key] = np.mean(padded_folds, axis=0)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(avg_history['loss'], label='Train Loss', color='blue', linewidth=2)
        axes[0].plot(avg_history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
        axes[0].set_xlabel('Boosting Rounds')
        axes[0].set_ylabel('Logloss')
        axes[0].set_title('Loss Curve', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, linestyle=':', alpha=0.6)
        
        if 'accuracy' in avg_history:
            axes[1].plot(avg_history['accuracy'], label='Train Accuracy', color='blue', linewidth=2)
            axes[1].plot(avg_history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
            axes[1].set_xlabel('Boosting Rounds')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Accuracy Curve', fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, linestyle=':', alpha=0.6)
            
        plt.tight_layout()
        plt.show()

    print(" -> Đang xuất biểu đồ Loss & Accuracy nằm ngang...")
    plot_training_history_average(training_histories)

    def plot_multiclass_roc(y_true, y_probs, n_classes, classes_names, title):
        y_test_ohe = label_binarize(y_true, classes=range(n_classes))
        plt.figure(figsize=(9, 7))
        colors = ['dodgerblue', 'crimson', 'forestgreen', 'darkorange']
        
        for i in range(n_classes):
            if i < y_probs.shape[1]:
                fpr, tpr, _ = roc_curve(y_test_ohe[:, i], y_probs[:, i])
                roc_auc = auc(fpr, tpr)
                if not np.isnan(roc_auc):
                    plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                             label=f'ROC curve - {classes_names[i]} (AUC = {roc_auc:.4f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=15, fontweight='bold', pad=15)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

    print(" -> Đang xuất biểu đồ ROC cho Test R5.2...")
    plot_multiclass_roc(y_r52, r52_probs_accum, 4, classes_names, title="Biểu đồ ROC Test R5.2 (LGBM Fuzzy Mix)")

    total_execution_time = (time.time() - global_start_time) / 60.0
    print("\n" + "="*75)
    print(f"🏁 TOÀN BỘ PIPELINE HOÀN TẤT TRONG: {total_execution_time:.2f} PHÚT.")
    print("="*75)