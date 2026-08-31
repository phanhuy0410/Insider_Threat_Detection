import pandas as pd
import gc
import os, re
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import CategoricalFocalCrossentropy
import pickle
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.regularizers import l2
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize

# ------------------------------------------------------------------------------
# 1. CỐ ĐỊNH SEED VÀ CẤU HÌNH ĐƯỜNG DẪN
# ------------------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
print(f"🔒 Hệ thống đã khóa cứng tính ngẫu nhiên với SEED = {SEED}")

TENSOR_DIR = "/kaggle/input/datasets/doanthimo80/tensor-data-static"
PATH_SESSION = '/kaggle/input/datasets/phanthanhhoang/cert-r42-session/session_r4.2.parquet'

TENSOR_DIR_R52 = "/kaggle/input/datasets/doanthimo80/tensor-data-r52-new" 
PATH_SESSION_R52 = '/kaggle/input/datasets/phanthanhhoang/r5-2-session/sessionr5.2.csv'
PATH_USERDAY_R52_TEST = f"{TENSOR_DIR_R52}/userday_clean_test_r52.parquet"

classes_names = ['Benign', 'Scen1', 'Scen2', 'Scen3']

sample_data = np.load(f"{TENSOR_DIR}/dev_chunk_0.npz")
_, T_sess, F_sess = sample_data['X_sess'].shape
del sample_data; gc.collect()

# ==============================================================================
# 🎛️ BẢNG ĐIỀU KHIỂN THÍ NGHIỆM ABLATION
# ==============================================================================
EXPERIMENT_MODE = "ALL"  # Đổi thành "ALL" nếu muốn mờ toàn bộ

# --- DANH SÁCH 70 CỘT ---
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
df_sess_rec = pd.read_parquet(PATH_SESSION, columns=['user', 'starttime'])
test_idx_rec = int(len(df_sess_rec) * 0.80)
unique_users_dev = sorted(df_sess_rec.iloc[:test_idx_rec]['user'].unique().tolist())
user_mapping_dev = {u: i for i, u in enumerate(unique_users_dev)}
del df_sess_rec; gc.collect()

df_sess_r52 = pd.read_csv(PATH_SESSION_R52, usecols=['user', 'starttime', 'insider'])
test_idx_r52 = int(len(df_sess_r52) * 0.80)
df_sess_dev_r52 = df_sess_r52.iloc[:test_idx_r52]
df_sess_test_raw_r52 = df_sess_r52.iloc[test_idx_r52:]

known_insiders_dev_r52 = set(df_sess_dev_r52[df_sess_dev_r52['insider'] != 0]['user'])
df_sess_test_r52 = df_sess_test_raw_r52[~df_sess_test_raw_r52['user'].isin(known_insiders_dev_r52)]

user_mapping_test_r52 = {u: i for i, u in enumerate(sorted(df_sess_test_r52['user'].unique().tolist()))}
del df_sess_r52, df_sess_dev_r52, df_sess_test_raw_r52, df_sess_test_r52; gc.collect()

# ==============================================================================
# 🛡️ CÁC CLASS GENERATOR VÀ HÀM ĐỌC DỮ LIỆU
# ==============================================================================
class BalancedUniversalGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, X_cat=None, batch_size=256, shuffle=True, hacker_ratio=0.25, **kwargs):
        super().__init__(**kwargs)
        self.X = X; self.y = y; self.X_cat = X_cat; self.batch_size = batch_size; self.shuffle = shuffle; self.hacker_ratio = hacker_ratio 
        self._balance_indices()
    def _balance_indices(self):
        y_labels = np.argmax(self.y, axis=1)
        idx_benign = np.where(y_labels == 0)[0]
        idx_hacker = np.where(y_labels != 0)[0]
        if len(idx_hacker) == 0: self.indices = np.arange(len(self.y))
        else:
            target_hacker_count = int(len(idx_benign) * self.hacker_ratio)
            hacker_oversampled = np.random.choice(idx_hacker, target_hacker_count, replace=True)
            self.indices = np.concatenate([idx_benign, hacker_oversampled])
        if self.shuffle: np.random.shuffle(self.indices)
    def __len__(self): return int(np.ceil(len(self.indices) / self.batch_size))
    def __getitem__(self, index):
        batch_idx = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        batch_X = np.concatenate([self.X[batch_idx], self.X_cat[batch_idx]], axis=-1) if self.X_cat is not None else self.X[batch_idx]
        return batch_X, self.y[batch_idx]
    def on_epoch_end(self):
        if self.shuffle: self._balance_indices()

class UniversalGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, X_cat=None, batch_size=256, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.X = X; self.y = y; self.X_cat = X_cat; self.batch_size = batch_size; self.shuffle = shuffle
        self.indices = np.arange(len(self.y))
        if self.shuffle: np.random.shuffle(self.indices)
    def __len__(self): return int(np.ceil(len(self.y) / self.batch_size))
    def __getitem__(self, index):
        batch_idx = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        batch_X = np.concatenate([self.X[batch_idx], self.X_cat[batch_idx]], axis=-1) if self.X_cat is not None else self.X[batch_idx]
        return batch_X, self.y[batch_idx]
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indices)

def load_pass1_session(chunk_dir, prefix):
    X_s, y, uid, w_list, day_list = [], [], [], [], []
    files = [os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.startswith(f"{prefix}_chunk")]
    files.sort(key=lambda x: int(re.search(r'chunk_(\d+)', x).group(1)))
    for f in files:
        with np.load(f) as data:
            if len(data['y']) == 0: continue
            X_s.append(data['X_sess']); y.append(data['y']); uid.append(data['user_id'])
            w_list.append(data['w']); day_list.append(data['day'])
    return np.concatenate(X_s), np.concatenate(y), np.concatenate(uid), np.concatenate(w_list), np.concatenate(day_list)

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

# ==============================================================================
# 🧠 LỚP HỌC MỜ THÍCH NGHI (TENSORFLOW / KERAS VERSION)
# ==============================================================================
def compute_fuzzy_init_numpy(X, fuzzy_indices, num_rules=3):
    if num_rules != 3:
        raise ValueError("Kiến trúc phân vị Q25, Q50, Q75 yêu cầu bắt buộc num_rules=3")
    if fuzzy_indices is not None and len(fuzzy_indices) > 0:
        X_fuz = X[..., fuzzy_indices]
    else: 
        X_fuz = X
    fuz_dim = X_fuz.shape[-1]
    centers_init = np.zeros((fuz_dim, 3), dtype=np.float32)
    sigmas_init = np.zeros((fuz_dim, 3), dtype=np.float32)
    for i in range(fuz_dim):
        col_data = X_fuz[..., i].flatten()
        q25, q50, q75 = np.percentile(col_data, [25, 50, 75])
        centers_init[i] = [q25, q50, q75]
        # Hệ số 1.1774 = sqrt(2 * ln(2)), chuẩn của Gaussian Fuzzy Set
        s1 = max((q50 - q25) / 1.1774, 1e-3) 
        s2 = max((q75 - q25) / (2 * 1.1774), 1e-3) 
        s3 = max((q75 - q50) / 1.1774, 1e-3)
        
        sigmas_init[i] = [s1, s2, s3]
    return centers_init, sigmas_init

class FuzzyMembershipLayer(Layer):
    def __init__(self, centers_init, sigmas_init, fuzzy_indices=None, concat_original=True, **kwargs):
        super(FuzzyMembershipLayer, self).__init__(**kwargs)
        self.fuzzy_indices = fuzzy_indices  
        self.concat_original = concat_original
        self.centers_init = centers_init
        self.sigmas_init = sigmas_init
        self.num_rules = centers_init.shape[-1] 

    def build(self, input_shape):
        if self.fuzzy_indices is not None and len(self.fuzzy_indices) > 0: 
            self.fuz_dim = len(self.fuzzy_indices)
        else: 
            self.fuz_dim = input_shape[-1]
            
        self.centers = self.add_weight(name='fuzzy_centers', shape=(self.fuz_dim, self.num_rules),
                                       initializer=tf.keras.initializers.Constant(self.centers_init), 
                                       trainable=False)
        self.sigmas = self.add_weight(name='fuzzy_sigmas', shape=(self.fuz_dim, self.num_rules),
                                      initializer=tf.keras.initializers.Constant(self.sigmas_init), 
                                      trainable=False)
        super(FuzzyMembershipLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        if self.fuzzy_indices is not None and len(self.fuzzy_indices) > 0:
            x_fuz = tf.gather(inputs, self.fuzzy_indices, axis=-1)
        else: 
            x_fuz = inputs
        x_expanded = tf.expand_dims(x_fuz, axis=-1)
        safe_sigmas = tf.maximum(self.sigmas, 1e-5)      
        c = self.centers
        s = safe_sigmas
        for _ in range(len(inputs.shape) - 1):
            c = tf.expand_dims(c, axis=0); s = tf.expand_dims(s, axis=0)

        diff = (x_expanded - c) / s
        memberships = tf.exp(-0.5 * tf.square(diff))
        
        input_shape_tf = tf.shape(inputs)
        if len(inputs.shape) == 3: 
            new_shape = [input_shape_tf[0], input_shape_tf[1], self.fuz_dim * self.num_rules]
        else:                      
            new_shape = [input_shape_tf[0], self.fuz_dim * self.num_rules]
            
        reshaped = tf.reshape(memberships, new_shape)
        if self.concat_original: 
            return tf.concat([inputs, reshaped], axis=-1)
        return reshaped

class GatedTemporalAttention(Layer):
    def __init__(self, **kwargs): super(GatedTemporalAttention, self).__init__(**kwargs)
    def build(self, input_shape):
        dim = input_shape[-1]
        self.W_v = self.add_weight(name='W_v', shape=(dim, dim), initializer='glorot_uniform', trainable=True)
        self.W_u = self.add_weight(name='W_u', shape=(dim, dim), initializer='glorot_uniform', trainable=True)
        self.w = self.add_weight(name='w', shape=(dim, 1), initializer='glorot_uniform', trainable=True)
        super(GatedTemporalAttention, self).build(input_shape)
    def call(self, x):
        v = K.tanh(K.dot(x, self.W_v)) 
        u = K.sigmoid(K.dot(x, self.W_u)) 
        gated_x = v * u 
        e = K.dot(gated_x, self.w)
        alpha = K.softmax(e, axis=1)
        return K.sum(x * alpha, axis=1)

def tcn_residual_block(x, n_filters, kernel_size, dilation_rate, dropout_rate, block_name):
    conv = Conv1D(filters=n_filters, 
                  kernel_size=kernel_size, 
                  dilation_rate=dilation_rate, 
                  padding='causal',
                  kernel_initializer='he_normal',
                  name=f"{block_name}_conv")(x)
    
    bn = LayerNormalization(name=f"{block_name}_bn")(conv)
    act = Activation('relu', name=f"{block_name}_act")(bn)
    drop = SpatialDropout1D(dropout_rate, name=f"{block_name}_drop")(act)
    
    if x.shape[-1] != n_filters:
        shortcut = Conv1D(filters=n_filters, kernel_size=1, padding='same', name=f"{block_name}_shortcut")(x)
    else:
        shortcut = x
    res = Add(name=f"{block_name}_add")([drop, shortcut])
    return Activation('relu', name=f"{block_name}_res_out")(res)
    
# ==============================================================================
# 🤖 3 KIẾN TRÚC MẠNG ENCODER CHÍNH
# ==============================================================================
def build_session_encoder(T_sess, F_sess, flat_fuzzy_indices, centers_init, sigmas_init, learning_rate=0.0001):
    inp = Input(shape=(T_sess, F_sess), name='Input_Session')
    
    x = FuzzyMembershipLayer(centers_init=centers_init, sigmas_init=sigmas_init, 
                             fuzzy_indices=flat_fuzzy_indices, concat_original=True, name='Session_Fuzzy')(inp)
    
    x = Dense(192, activation='relu')(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = tcn_residual_block(x, n_filters=64, kernel_size=3, dilation_rate=1, dropout_rate=0.3, block_name="Sess_TCN_D1")
    x = GatedTemporalAttention()(x)
    latent = Dense(64, activation='relu')(x) 
    latent_bn = BatchNormalization()(latent) 
    out = Dense(4, activation='softmax')(latent_bn)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss=CategoricalFocalCrossentropy(gamma=2.0), 
                  metrics=['accuracy'])
    return model

def build_user_encoder(F_user, flat_fuzzy_indices, centers_init, sigmas_init, learning_rate=0.0001):
    inp = Input(shape=(F_user,), name='Input_User')
    
    x = FuzzyMembershipLayer(centers_init=centers_init, sigmas_init=sigmas_init, 
                             fuzzy_indices=flat_fuzzy_indices, concat_original=True, name='User_Fuzzy')(inp)
    
    x = Reshape((-1, 1))(x) 
    x = Dense(32, activation='relu')(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = tcn_residual_block(x, n_filters=32, kernel_size=3, dilation_rate=1, dropout_rate=0.2, block_name="User_TCN_D1")
    x = GatedTemporalAttention()(x)
    latent = Dense(32, activation='relu')(x)
    latent_bn = BatchNormalization()(latent) 
    out = Dense(4, activation='softmax')(latent_bn)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss=CategoricalFocalCrossentropy(gamma=2.0), 
                  metrics=['accuracy'])
    return model

def build_score_level_fusion(fused_dim=8):
    inp = Input(shape=(fused_dim,), name='Input_8D_Scores')
    x = Dense(fused_dim * 2, activation='relu')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(fused_dim, activation='relu')(x)
    x_added = add([inp, x]) # Residual Block
    x_final = Dense(16, activation='relu', kernel_regularizer=l2(1e-3))(x_added)
    x_final = BatchNormalization()(x_final)
    out = Dense(4, activation='softmax')(x_final)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss=CategoricalFocalCrossentropy(gamma=2.0), 
                  metrics=['accuracy'])
    return model

# ==============================================================================
# 🌀 PIPELINE HUẤN LUYỆN
# ==============================================================================
def train_single_fold(fold_idx, train_weeks, val_weeks, X_dev_sess_full, y_dev_full, uid_dev_full, w_dev_full, 
                      global_user_lookup, F_user, 
                      chunk_files_r52_test, OFFSET,
                      sess_fuzzy_indices, user_fuzzy_indices, TMP_RESULT_DIR): 
    
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    print(f"\n{'='*20} KHỞI ĐỘNG TIẾN TRÌNH FOLD {fold_idx + 1} {'='*20}")
    
    train_mask = np.isin(w_dev_full, train_weeks)
    idx_train_raw = np.where(train_mask)[0]
    val_mask = np.isin(w_dev_full, val_weeks)
    idx_val = np.where(val_mask)[0]
    if len(idx_val) == 0: return
    
    # --- KMEANS DOWNSAMPLING ---
    X_tr_sess_flat = X_dev_sess_full[idx_train_raw].reshape(len(idx_train_raw), -1)
    np.nan_to_num(X_tr_sess_flat, nan=0.0, posinf=65000.0, neginf=-65000.0, copy=False)
    y_train_temp = y_dev_full[idx_train_raw]
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
    
    X_train_sess = X_dev_sess_full[keep_global_idx]
    y_train = y_dev_full[keep_global_idx]
    uid_train = uid_dev_full[keep_global_idx]
    
    X_val_sess = X_dev_sess_full[idx_val]
    y_val = y_dev_full[idx_val]
    uid_val = uid_dev_full[idx_val]
    
    X_train_user = np.array([global_user_lookup.get(u, np.zeros(F_user)) for u in uid_train], dtype=np.float32)
    X_val_user = np.array([global_user_lookup.get(u, np.zeros(F_user)) for u in uid_val], dtype=np.float32)
    
    # --- CHUẨN HÓA DỮ LIỆU ---
    np.nan_to_num(X_train_sess, nan=0.0, posinf=65000.0, neginf=-65000.0, copy=False)
    np.nan_to_num(X_val_sess, nan=0.0, posinf=65000.0, neginf=-65000.0, copy=False)
    scaler_sess = StandardScaler(copy=False)
    X_train_sess = scaler_sess.fit_transform(X_train_sess.reshape(-1, F_sess)).astype(np.float32, copy=False).reshape(-1, T_sess, F_sess)
    X_val_sess = scaler_sess.transform(X_val_sess.reshape(-1, F_sess)).astype(np.float32, copy=False).reshape(-1, T_sess, F_sess)

    np.nan_to_num(X_train_user, nan=0.0, posinf=65000.0, neginf=-65000.0, copy=False)
    np.nan_to_num(X_val_user, nan=0.0, posinf=65000.0, neginf=-65000.0, copy=False)
    scaler_user = StandardScaler(copy=False)
    X_train_user = scaler_user.fit_transform(X_train_user).astype(np.float32, copy=False)
    X_val_user = scaler_user.transform(X_val_user).astype(np.float32, copy=False)
    
    y_train_ohe = tf.keras.utils.to_categorical(y_train, num_classes=4)
    y_val_ohe = tf.keras.utils.to_categorical(y_val, num_classes=4)
    tf.keras.backend.clear_session()

    # 🔥 BƯỚC KHỞI TẠO TỌA ĐỘ MỜ BẰNG NUMPY
    sess_c_init, sess_s_init = compute_fuzzy_init_numpy(X_train_sess, sess_fuzzy_indices, num_rules=3)
    user_c_init, user_s_init = compute_fuzzy_init_numpy(X_train_user, user_fuzzy_indices, num_rules=3)
    
    # --- HUẤN LUYỆN ENCODERS ---
    session_encoder = build_session_encoder(T_sess, F_sess, sess_fuzzy_indices, sess_c_init, sess_s_init, learning_rate=0.0001)
    user_encoder = build_user_encoder(F_user, user_fuzzy_indices, user_c_init, user_s_init, learning_rate=0.0001)
    classifier = build_score_level_fusion(fused_dim=8) 
    
    es_sess = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    es_user = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    es_clf = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    lr_sess = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=0)
    lr_user = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=0)
    rlr_clf = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
    
    gen_train_sess = BalancedUniversalGenerator(X_train_sess, y_train_ohe, hacker_ratio=0.25)
    gen_val_sess = UniversalGenerator(X_val_sess, y_val_ohe, shuffle=False)
    
    start_time_sess = time.time()
    session_encoder.fit(gen_train_sess, validation_data=gen_val_sess, epochs=100, callbacks=[es_sess, lr_sess], verbose=0)
    duration_sess = time.time() - start_time_sess
    del gen_train_sess, gen_val_sess; gc.collect()

    gen_train_user = BalancedUniversalGenerator(X_train_user, y_train_ohe, hacker_ratio=0.25)
    gen_val_user = UniversalGenerator(X_val_user, y_val_ohe, shuffle=False)
    
    start_time_user = time.time()
    user_encoder.fit(gen_train_user, validation_data=gen_val_user, epochs=100, callbacks=[es_user, lr_user], verbose=0)
    duration_user = time.time() - start_time_user
    del gen_train_user, gen_val_user; gc.collect()

    # --- RÚT XÁC SUẤT ĐỂ FUSION (SỬ DỤNG PREDICT NHANH) ---
    prob_train_sess = session_encoder.predict(X_train_sess, batch_size=1024, verbose=0)
    prob_train_user = user_encoder.predict(X_train_user, batch_size=1024, verbose=0)
    train_scores_raw = np.hstack([prob_train_sess, prob_train_user]).astype(np.float32, copy=False)
    del X_train_sess, X_train_user, prob_train_sess, prob_train_user; gc.collect() 

    prob_val_sess = session_encoder.predict(X_val_sess, batch_size=1024, verbose=0)
    prob_val_user = user_encoder.predict(X_val_user, batch_size=1024, verbose=0)
    val_scores_raw = np.hstack([prob_val_sess, prob_val_user]).astype(np.float32, copy=False)
    del X_val_sess, X_val_user, prob_val_sess, prob_val_user; gc.collect()

    smote = SMOTE(sampling_strategy={1: 10000, 2: 10000, 3: 10000}, random_state=SEED, k_neighbors=5)
    train_scores_bal, y_train_bal = smote.fit_resample(train_scores_raw, y_train)
    y_train_bal_ohe = tf.keras.utils.to_categorical(y_train_bal, 4)
    del train_scores_raw; gc.collect()
    
    gen_train_clf = UniversalGenerator(train_scores_bal, y_train_bal_ohe, batch_size=256, shuffle=True)
    gen_val_clf = UniversalGenerator(val_scores_raw, y_val_ohe, batch_size=256, shuffle=False)
    
    hist_clf = classifier.fit(gen_train_clf, validation_data=gen_val_clf, epochs=100, callbacks=[es_clf, rlr_clf], verbose=0)
    clf_history = hist_clf.history.copy()
    del gen_train_clf, gen_val_clf, hist_clf; gc.collect()
    
    fold_train_time_minutes = (duration_sess + duration_user) / 60.0
    print(f" ⏱️ THỜI GIAN TRAIN FOLD {fold_idx + 1}: {fold_train_time_minutes:.2f} phút")

    val_probs = classifier.predict(val_scores_raw, batch_size=1024, verbose=0)
    fold_auc = roc_auc_score(y_val_ohe, val_probs, multi_class='ovr', average='macro')
    print(f" 🔥 MACRO AUC FOLD {fold_idx + 1} (VAL R4.2): {fold_auc:.4f}")

    # ---------------------------------------------------------
    # 🎯 CHỈ INFERENCE TẬP TEST R5.2 
    # ---------------------------------------------------------
    prob_r52_sess_list, prob_r52_user_list = [], []
    for f in chunk_files_r52_test: 
        with np.load(f) as data:
            if len(data['y']) == 0: continue
            mask = data['y'] != 4
            if not np.any(mask): continue
            
            batch_sess = data['X_sess'][mask]
            batch_uid = data['user_id'][mask] + OFFSET

        batch_sess = np.nan_to_num(batch_sess, nan=0.0, posinf=65000.0, neginf=-65000.0, copy=False)
        batch_sess_s = scaler_sess.transform(batch_sess.reshape(-1, F_sess)).astype(np.float32, copy=False).reshape(-1, T_sess, F_sess)
        prob_r52_sess_list.append(session_encoder.predict(batch_sess_s, batch_size=1024, verbose=0))
        
        batch_user_raw = np.array([global_user_lookup.get(u, np.zeros(F_user)) for u in batch_uid], dtype=np.float32)
        batch_user_raw = np.nan_to_num(batch_user_raw, nan=0.0, posinf=65000.0, neginf=-65000.0, copy=False)
        batch_user_s = scaler_user.transform(batch_user_raw).astype(np.float32, copy=False)
        prob_r52_user_list.append(user_encoder.predict(batch_user_s, batch_size=1024, verbose=0))

    r52_scores_raw = np.hstack([np.concatenate(prob_r52_sess_list, axis=0), np.concatenate(prob_r52_user_list, axis=0)]).astype(np.float32, copy=False)
    r52_probs_fold = classifier.predict(r52_scores_raw, batch_size=1024, verbose=0)
    del prob_r52_sess_list, prob_r52_user_list, r52_scores_raw; gc.collect()

    np.savez(
        os.path.join(TMP_RESULT_DIR, f"fold_{fold_idx}.npz"),
        val_probs = val_probs,        
        y_val = y_val,                
        r52_probs = r52_probs_fold, 
        fold_auc = np.array([fold_auc])
    )
    with open(os.path.join(TMP_RESULT_DIR, f"history_{fold_idx}.pkl"), 'wb') as f: pickle.dump(clf_history, f)
        
    print(f" ✅ TIẾN TRÌNH FOLD {fold_idx + 1} HOÀN TẤT!")
    return

# ==============================================================================
# 🚀 VÙNG AN TOÀN MAIN PROCESS
# ==============================================================================
if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    print("\n" + "="*75)
    print(f"🚀 PIPELINE DL: TRAIN R4.2 | VAL R4.2 | TEST R5.2 | MỨC CỘT: {EXPERIMENT_MODE}")
    print("="*75)
    
    global_start_time = time.time()
    
    # 1. NẠP DỮ LIỆU DEV R4.2 ĐỂ TRAIN/VAL
    X_dev_42, y_dev_42, uid_dev_42, w_dev_42, _ = load_pass1_session(TENSOR_DIR, "dev")
    user_lookup_42_dev, user_feature_cols = load_userday_lookup(f"{TENSOR_DIR}/userday_clean_dev.parquet", user_mapping_dev)
    F_user = len(user_feature_cols) 
    
    # 2. NẠP DỮ LIỆU R5.2 LÀM TEST
    user_lookup_52_test, _ = load_userday_lookup(PATH_USERDAY_R52_TEST, user_mapping_test_r52)
    _, y_r52_full, _, _, _ = load_pass1_session(TENSOR_DIR_R52, "test_r52")
    y_r52 = y_r52_full[y_r52_full != 4] 

    max_id_r42 = max(user_mapping_dev.values()) if user_mapping_dev else 0
    OFFSET = max_id_r42 + 1
    print(f" -> Tự động tính OFFSET = {OFFSET} (ID R5.2 sẽ nối tiếp ngay sau R4.2)")
    uid_dev_52 = uid_dev_52 + OFFSET 
    user_lookup_52_dev = {k + OFFSET: v for k, v in user_lookup_52_dev.items()}
    user_lookup_52_test = {k + OFFSET: v for k, v in user_lookup_52_test.items()}
    
    global_user_lookup = {**user_lookup_42_dev, **user_lookup_52_dev, **user_lookup_52_test}

    # ==========================================================================
    # 🔥 LỌC TỌA ĐỘ CHO FUZZY LAYER THEO KIẾN TRÚC TÁCH NHÁNH DL
    # ==========================================================================
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
    
    if len(session_feature_cols) < F_sess: session_feature_cols.extend([f"Unknown_{i}" for i in range(F_sess - len(session_feature_cols))])
    elif len(session_feature_cols) > F_sess: session_feature_cols = session_feature_cols[:F_sess]
    del df_tmp, df_dev_tmp; gc.collect()
    
    sess_fuzzy_indices = []
    user_fuzzy_indices = []
    
    if EXPERIMENT_MODE == "ALL":
        print(f"🌟 ĐANG CHẠY CHẾ ĐỘ FULL: Mờ toàn bộ {F_sess} Session và {F_user} User!")
        sess_fuzzy_indices = list(range(F_sess))
        user_fuzzy_indices = list(range(F_user))
    else:
        for col in FUZZY_SELECTED_FEATURES:
            if col in session_feature_cols:
                idx = session_feature_cols.index(col)
                if idx < F_sess: sess_fuzzy_indices.append(idx)
            elif col in user_feature_cols:
                idx = user_feature_cols.index(col)
                user_fuzzy_indices.append(idx)
        print(f"🌟 ÁNH XẠ THÀNH CÔNG: Lấy {len(sess_fuzzy_indices)} Session + {len(user_fuzzy_indices)} User để làm Mờ.")

    TMP_RESULT_DIR = f"./temp_results_dl_ablation_{EXPERIMENT_MODE.lower()}"
    os.makedirs(TMP_RESULT_DIR, exist_ok=True)
    
    time_folds = [
        {"train_weeks": list(range(0, 30)), "val_weeks": list(range(31, 35))},
        {"train_weeks": list(range(0, 35)), "val_weeks": list(range(36, 40))},
        {"train_weeks": list(range(0, 40)), "val_weeks": list(range(41, 45))},
        {"train_weeks": list(range(0, 45)), "val_weeks": list(range(46, 50))},
        {"train_weeks": list(range(0, 50)), "val_weeks": list(range(51, 56))},
    ]
    
    chunk_files_r52_test = [os.path.join(TENSOR_DIR_R52, f) for f in os.listdir(TENSOR_DIR_R52) if f.startswith("test_r52_chunk")]
    chunk_files_r52_test.sort(key=lambda x: int(re.search(r'chunk_(\d+)', x).group(1)))

    for fold_idx, fold_cfg in enumerate(time_folds):
        train_weeks = fold_cfg["train_weeks"]
        val_weeks = fold_cfg["val_weeks"]
        print(f"\n⏳ Đang nạp lệnh cho Fold {fold_idx + 1}")
        p = mp.Process(
            target=train_single_fold, 
            args=(fold_idx, train_weeks, val_weeks, 
                  X_dev_42, y_dev_42, uid_dev_42, w_dev_42, 
                  global_user_lookup, F_user, 
                  chunk_files_r52_test, OFFSET,
                  sess_fuzzy_indices, user_fuzzy_indices, TMP_RESULT_DIR)
        )
        p.start(); p.join()  
        if p.exitcode != 0: break 

    print("\n" + "="*75)
    print("📥 ĐANG THU THẬP KẾT QUẢ TRÊN TẤT CẢ TẬP DỮ LIỆU...")
    val_r42_probs_list, val_r42_y_list, fold_aucs, training_histories = [], [], [], []
    r52_probs_accum = np.zeros((len(y_r52), 4)) 

    for fold_idx in range(len(time_folds)):
        file_path = os.path.join(TMP_RESULT_DIR, f"fold_{fold_idx}.npz")
        hist_path = os.path.join(TMP_RESULT_DIR, f"history_{fold_idx}.pkl")
        if os.path.exists(file_path):
            with np.load(file_path) as data:
                val_r42_probs_list.append(data['val_probs']) 
                val_r42_y_list.append(data['y_val'])
                fold_aucs.append(data['fold_auc'][0])
                r52_probs_accum += (data['r52_probs'] / len(time_folds))
        if os.path.exists(hist_path):
            with open(hist_path, 'rb') as f:
                training_histories.append(pickle.load(f))

    print(f"🏆 TRUNG BÌNH MACRO AUC {len(fold_aucs)} FOLD (VAL R4.2): {np.mean(fold_aucs):.4f}")
    val_r42_probs_all = np.vstack(val_r42_probs_list)
    val_r42_y_all = np.concatenate(val_r42_y_list)

    # ==============================================================================
    # 💾 ĐÁNH GIÁ NGƯỠNG (THRESHOLD) TỪ VAL R4.2
    # ==============================================================================
    oof_benign_probs = val_r42_probs_all[(val_r42_y_all == 0)]
    THRESH_S1 = np.percentile(oof_benign_probs[:, 1], 99.999)
    THRESH_S2 = np.percentile(oof_benign_probs[:, 2], 99.999)  
    THRESH_S3 = np.percentile(oof_benign_probs[:, 3], 99.999)
    print(f"\n📏 Ngưỡng cố định lấy từ R4.2 Val: S1={THRESH_S1:.4f}, S2={THRESH_S2:.4f}, S3={THRESH_S3:.4f}")
    
    # ---- HÀM DÙNG CHUNG ĐỂ ÁP DỤNG NGƯỠNG ----
    def apply_thresholds(probs, t1, t2, t3):
        preds = np.zeros(len(probs), dtype=int)
        for i in range(len(probs)):
            cands = []
            if probs[i, 1] >= t1: cands.append((1, probs[i, 1]))
            if probs[i, 2] >= t2: cands.append((2, probs[i, 2]))
            if probs[i, 3] >= t3: cands.append((3, probs[i, 3]))
            if cands: preds[i] = max(cands, key=lambda item: item[1])[0]
        return preds
    
    # ==============================================================================
    # 🌍 KẾT QUẢ KIỂM CHÉO TRÊN TẬP DỮ LIỆU R5.2
    # ==============================================================================
    print("\n" + "*"*70)
    print("🌍 KẾT QUẢ KIỂM CHÉO TRÊN TẬP DỮ LIỆU R5.2 (CHỈ DÙNG NGƯỠNG R4.2)")
    print("*"*70)
    
    r52_preds = apply_thresholds(r52_probs_accum, THRESH_S1, THRESH_S2, THRESH_S3)
    
    r52_auc = roc_auc_score(tf.keras.utils.to_categorical(y_r52, 4), r52_probs_accum, multi_class='ovr', average='macro')
    print(f"\n🚀 MACRO AUC TẬP KIỂM CHÉO (R5.2): {r52_auc:.4f} 🚀\n")
    print(classification_report(y_r52, r52_preds, target_names=classes_names, digits=4, zero_division=0))
    
    # 🔥 ĐÃ SỬA LỖI Ở ĐÂY: Khai báo ma trận nhầm lẫn trước khi vẽ
    conf_matrix_r52 = confusion_matrix(y_r52, r52_preds)
    
    # ==============================================================================
    # 🎨 XUẤT CÁC BIỂU ĐỒ TRỰC QUAN
    # ==============================================================================
    def plot_confusion_matrix_percent(cm, classes, title='Confusion Matrix'):
        plt.figure(figsize=(8, 6))
        row_sums = np.where(cm.sum(axis=1) == 0, 1, cm.sum(axis=1))[:, np.newaxis] 
        cm_percentages = cm.astype('float') / row_sums
        sns.heatmap(cm_percentages, annot=True, fmt='.2%', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(title, fontsize=15, fontweight='bold', pad=15)
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    print(" -> Đang xuất biểu đồ Confusion Matrix cho R5.2...")
    plot_confusion_matrix_percent(conf_matrix_r52, classes_names, title="Ma Trận Nhầm Lẫn Test R5.2")
    
    def plot_training_history_average(histories):
        if not histories: return
        max_epochs = max([len(h['loss']) for h in histories])
        print(f" -> Đang tính trung bình {len(histories)} Fold (Kéo giãn các Fold ngắn cho bằng {max_epochs} Epochs)...")
        
        avg_history = {}
        for key in histories[0].keys():
            padded_folds = [np.pad(h[key], (0, max_epochs - len(h[key])), mode='edge') for h in histories]
            avg_history[key] = np.mean(padded_folds, axis=0)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(avg_history['loss'], label='Train Loss', color='blue', linewidth=2)
        axes[0].plot(avg_history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Learning Curve', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, linestyle=':', alpha=0.6)
        
        if 'accuracy' in avg_history:
            axes[1].plot(avg_history['accuracy'], label='Train Accuracy', color='blue', linewidth=2)
            axes[1].plot(avg_history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
            axes[1].set_xlabel('Epochs')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Accuracy Curve', fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, linestyle=':', alpha=0.6)
            
        plt.tight_layout()
        plt.show()
    
    print(" -> Đang xuất biểu đồ Loss & Accuracy...")
    plot_training_history_average(training_histories)
    
    def plot_multiclass_roc(y_true, y_probs, n_classes, classes_names, title):
        y_test_ohe = tf.keras.utils.to_categorical(y_true, n_classes)
        plt.figure(figsize=(9, 7))
        colors = ['dodgerblue', 'crimson', 'forestgreen', 'darkorange']
        
        for i in range(n_classes):
            if i < y_probs.shape[1]:
                fpr, tpr, _ = roc_curve(y_test_ohe[:, i], y_probs[:, i])
                roc_auc = auc(fpr, tpr)
                if not np.isnan(roc_auc):
                    plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2.5,
                             label=f'ROC {classes_names[i]} (AUC = {roc_auc:.4f})')
    
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=15, fontweight='bold', pad=15)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()
    
    print(" -> Đang xuất biểu đồ ROC cho R5.2...")
    plot_multiclass_roc(y_r52, r52_probs_accum, 4, classes_names, title=f"Biểu đồ ROC R5.2 (Ablation Mode: {EXPERIMENT_MODE})")
    
    print("\n" + "="*75)
    print(f"🏁 TOÀN BỘ PIPELINE HOÀN TẤT TRONG: {(time.time() - global_start_time) / 60.0:.2f} PHÚT.")
    print("="*75)