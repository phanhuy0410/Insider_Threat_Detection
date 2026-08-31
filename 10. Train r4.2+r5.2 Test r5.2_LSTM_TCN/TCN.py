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
from tensorflow.keras.layers import Layer, Input, GRU, Bidirectional, LSTM, Dense, Dropout, BatchNormalization, GlobalMaxPooling1D, GlobalAveragePooling1D, Activation
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization, Embedding, Concatenate,  
                                     Reshape, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization, Add, Layer)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D, SpatialDropout1D, add
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.regularizers import l2

# ------------------------------------------------------------------------------
# 1. CỐ ĐỊNH SEED VÀ CẤU HÌNH ĐƯỜNG DẪN
# ------------------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
print(f"🔒 Hệ thống đã khóa cứng tính ngẫu nhiên với SEED = {SEED}")

# ĐƯỜNG DẪN R4.2 (GỐC)
TENSOR_DIR = "/kaggle/input/datasets/doanthimo80/tensor-data-static"
PATH_SESSION = '/kaggle/input/datasets/phanthanhhoang/cert-r42-session/session_r4.2.parquet'

# 🔥 ĐƯỜNG DẪN R5.2
TENSOR_DIR_R52 = "/kaggle/input/datasets/doanthimo80/tensor-data-r52-new" 
PATH_SESSION_R52 = '/kaggle/input/datasets/phanthanhhoang/r5-2-session/sessionr5.2.csv'
PATH_USERDAY_R52_DEV = f"{TENSOR_DIR_R52}/userday_clean_dev_r52.parquet"
PATH_USERDAY_R52_TEST = f"{TENSOR_DIR_R52}/userday_clean_test_r52.parquet"

# KHÓA CHẶT 4 LỚP
classes_names = ['Benign', 'Scen1', 'Scen2', 'Scen3']

sample_data = np.load(f"{TENSOR_DIR}/dev_chunk_0.npz")
_, T_sess, F_sess = sample_data['X_sess'].shape
del sample_data; gc.collect()

# ==============================================================================
# 🔄 KHÔI PHỤC USER MAPPING (ĐỒNG BỘ R4.2 VÀ R5.2)
# ==============================================================================
print("🔄 Đang khôi phục User Mapping R4.2...")
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
del df_sess_rec, df_sess_dev_rec, df_sess_test_raw_rec, df_sess_test_rec; gc.collect()

print("🔄 Đang khôi phục User Mapping R5.2 (Dev và Test chung 1 Dictionary để dễ quản lý)...")
df_sess_r52_rec = pd.read_csv(PATH_SESSION_R52)
unique_users_r52 = sorted(df_sess_r52_rec['user'].unique().tolist())

# Bắt đầu ID R5.2 nối tiếp ngay sau R4.2 để tránh trùng lặp
max_id_r42 = max(user_mapping_dev.values()) if user_mapping_dev else 0
user_mapping_r52 = {u: i + max_id_r42 + 1 for i, u in enumerate(unique_users_r52)}
del df_sess_r52_rec; gc.collect()

print(f"✅ User R4.2 Dev: {len(user_mapping_dev)} | Test: {len(user_mapping_test)} | User R5.2: {len(user_mapping_r52)}")

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

# ==============================================================================
# 🧠 CÁC LỚP CUSTOM LAYER VÀ ENCODER
# ==============================================================================
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
                  padding='causal', # Bắt buộc dùng causal cho time-series
                  kernel_initializer='he_normal',
                  name=f"{block_name}_conv")(x)
    
    bn = BatchNormalization(name=f"{block_name}_bn")(conv)
    act = Activation('relu', name=f"{block_name}_act")(bn)
    drop = SpatialDropout1D(dropout_rate, name=f"{block_name}_drop")(act)
    
    if x.shape[-1] != n_filters:
        shortcut = Conv1D(filters=n_filters, kernel_size=1, padding='same', name=f"{block_name}_shortcut")(x)
    else:
        shortcut = x
    res = Add(name=f"{block_name}_add")([drop, shortcut])
    return Activation('relu', name=f"{block_name}_res_out")(res)
    
def build_session_encoder(T_sess, F_sess, learning_rate=0.0001):
    inp = Input(shape=(T_sess, F_sess), name='Input_Session')
    x = tcn_residual_block(x, n_filters=64, kernel_size=3, dilation_rate=1, dropout_rate=0.2, block_name="Sess_TCN_D1")
    x = GatedTemporalAttention(name='Session_Temporal_Attention')(x)
    latent = Dense(64, activation='relu', name='Session_Latent')(x) 
    latent_bn = BatchNormalization(name='Session_Latent_BN')(latent) 
    out = Dense(4, activation='softmax')(latent_bn)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=CategoricalFocalCrossentropy(gamma=2.0), metrics=['accuracy'])
    return model

def build_user_encoder(F_user, learning_rate=0.0001):
    inp = Input(shape=(F_user,), name='Input_User')
    x = Reshape((F_user, 1), name='Expand_Dims_Reshape')(inp)
    x = tcn_residual_block(x, n_filters=32, kernel_size=3, dilation_rate=1, dropout_rate=0.2, block_name="User_TCN_D1")
    x = GatedTemporalAttention()(x)
    latent = Dense(32, activation='relu', name='User_Latent')(x)
    latent_bn = BatchNormalization(name='User_Latent_BN')(latent) 
    out = Dense(4, activation='softmax')(latent_bn)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=CategoricalFocalCrossentropy(gamma=2.0), metrics=['accuracy'])
    return model

def build_score_level_fusion(fused_dim=8):
    inp = Input(shape=(fused_dim,), name='Input_8D_Scores')
    x = Dense(fused_dim * 2, activation='relu', name='Cross_Correlation_1')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(fused_dim, activation='relu', name='Cross_Correlation_2')(x)
    x_added = add([inp, x], name='Residual_Add')
    x_final = Dense(16, activation='relu', name='Final_Hidden_Layer', kernel_regularizer=l2(1e-3))(x_added)
    x_final = BatchNormalization()(x_final)
    out = Dense(4, activation='softmax', name='Final_Decision')(x_final)
    model = Model(inputs=inp, outputs=out, name="Bi_View_NN_Fusion")
    model.compile(optimizer=Adam(learning_rate=0.00001), loss=CategoricalFocalCrossentropy(gamma=2.0), metrics=['accuracy'])
    return model

# ==============================================================================
# 🌀 PIPELINE HUẤN LUYỆN: JOINT TRAINING (R4.2 + R5.2)
# ==============================================================================
def train_single_fold_joint(fold_idx, train_weeks, val_weeks, X_mix_sess_full, y_mix_full, uid_mix_full, w_mix_full, 
                            global_user_lookup, F_user, 
                            user_lookup_test, chunk_files_test, len_y_test,
                            chunk_files_r52_test): 
    
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    print(f"\n{'='*20} KHỞI ĐỘNG TIẾN TRÌNH FOLD {fold_idx + 1} (JOINT TRAINING) {'='*20}")
    
    # Chia Train/Val trên tập Mix
    train_mask = np.isin(w_mix_full, train_weeks)
    val_mask = np.isin(w_mix_full, val_weeks)
    idx_train_raw = np.where(train_mask)[0]
    idx_val = np.where(val_mask)[0]
    if len(idx_val) == 0: return
    
    # --- KMEANS DOWNSAMPLING ---
    X_tr_sess_flat = X_mix_sess_full[idx_train_raw].reshape(len(idx_train_raw), -1)
    y_train_temp = y_mix_full[idx_train_raw]
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
        n_cluster = len(sorted_idx)
        near_end, mid_end = int(n_cluster * 0.30), int(n_cluster * 0.70)
        near_idx, mid_idx, far_idx = sorted_idx[:near_end], sorted_idx[near_end:mid_end], sorted_idx[mid_end:int(n_cluster * 0.95)]
        n_near, n_mid = int(n_draw * 0.50), int(n_draw * 0.30)
        n_far = n_draw - n_near - n_mid
        selected = []
        if len(near_idx) > 0: selected.extend(near_idx[:min(n_near, len(near_idx))])
        if len(mid_idx) > 0 and n_mid > 0: 
            start = max(0, len(mid_idx)//2 - n_mid//2)
            selected.extend(mid_idx[start:min(len(mid_idx), start + n_mid)])
        if len(far_idx) > 0 and n_far > 0: selected.extend(far_idx[-min(n_far, len(far_idx)):])
        chosen_benign_idx.extend(selected)
    
    final_benign_idx = idx_benign[np.array(chosen_benign_idx)]
    if len(final_benign_idx) > target_benign: final_benign_idx = np.random.choice(final_benign_idx, target_benign, replace=False)
    elif len(final_benign_idx) < target_benign: final_benign_idx = np.concatenate([final_benign_idx, np.random.choice(np.setdiff1d(idx_benign, final_benign_idx), target_benign - len(final_benign_idx), replace=False)])
    
    keep_indices = np.concatenate([final_benign_idx, idx_attack])
    keep_indices.sort()
    keep_global_idx = idx_train_raw[keep_indices]
    
    X_train_sess = X_mix_sess_full[keep_global_idx]
    y_train = y_mix_full[keep_global_idx]
    uid_train = uid_mix_full[keep_global_idx]
    
    X_val_sess = X_mix_sess_full[idx_val]
    y_val = y_mix_full[idx_val]
    uid_val = uid_mix_full[idx_val]
    
    X_train_user = np.array([global_user_lookup.get(u, np.zeros(F_user)) for u in uid_train], dtype=np.float32)
    X_val_user = np.array([global_user_lookup.get(u, np.zeros(F_user)) for u in uid_val], dtype=np.float32)
    
    # --- CHUẨN HÓA ---
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
    
    session_encoder = build_session_encoder(T_sess, F_sess, learning_rate=0.0001)
    user_encoder = build_user_encoder(F_user)
    classifier = build_score_level_fusion(fused_dim=8) 
    
    es_sess = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    es_user = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    es_clf = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    lr_sess = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=0)
    lr_user = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=0)
    rlr_clf = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
    
    gen_train_sess = BalancedUniversalGenerator(X_train_sess, y_train_ohe, hacker_ratio=0.25)
    gen_val_sess = UniversalGenerator(X_val_sess, y_val_ohe, shuffle=False)
    
    # ⏱️ Đo thời gian Session Encoder
    start_time_sess = time.time()
    session_encoder.fit(gen_train_sess, validation_data=gen_val_sess, epochs=100, callbacks=[es_sess, lr_sess], verbose=0)
    duration_sess = time.time() - start_time_sess
    del gen_train_sess, gen_val_sess; gc.collect()

    gen_train_user = BalancedUniversalGenerator(X_train_user, y_train_ohe, hacker_ratio=0.25)
    gen_val_user = UniversalGenerator(X_val_user, y_val_ohe, shuffle=False)
    
    # ⏱️ Đo thời gian User Encoder
    start_time_user = time.time()
    user_encoder.fit(gen_train_user, validation_data=gen_val_user, epochs=100, callbacks=[es_user, lr_user], verbose=0)
    duration_user = time.time() - start_time_user
    del gen_train_user, gen_val_user; gc.collect()

    # --- RÚT XÁC SUẤT ĐỂ FUSION ---
    def predict_eager_batch(model, X_data, X_cat=None, batch_size=512):
        res = []
        for start_idx in range(0, len(X_data), batch_size):
            end_idx = min(start_idx + batch_size, len(X_data))
            batch = X_data[start_idx:end_idx]
            if X_cat is not None:
                batch_cat = X_cat[start_idx:end_idx]
                batch = np.concatenate([batch, batch_cat], axis=-1)
            res.append(model(batch, training=False).numpy())
        return np.concatenate(res, axis=0)
    
    prob_train_sess = predict_eager_batch(session_encoder, X_train_sess, batch_size=512)
    prob_train_user = predict_eager_batch(user_encoder, X_train_user, batch_size=512)
    train_scores_raw = np.hstack([prob_train_sess, prob_train_user]).astype(np.float32, copy=False)
    del X_train_sess, X_train_user, prob_train_sess, prob_train_user; gc.collect() 

    prob_val_sess = predict_eager_batch(session_encoder, X_val_sess, batch_size=512)
    prob_val_user = predict_eager_batch(user_encoder, X_val_user, batch_size=512)
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

    # Tính tổng thời gian mô hình HỌC THỰC SỰ
    fold_train_time_minutes = (duration_sess + duration_user) / 60.0
    print(f" ⏱️ THỜI GIAN TRAIN FOLD {fold_idx + 1} (Chỉ tính lúc fit): {fold_train_time_minutes:.2f} phút")

    val_probs = predict_eager_batch(classifier, val_scores_raw, batch_size=512)
    fold_auc = roc_auc_score(y_val_ohe, val_probs, multi_class='ovr', average='macro')
    print(f" 🔥 MACRO AUC FOLD {fold_idx + 1} (TRÊN TẬP VAL MIX): {fold_auc:.4f}")

    # ---------------------------------------------------------
    # 🎯 INFERENCE TẬP TEST R4.2
    # ---------------------------------------------------------
    prob_test_sess_list, prob_test_user_list = [], []
    for f in chunk_files_test: 
        with np.load(f) as data:
            if len(data['y']) == 0: continue
            batch_sess, batch_uid = data['X_sess'], data['user_id']

        batch_sess = np.nan_to_num(batch_sess, nan=0.0, posinf=65000.0, neginf=-65000.0, copy=False)
        batch_sess_s = scaler_sess.transform(batch_sess.reshape(-1, F_sess)).astype(np.float32, copy=False).reshape(-1, T_sess, F_sess)
        prob_test_sess_list.append(predict_eager_batch(session_encoder, batch_sess_s, batch_size=512))
        
        batch_user_raw = np.array([user_lookup_test.get(u, np.zeros(F_user)) for u in batch_uid], dtype=np.float32)
        batch_user_raw = np.nan_to_num(batch_user_raw, nan=0.0, posinf=65000.0, neginf=-65000.0, copy=False)
        batch_user_s = scaler_user.transform(batch_user_raw).astype(np.float32, copy=False)
        prob_test_user_list.append(predict_eager_batch(user_encoder, batch_user_s, batch_size=512))

    prob_test_sess = np.concatenate(prob_test_sess_list, axis=0)
    prob_test_user = np.concatenate(prob_test_user_list, axis=0)
    test_scores_raw = np.hstack([prob_test_sess, prob_test_user]).astype(np.float32, copy=False)
    test_probs_fold = predict_eager_batch(classifier, test_scores_raw, batch_size=512)
    del prob_test_sess_list, prob_test_user_list, prob_test_sess, prob_test_user, test_scores_raw; gc.collect()

    # ---------------------------------------------------------
    # 🎯 INFERENCE TẬP TEST R5.2 (LỌC BỎ KỊCH BẢN 4)
    # ---------------------------------------------------------
    prob_r52_sess_list, prob_r52_user_list = [], []
    for f in chunk_files_r52_test: 
        with np.load(f) as data:
            if len(data['y']) == 0: continue
            
            mask = data['y'] != 4
            if not np.any(mask): continue
            
            batch_sess = data['X_sess'][mask]
            batch_uid = data['user_id'][mask]

        batch_sess = np.nan_to_num(batch_sess, nan=0.0, posinf=65000.0, neginf=-65000.0, copy=False)
        batch_sess_s = scaler_sess.transform(batch_sess.reshape(-1, F_sess)).astype(np.float32, copy=False).reshape(-1, T_sess, F_sess)
        prob_r52_sess_list.append(predict_eager_batch(session_encoder, batch_sess_s, batch_size=512))
        
        batch_user_raw = np.array([global_user_lookup.get(u, np.zeros(F_user)) for u in batch_uid], dtype=np.float32)
        batch_user_raw = np.nan_to_num(batch_user_raw, nan=0.0, posinf=65000.0, neginf=-65000.0, copy=False)
        batch_user_s = scaler_user.transform(batch_user_raw).astype(np.float32, copy=False)
        prob_r52_user_list.append(predict_eager_batch(user_encoder, batch_user_s, batch_size=512))

    prob_r52_sess = np.concatenate(prob_r52_sess_list, axis=0)
    prob_r52_user = np.concatenate(prob_r52_user_list, axis=0)
    r52_scores_raw = np.hstack([prob_r52_sess, prob_r52_user]).astype(np.float32, copy=False)
    r52_probs_fold = predict_eager_batch(classifier, r52_scores_raw, batch_size=512)
    del prob_r52_sess_list, prob_r52_user_list, prob_r52_sess, prob_r52_user, r52_scores_raw; gc.collect()

    # --- LƯU KẾT QUẢ VÀ THỜI GIAN TRAIN ---
    np.savez(
        os.path.join(TMP_RESULT_DIR, f"fold_{fold_idx}.npz"),
        val_probs = val_probs,        
        y_val = y_val,                
        test_probs = test_probs_fold, 
        r52_probs = r52_probs_fold, 
        fold_auc = np.array([fold_auc]),
        train_time = np.array([fold_train_time_minutes]) 
    )
    
    with open(os.path.join(TMP_RESULT_DIR, f"history_{fold_idx}.pkl"), 'wb') as f:
        pickle.dump(clf_history, f)
        
    print(f" ✅ TIẾN TRÌNH FOLD {fold_idx + 1} HOÀN TẤT. TRẢ LẠI 100% RAM!")
    return

# ==============================================================================
# 🚀 VÙNG AN TOÀN MAIN PROCESS
# ==============================================================================
if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    print("\n" + "="*75)
    print("🚀 BẮT ĐẦU CHẠY TIME-AWARE K-FOLD (MULTI-DOMAIN JOINT TRAINING R4.2 + R5.2)")
    print("="*75)
    
    # 1. NẠP DỮ LIỆU DEV R4.2
    X_dev_42, y_dev_42, uid_dev_42, w_dev_42, day_dev_42, _, _ = load_pass1_session(TENSOR_DIR, "dev")
    user_lookup_42_dev, feature_cols_ud = load_userday_lookup(f"{TENSOR_DIR}/userday_clean_dev.parquet", user_mapping_dev)
    F_user = len(feature_cols_ud) 
    user_lookup_test, _ = load_userday_lookup(f"{TENSOR_DIR}/userday_clean_test.parquet", user_mapping_test)
    _, y_test, _, _, _, _, chunk_files_test = load_pass1_session(TENSOR_DIR, "test")
    
    # 2. NẠP DỮ LIỆU DEV R5.2
    print("📥 Nạp dữ liệu R5.2 để Joint Training...")
    user_lookup_52_dev, _ = load_userday_lookup(PATH_USERDAY_R52_DEV, user_mapping_r52)
    user_lookup_52_test, _ = load_userday_lookup(PATH_USERDAY_R52_TEST, user_mapping_r52)
    X_dev_52, y_dev_52_full, uid_dev_52, w_dev_52, day_dev_52, _, _ = load_pass1_session(TENSOR_DIR_R52, "dev_r52")
    _, y_r52_full, _, _, _, _, chunk_files_r52_test = load_pass1_session(TENSOR_DIR_R52, "test_r52")

    # 🔥 BƯỚC ĐỒNG BỘ: XÓA SCEN 4 VÀ NỐI DỮ LIỆU (CONCAT)
    mask_dev_52 = y_dev_52_full != 4
    X_dev_52 = X_dev_52[mask_dev_52]
    y_dev_52 = y_dev_52_full[mask_dev_52]
    uid_dev_52 = uid_dev_52[mask_dev_52]
    w_dev_52 = w_dev_52[mask_dev_52]
    day_dev_52 = day_dev_52[mask_dev_52]
    
    y_r52 = y_r52_full[y_r52_full != 4] # Chỉ giữ lại nhãn để tính AUC r52 Test

    # 🔥 GỘP CHUNG DEV 4.2 VÀ DEV 5.2 THÀNH TẬP MIX
    X_mix_sess_full = np.concatenate([X_dev_42, X_dev_52], axis=0)
    X_mix_sess_full = np.nan_to_num(X_mix_sess_full, nan=0.0, posinf=65000.0, neginf=-65000.0)
    y_mix_full = np.concatenate([y_dev_42, y_dev_52], axis=0)
    uid_mix_full = np.concatenate([uid_dev_42, uid_dev_52], axis=0)
    w_mix_full = np.concatenate([w_dev_42, w_dev_52], axis=0)
    
    max_id_r42 = max(user_mapping_dev.values()) if user_mapping_dev else 0
    OFFSET = max_id_r42 + 1
    print(f" -> Tự động tính OFFSET = {OFFSET} (ID R5.2 sẽ nối tiếp ngay sau R4.2)")
    uid_dev_52 = uid_dev_52 + OFFSET 
    user_lookup_52_dev = {k + OFFSET: v for k, v in user_lookup_52_dev.items()}
    user_lookup_52_test = {k + OFFSET: v for k, v in user_lookup_52_test.items()}

    # 🔥 GỘP CHUNG TỪ ĐIỂN USER LOOKUP ĐỂ TRA CỨU ĐỒNG THỜI
    global_user_lookup = {**user_lookup_42_dev, **user_lookup_52_dev, **user_lookup_52_test}
    
    print(f" -> Đã gộp thành công! Tổng số mẫu Train/Val (Mix): {len(y_mix_full)}")
    print(f" -> R5.2 Test (không Scen4): {len(y_r52)}")

    del X_dev_42, X_dev_52; gc.collect()

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
        train_weeks = fold_cfg["train_weeks"]
        val_weeks = fold_cfg["val_weeks"]
        print(f"\n⏳ Đang nạp lệnh cho Fold {fold_idx + 1}")
        p = mp.Process(
            target=train_single_fold_joint, 
            args=(fold_idx, train_weeks, val_weeks, X_mix_sess_full, y_mix_full, uid_mix_full, w_mix_full, 
                  global_user_lookup, F_user, 
                  user_lookup_test, chunk_files_test, len(y_test),
                  chunk_files_r52_test) # Không cần truyền user_lookup_r52 vì đã có global_user_lookup
        )
        p.start()
        p.join()  
        if p.exitcode != 0: break 

    print("\n" + "="*75)
    print("📥 ĐANG THU THẬP KẾT QUẢ TRÊN TẤT CẢ TẬP DỮ LIỆU...")
    oof_probs_list, oof_y_list, fold_aucs, training_histories = [], [], [], []
    fold_train_times = [] 
    
    test_probs_accum = np.zeros((len(y_test), 4)) 
    r52_probs_accum = np.zeros((len(y_r52), 4)) 

    for fold_idx in range(len(time_folds)):
        file_path = os.path.join(TMP_RESULT_DIR, f"fold_{fold_idx}.npz")
        hist_path = os.path.join(TMP_RESULT_DIR, f"history_{fold_idx}.pkl")
        if os.path.exists(file_path):
            with np.load(file_path) as data:
                oof_probs_list.append(data['val_probs'])
                oof_y_list.append(data['y_val'])
                fold_aucs.append(data['fold_auc'][0])
                test_probs_accum += (data['test_probs'] / len(time_folds))
                r52_probs_accum += (data['r52_probs'] / len(time_folds))
                
                if 'train_time' in data:
                    fold_train_times.append(data['train_time'][0])
                    
        if os.path.exists(hist_path):
            with open(hist_path, 'rb') as f:
                training_histories.append(pickle.load(f))

    print(f"🏆 TRUNG BÌNH MACRO AUC SAU {len(fold_aucs)} FOLD (VAL MIX): {np.mean(fold_aucs):.4f}")
    
    if fold_train_times:
        print(f"⏱️ THỜI GIAN TRAIN TRUNG BÌNH/FOLD: {np.mean(fold_train_times):.2f} phút")
        print(f"⏱️ TỔNG THỜI GIAN TRAIN (CÁC FOLD): {np.sum(fold_train_times):.2f} phút")
        
    oof_probs_all = np.vstack(oof_probs_list)
    oof_y_all = np.concatenate(oof_y_list)

    # ==============================================================================
    # 💾 LƯU TOÀN BỘ KẾT QUẢ RA DISK ĐỂ TUNE NGƯỠNG SAU (CHẠY NGẦM KAGGLE)
    # ==============================================================================
    print("\n" + "="*75)
    print("💾 ĐANG LƯU KẾT QUẢ PREDICT VÀO /kaggle/working/ ĐỂ TUNE NGƯỠNG OFFLINE...")
    
    SAVE_PATH = '/kaggle/working/joint_training_predictions.npz'
    np.savez_compressed(
        SAVE_PATH,
        oof_probs_all=oof_probs_all,       # Dùng để tìm ngưỡng (Threshold)
        oof_y_all=oof_y_all,               # Nhãn gốc của tập Val Mix
        test_probs_accum=test_probs_accum, # Xác suất dự đoán của R4.2 Test
        y_test=y_test,                     # Nhãn gốc của R4.2 Test
        r52_probs_accum=r52_probs_accum,   # Xác suất dự đoán của R5.2 Test
        y_r52=y_r52                        # Nhãn gốc của R5.2 Test
    )
    print(f"✅ Đã lưu thành công file: {SAVE_PATH}")
    print("="*75)
    
    # ==============================================================================
    # 💾 ĐÁNH GIÁ VÀ IN BÁO CÁO TEST R4.2
    # ==============================================================================
    # Điểm oof_benign_probs bây giờ là Mix, nên ngưỡng (Threshold) cũng là Mix Threshold
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
    
    test_auc = roc_auc_score(tf.keras.utils.to_categorical(y_test, 4), test_probs_accum, multi_class='ovr', average='macro')
    print(f"\n🔥 MACRO AUC TẬP TEST (R4.2 GỐC): {test_auc:.4f} 🔥\n")
    print(classification_report(y_test, test_preds, target_names=classes_names, digits=4, zero_division=0))
    
    conf_matrix_test = confusion_matrix(y_test, test_preds)

    # ==============================================================================
    # 🌍 ĐÁNH GIÁ VÀ IN BÁO CÁO TEST R5.2 (TÁI HIỆU CHỈNH NGƯỠNG)
    # ==============================================================================
    print("\n" + "*"*70)
    print("🌍 KẾT QUẢ KIỂM CHÉO TRÊN TẬP DỮ LIỆU R5.2 (NGƯỠNG TỰ ĐỘNG - 4 LỚP)")
    print("*"*70)
    
    r52_benign_probs = r52_probs_accum[(y_r52 == 0)]
    THRESH_R52_S1 = np.percentile(r52_benign_probs[:, 1], 99.99)
    THRESH_R52_S2 = np.percentile(r52_benign_probs[:, 2], 99.5)  
    THRESH_R52_S3 = np.percentile(r52_benign_probs[:, 3], 99.99)
    
    r52_preds = np.zeros(len(r52_probs_accum), dtype=int)
    for i in range(len(r52_probs_accum)):
        candidates = []
        if r52_probs_accum[i, 1] >= THRESH_R52_S1: candidates.append((1, r52_probs_accum[i, 1]))
        if r52_probs_accum[i, 2] >= THRESH_R52_S2: candidates.append((2, r52_probs_accum[i, 2]))
        if r52_probs_accum[i, 3] >= THRESH_R52_S3: candidates.append((3, r52_probs_accum[i, 3]))
        if candidates: r52_preds[i] = max(candidates, key=lambda item: item[1])[0]
    
    r52_auc = roc_auc_score(tf.keras.utils.to_categorical(y_r52, 4), r52_probs_accum, multi_class='ovr', average='macro')
    
    print(f"\n🚀 MACRO AUC TẬP TEST (R5.2 ĐÃ BỎ SCEN 4): {r52_auc:.4f} 🚀\n")
    print(classification_report(y_r52, r52_preds, target_names=classes_names, digits=4, zero_division=0))
    
    conf_matrix_r52 = confusion_matrix(y_r52, r52_preds)

    # ==============================================================================
    # 🎨 XUẤT CÁC BIỂU ĐỒ ĐỂ BÁO CÁO LUẬN VĂN
    # ==============================================================================
    
    def plot_confusion_matrix_percent(cm, classes, title='Confusion Matrix'):
        plt.figure(figsize=(8, 6))
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        row_sums_safe = np.where(row_sums == 0, 1, row_sums) 
        cm_percentages = cm.astype('float') / row_sums_safe

        sns.heatmap(cm_percentages, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        
        plt.title(title, fontsize=15, fontweight='bold', pad=15)
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        plt.show()

    print("\n -> Đang xuất biểu đồ Confusion Matrix cho R4.2...")
    plot_confusion_matrix_percent(conf_matrix_test, classes_names, title="Ma Trận Nhầm Lẫn R4.2 Test (Joint Training)")

    print(" -> Đang xuất biểu đồ Confusion Matrix cho R5.2...")
    plot_confusion_matrix_percent(conf_matrix_r52, classes_names, title="Ma Trận Nhầm Lẫn R5.2 Test (Joint Training)")

    def plot_training_history_average(histories):
        if not histories: return
        max_epochs = max([len(h['loss']) for h in histories])
        print(f" -> Đang tính trung bình 5 Fold (Kéo giãn các Fold ngắn cho bằng {max_epochs} Epochs)...")
        
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
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Learning Curve (Joint Training)', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, linestyle=':', alpha=0.6)
        
        if 'accuracy' in avg_history:
            axes[1].plot(avg_history['accuracy'], label='Train Accuracy', color='blue', linewidth=2)
            axes[1].plot(avg_history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
            axes[1].set_xlabel('Epochs')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Accuracy Curve (Joint Training)', fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, linestyle=':', alpha=0.6)
            
        plt.tight_layout()
        plt.show()

    print(" -> Đang xuất biểu đồ Loss & Accuracy...")
    plot_training_history_average(training_histories)

    def plot_multiclass_roc(y_test, y_probs, n_classes, classes_names, title):
        y_test_ohe = tf.keras.utils.to_categorical(y_test, n_classes)
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
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=15, fontweight='bold', pad=15)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

    print(" -> Đang xuất biểu đồ ROC cho R4.2...")
    plot_multiclass_roc(y_test, test_probs_accum, 4, classes_names, title="Biểu đồ ROC")
    
    print(" -> Đang xuất biểu đồ ROC cho R5.2...")
    plot_multiclass_roc(y_r52, r52_probs_accum, 4, classes_names, title="Biểu đồ ROC")