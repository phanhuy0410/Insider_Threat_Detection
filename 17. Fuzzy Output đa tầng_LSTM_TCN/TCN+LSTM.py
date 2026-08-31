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
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.cluster import MiniBatchKMeans
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.regularizers import l2
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.optimizers import AdamW

# ------------------------------------------------------------------------------
# 1. CỐ ĐỊNH SEED VÀ CẤU HÌNH ĐƯỜNG DẪN
# ------------------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
print(f"🔒 Hệ thống đã khóa cứng tính ngẫu nhiên với SEED = {SEED}")

# CHỈ CÒN DUY NHẤT 1 BỘ DỮ LIỆU TENSOR (118 CỘT)
TENSOR_DIR = "/kaggle/input/datasets/doanthimo80/tensor-data-static"
TENSOR_DIR_R52 = "/kaggle/input/datasets/doanthimo80/tensor-data-r52-new" 

PATH_SESSION = '/kaggle/input/datasets/phanthanhhoang/cert-r42-session/session_r4.2.parquet'
PATH_SESSION_R52 = '/kaggle/input/datasets/phanthanhhoang/r5-2-session/sessionr5.2.csv'
PATH_USERDAY_R52_DEV = f"{TENSOR_DIR_R52}/userday_clean_dev_r52.parquet"
PATH_USERDAY_R52_TEST = f"{TENSOR_DIR_R52}/userday_clean_test_r52.parquet"

classes_names = ['Benign', 'Scen1', 'Scen2', 'Scen3']

sample_data = np.load(f"{TENSOR_DIR}/dev_chunk_0.npz")
_, T_sess, F_sess = sample_data['X_sess'].shape
del sample_data; gc.collect()

# ==============================================================================
# 🔄 KHÔI PHỤC USER MAPPING
# ==============================================================================
print("🔄 Đang khôi phục User Mapping R4.2 (Chỉ lấy Dev để Train/Val)...")
df_sess_rec = pd.read_parquet(PATH_SESSION, columns=['user', 'starttime'])
df_sess_rec = df_sess_rec.sort_values(by='starttime').reset_index(drop=True)
test_idx_rec = int(len(df_sess_rec) * 0.80)
unique_users_dev = sorted(df_sess_rec.iloc[:test_idx_rec]['user'].unique().tolist())
user_mapping_dev = {u: i for i, u in enumerate(unique_users_dev)}
del df_sess_rec; gc.collect()

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
# 🧠 CÔNG CỤ FUZZY LAYER & CUSTOM LAYER CHUẨN LOGITS
# ==============================================================================
class GaussianClassOutputLayer(Layer):
    def __init__(self, num_classes=4, init_sigma=1.0, **kwargs):
        super(GaussianClassOutputLayer, self).__init__(**kwargs)
        self.num_classes = int(num_classes)
        self.init_sigma = init_sigma

    def build(self, input_shape):
        self.latent_dim = input_shape[-1]
        self.centers = self.add_weight(name='centers', shape=(self.num_classes, self.latent_dim), initializer='glorot_uniform', trainable=True)
        raw_sigma_init = np.log(np.exp(self.init_sigma) - 1.0)
        self.raw_sigmas = self.add_weight(name='raw_sigmas', shape=(self.num_classes, self.latent_dim), initializer=tf.keras.initializers.Constant(raw_sigma_init), trainable=True)
        super(GaussianClassOutputLayer, self).build(input_shape)

    def call(self, z):
        sigma = tf.nn.softplus(self.raw_sigmas) + 1e-5
        z_exp = tf.expand_dims(z, axis=1)
        c_exp = tf.expand_dims(self.centers, axis=0)
        s_exp = tf.expand_dims(sigma, axis=0)
        diff = (z_exp - c_exp) / s_exp
        logits = -0.5 * tf.reduce_mean(tf.square(diff), axis=-1)
        return logits

def output_to_score(logits):
    score = tf.exp(logits - tf.reduce_max(logits, axis=1, keepdims=True))
    score = score / (tf.reduce_sum(score, axis=1, keepdims=True) + 1e-9)
    return score

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
# 🤖 TẦNG 1
# ==============================================================================
def build_session_encoder(T_sess, F_sess, learning_rate=0.0001):
    inp = Input(shape=(T_sess, F_sess), name='T1_Input_Session')
    x = Dense(128, activation='relu')(inp)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dropout(0.1)(x)
    x = tcn_residual_block(x, n_filters=64, kernel_size=3, dilation_rate=1, dropout_rate=0.2, block_name="Sess_TCN_D1")
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = GatedTemporalAttention()(x)
    
    latent = Dense(64, activation='relu')(x) 
    latent_bn = BatchNormalization()(latent) 
    out = GaussianClassOutputLayer(num_classes=4, init_sigma=1.0)(latent_bn)
    
    model = Model(inputs=inp, outputs=out, name="Session_Encoder_T1")
    model.compile(optimizer=AdamW(learning_rate=0.0001, weight_decay=1e-4), 
                  loss=CategoricalFocalCrossentropy(gamma=2.0, from_logits=True), metrics=['accuracy'])
    return model

def build_user_encoder(F_user, learning_rate=0.0001):
    inp = Input(shape=(F_user,), name='T1_Input_User')
    x = Dense(32, activation='relu')(inp)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Reshape((-1, 1))(x)
    x = tcn_residual_block(x, n_filters=32, kernel_size=3, dilation_rate=1, dropout_rate=0.2, block_name="User_TCN_D1")
    x = LSTM(32, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = GatedTemporalAttention()(x)
    
    latent = Dense(16, activation='relu')(x)
    latent_bn = BatchNormalization()(latent) 
    out = GaussianClassOutputLayer(num_classes=4, init_sigma=1.0)(latent_bn)
    
    model = Model(inputs=inp, outputs=out, name="User_Encoder_T1")
    model.compile(optimizer=AdamW(learning_rate=0.0001, weight_decay=1e-4), 
                  loss=CategoricalFocalCrossentropy(gamma=2.0, from_logits=True), metrics=['accuracy'])
    return model

def build_score_level_fusion(fused_dim=8, learning_rate=0.0001):
    inp = Input(shape=(fused_dim,), name='T1_Input_8D_Scores')
    x = Dense(fused_dim * 2, activation='relu')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(fused_dim, activation='relu')(x)
    x_added = add([inp, x])
    x_final = Dense(16, activation='relu', name='Final_Hidden_Layer', kernel_regularizer=l2(1e-4))(x_added)
    x_final = BatchNormalization()(x_final)
    out = GaussianClassOutputLayer(num_classes=4, init_sigma=1.0, name='Final_Fuzzy_Decision')(x_final)
    model = Model(inputs=inp, outputs=out, name="Full_Fuzzy_Fusion_T1")
    model.compile(optimizer=AdamW(learning_rate=0.0001, weight_decay=1e-4), 
                  loss=CategoricalFocalCrossentropy(gamma=2.0, from_logits=True), metrics=['accuracy'])
    return model

# ==============================================================================
# 🤖 TẦNG 2: INDEPENDENT EXPERT (SCEN2 VS BENIGN)
# ==============================================================================
def build_full_expert_t2(T_sess, F_sess, F_user, learning_rate=0.0001):
    inp_sess = Input(shape=(T_sess, F_sess), name='T2_Input_Session')
    inp_user = Input(shape=(F_user,), name='T2_Input_User')

    # Session Encoder Riêng biệt
    x_sess = Dense(128, activation='relu')(inp_sess)
    x_sess = LayerNormalization(epsilon=1e-6)(x_sess)
    x = tcn_residual_block(x_sess, n_filters=64, kernel_size=3, dilation_rate=1, dropout_rate=0.2, block_name="Sess_TCN_T2")
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x_sess = GatedTemporalAttention()(x_sess)
    lat_sess = Dense(64, activation='relu')(x_sess)

    # User Encoder Riêng biệt
    x_user = Dense(32, activation='relu')(inp_user)
    x_user = LayerNormalization(epsilon=1e-6)(x_user)
    x_user = Reshape((-1, 1))(x_user)
    x = tcn_residual_block(x_user, n_filters=32, kernel_size=3, dilation_rate=1, dropout_rate=0.2, block_name="User_TCN_T2")
    x = LSTM(32, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x_user = GatedTemporalAttention()(x_user)
    lat_user = Dense(16, activation='relu')(x_user)

    merged = Concatenate(name='T2_Feature_Fusion')([lat_sess, lat_user])    
    merged = Dense(64, activation='relu')(merged)
    latent_final = BatchNormalization(name='T2_Latent_BN')(merged)
    out = GaussianClassOutputLayer(num_classes=2, init_sigma=1.0, name='T2_Fuzzy_Decision')(latent_final)
    model = Model(inputs=[inp_sess, inp_user], outputs=out, name="Independent_Expert_T2")
    model.compile(optimizer=AdamW(learning_rate=learning_rate, weight_decay=1e-4), 
                  loss=CategoricalFocalCrossentropy(alpha=[0.3, 0.7], gamma=2.0, from_logits=True), metrics=['accuracy'])
    return model

# ==============================================================================
# 🌀 PIPELINE HUẤN LUYỆN FOLD
# ==============================================================================
def train_single_fold_ablation(fold_idx, train_weeks, val_weeks, 
                               X_mix, y_mix, uid_mix, w_mix, 
                               global_user_lookup, F_user, 
                               chunk_files_r52_test, OFFSET_TEST_52): 
    
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    print(f"\n{'='*20} KHỞI ĐỘNG TIẾN TRÌNH FOLD {fold_idx + 1} {'='*20}")
    
    train_mask = np.isin(w_mix, train_weeks)
    idx_train_raw = np.where(train_mask)[0]
    
    val_mask = np.isin(w_mix, val_weeks)
    idx_val_mix = np.where(val_mask)[0]
    if len(idx_val_mix) == 0: return
    
    X_train_sess_raw = X_mix[idx_train_raw]
    y_train_raw = y_mix[idx_train_raw]
    uid_train_raw = uid_mix[idx_train_raw]
    
    X_val_sess_raw = X_mix[idx_val_mix]
    y_val = y_mix[idx_val_mix]
    uid_val = uid_mix[idx_val_mix]

    # --- CHUẨN HÓA DỮ LIỆU CHUNG (DÙNG CHO CẢ T1 & T2) ---
    np.nan_to_num(X_train_sess_raw, nan=0.0, posinf=65000.0, neginf=-65000.0, copy=False)
    np.nan_to_num(X_val_sess_raw, nan=0.0, posinf=65000.0, neginf=-65000.0, copy=False)
    
    scaler_sess = StandardScaler(copy=False)
    X_train_sess_scaled = scaler_sess.fit_transform(X_train_sess_raw.reshape(-1, F_sess)).astype(np.float32, copy=False).reshape(-1, T_sess, F_sess)
    X_val_sess_scaled = scaler_sess.transform(X_val_sess_raw.reshape(-1, F_sess)).astype(np.float32, copy=False).reshape(-1, T_sess, F_sess)
    del X_train_sess_raw, X_val_sess_raw; gc.collect()

    # --- KMEANS DOWNSAMPLING ---
    idx_benign = np.where(y_train_raw == 0)[0]
    idx_attack = np.where(y_train_raw != 0)[0]
    
    X_tr_sess_flat = X_train_sess_scaled.reshape(len(idx_train_raw), -1)
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
        near_end = int(n_cluster * 0.30)
        n_near = int(n_draw * 0.50)
        selected = []
        if len(sorted_idx[:near_end]) > 0: selected.extend(sorted_idx[:near_end][:min(n_near, len(sorted_idx[:near_end]))])
        chosen_benign_idx.extend(selected)
    
    final_benign_idx = idx_benign[np.array(chosen_benign_idx)]
    if len(final_benign_idx) > target_benign: final_benign_idx = np.random.choice(final_benign_idx, target_benign, replace=False)
    elif len(final_benign_idx) < target_benign: final_benign_idx = np.concatenate([final_benign_idx, np.random.choice(np.setdiff1d(idx_benign, final_benign_idx), target_benign - len(final_benign_idx), replace=False)])
    
    keep_indices = np.concatenate([final_benign_idx, idx_attack])
    keep_indices.sort()
    
    X_train_sess = X_train_sess_scaled[keep_indices]
    y_train = y_train_raw[keep_indices]
    uid_train = uid_train_raw[keep_indices]
    X_val_sess = X_val_sess_scaled
    
    # --- XỬ LÝ USER FEATURES ---
    X_train_user = np.array([global_user_lookup.get(u, np.zeros(F_user)) for u in uid_train], dtype=np.float32)
    X_val_user = np.array([global_user_lookup.get(u, np.zeros(F_user)) for u in uid_val], dtype=np.float32)
    
    np.nan_to_num(X_train_user, nan=0.0, posinf=65000.0, neginf=-65000.0, copy=False)
    np.nan_to_num(X_val_user, nan=0.0, posinf=65000.0, neginf=-65000.0, copy=False)
    
    scaler_user = StandardScaler(copy=False)
    X_train_user = scaler_user.fit_transform(X_train_user).astype(np.float32, copy=False)
    X_val_user = scaler_user.transform(X_val_user).astype(np.float32, copy=False)
    
    y_train_ohe = tf.keras.utils.to_categorical(y_train, num_classes=4)
    y_val_ohe = tf.keras.utils.to_categorical(y_val, num_classes=4)
    tf.keras.backend.clear_session()
    
    # ==========================================================================
    # 🚀 HUẤN LUYỆN TẦNG 1
    # ==========================================================================
    session_encoder = build_session_encoder(T_sess, F_sess)
    user_encoder = build_user_encoder(F_user)
    classifier = build_score_level_fusion(fused_dim=8) 
    
    es_sess = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    es_user = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    es_clf = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    lr_sess = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=0)
    lr_user = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=0)
    lr_clf = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
    
    gen_train_sess = BalancedUniversalGenerator(X_train_sess, y_train_ohe, hacker_ratio=0.3)
    gen_val_sess = UniversalGenerator(X_val_sess, y_val_ohe, shuffle=False)
    start_time_sess = time.time()
    session_encoder.fit(gen_train_sess, validation_data=gen_val_sess, epochs=100, callbacks=[es_sess, lr_sess], verbose=0)
    duration_sess = time.time() - start_time_sess

    gen_train_user = BalancedUniversalGenerator(X_train_user, y_train_ohe, hacker_ratio=0.3)
    gen_val_user = UniversalGenerator(X_val_user, y_val_ohe, shuffle=False)
    start_time_user = time.time()
    user_encoder.fit(gen_train_user, validation_data=gen_val_user, epochs=100, callbacks=[es_user, lr_user], verbose=0)
    duration_user = time.time() - start_time_user
    del gen_train_sess, gen_val_sess, gen_train_user, gen_val_user; gc.collect()

    logits_train_sess = session_encoder.predict(X_train_sess, batch_size=1024, verbose=0)
    logits_train_user = user_encoder.predict(X_train_user, batch_size=1024, verbose=0)
    train_scores_raw = np.hstack([logits_train_sess, logits_train_user]).astype(np.float32, copy=False) 
    
    logits_val_sess = session_encoder.predict(X_val_sess, batch_size=1024, verbose=0)
    logits_val_user = user_encoder.predict(X_val_user, batch_size=1024, verbose=0)
    val_scores_raw = np.hstack([logits_val_sess, logits_val_user]).astype(np.float32, copy=False) 

    smote = SMOTE(sampling_strategy={1: 10000, 2: 10000, 3: 10000}, random_state=SEED, k_neighbors=5)
    train_scores_bal, y_train_bal = smote.fit_resample(train_scores_raw, y_train)
    y_train_bal_ohe = tf.keras.utils.to_categorical(y_train_bal, 4)
    
    gen_train_clf = UniversalGenerator(train_scores_bal, y_train_bal_ohe, batch_size=256, shuffle=True)
    gen_val_clf = UniversalGenerator(val_scores_raw, y_val_ohe, batch_size=256, shuffle=False)
    
    hist_clf = classifier.fit(gen_train_clf, validation_data=gen_val_clf, epochs=100, callbacks=[es_clf, lr_clf], verbose=0)
    clf_history = hist_clf.history.copy()
    del gen_train_clf, gen_val_clf, hist_clf, train_scores_bal; gc.collect()
    
    fold_train_time_minutes = (duration_sess + duration_user) / 60.0
    print(f" ⏱️ THỜI GIAN TRAIN FOLD {fold_idx + 1}: {fold_train_time_minutes:.2f} phút")

    # ==============================================================================
    # 🕵️ CHỌN MẪU TRAIN TẦNG 2
    # ==============================================================================
    print("\n 🕵️ BƯỚC 1: LỌC MẪU TRAIN TẦNG 2 (HARD + EASY BENIGN + ALL SCEN2)...")
    logits_tr_t1 = classifier.predict(train_scores_raw, batch_size=1024, verbose=0)
    scores_tr_t1 = output_to_score(tf.constant(logits_tr_t1)).numpy()
    preds_t1 = np.argmax(scores_tr_t1, axis=1)
    
    idx_fp = np.where((y_train == 0) & (preds_t1 == 2))[0]
    margin = scores_tr_t1[:, 0] - scores_tr_t1[:, 2]
    idx_borderline = np.where((y_train == 0) & (preds_t1 == 0) & (margin < 0.3))[0]
    candidate_idx = np.union1d(idx_fp, idx_borderline)
    
    idx_scen2_all = np.where(y_train == 2)[0]
    TOP_K = len(idx_scen2_all) * 3
    NUM_HARD = int(TOP_K * 0.5)
    NUM_EASY = TOP_K - NUM_HARD
    
    if len(candidate_idx) < NUM_HARD:
        remaining_benign = np.setdiff1d(np.where(y_train == 0)[0], candidate_idx)
        p_scen2_remaining = scores_tr_t1[remaining_benign, 2]
        needed = NUM_HARD - len(candidate_idx)
        top_needed_idx = remaining_benign[np.argsort(p_scen2_remaining)[-needed:]]
        candidate_idx = np.union1d(candidate_idx, top_needed_idx)

    latent_fusion_model = Model(inputs=classifier.input, outputs=classifier.get_layer('Final_Hidden_Layer').output)
    fusion_latent_features = latent_fusion_model.predict(train_scores_raw, batch_size=1024, verbose=0)
    
    fuzzy_layer = classifier.get_layer('Final_Fuzzy_Decision')
    centers, raw_sigmas = fuzzy_layer.get_weights()
    sigmas = np.log(1 + np.exp(raw_sigmas)) + 1e-5
    
    diff_scen2 = (fusion_latent_features[candidate_idx] - centers[2]) / sigmas[2]
    dist_scen2_candidates = np.mean(np.square(diff_scen2), axis=1)
    
    membership_scen2_candidates = scores_tr_t1[candidate_idx, 2]
    def min_max_scale(arr): return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-9)
    
    rank_scores = 0.5 * min_max_scale(membership_scen2_candidates) + 0.5 * min_max_scale(1.0 / (dist_scen2_candidates + 1e-5))
    
    hardest_relative_indices = np.argsort(rank_scores)[-NUM_HARD:]
    idx_benign_hard = candidate_idx[hardest_relative_indices]
    
    idx_benign_easy_pool = np.where((y_train == 0) & (preds_t1 == 0) & (margin > 0.8))[0]
    
    idx_benign_easy_pool = np.setdiff1d(idx_benign_easy_pool, idx_benign_hard)
    
    if len(idx_benign_easy_pool) >= NUM_EASY:
        idx_benign_easy = np.random.choice(idx_benign_easy_pool, NUM_EASY, replace=False)
    else:
        all_benign_left = np.setdiff1d(np.where(y_train == 0)[0], idx_benign_hard)
        idx_benign_easy = np.random.choice(all_benign_left, NUM_EASY, replace=False)
        
    idx_benign_final = np.concatenate([idx_benign_hard, idx_benign_easy])
    
    print(f"   -> Đã chốt Tầng 2: {len(idx_benign_hard)} Hard Benign + {len(idx_benign_easy)} Easy Benign + {len(idx_scen2_all)} Scen2.")
    # ==============================================================================
    # 🚀 HUẤN LUYỆN TẦNG 2
    # ==============================================================================
    idx_t2_train = np.concatenate([idx_benign_final, idx_scen2_all])
    np.random.shuffle(idx_t2_train)
    
    X_train_sess_t2 = X_train_sess[idx_t2_train] 
    X_train_user_t2 = X_train_user[idx_t2_train] 
    
    y_train_t2_raw = y_train[idx_t2_train]
    y_train_t2_bin = np.where(y_train_t2_raw == 2, 1, 0)
    y_train_t2_ohe = tf.keras.utils.to_categorical(y_train_t2_bin, num_classes=2)

    # Lấy Validation cho T2
    idx_t2_val = np.where((y_val == 0) | (y_val == 2))[0]
    X_val_sess_t2 = X_val_sess[idx_t2_val]
    X_val_user_t2 = X_val_user[idx_t2_val]
    y_val_t2_raw = y_val[idx_t2_val]
    y_val_t2_bin = np.where(y_val_t2_raw == 2, 1, 0)
    y_val_t2_ohe = tf.keras.utils.to_categorical(y_val_t2_bin, num_classes=2)

    es_t2 = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    lr_t2 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=0)
    
    model_t2 = build_full_expert_t2(T_sess, F_sess, F_user)
    print(" 🚀 Đang huấn luyện Independent Expert (Tầng 2)...")
    start_time_t2 = time.time()
    model_t2.fit([X_train_sess_t2, X_train_user_t2], y_train_t2_ohe, 
                 validation_data=([X_val_sess_t2, X_val_user_t2], y_val_t2_ohe), 
                 batch_size=128, epochs=100, callbacks=[lr_t2, es_t2], verbose=0)
    duration_t2 = time.time() - start_time_t2
    fold_train_time_minutes_t2 = (duration_t2) / 60.0
    total_train_2_stage = fold_train_time_minutes + fold_train_time_minutes_t2
    print(f" ⏱️ TỔNG THỜI GIAN TRAIN FOLD {fold_idx + 1} CẢ 2 TẦNG: {total_train_2_stage:.2f} phút")
    # ==============================================================================
    # 🌍 SUY LUẬN (INFERENCE)
    # ==============================================================================
    logits_val_t1 = classifier.predict(val_scores_raw, batch_size=1024, verbose=0)
    val_probs_t1 = output_to_score(tf.constant(logits_val_t1)).numpy()
    
    preds_val_t1 = np.argmax(val_probs_t1, axis=1)
    margin_val = val_probs_t1[:, 0] - val_probs_t1[:, 2]
    
    t1_confident_scen2_val = (preds_val_t1 == 2) & (val_probs_t1[:, 2] >= 0.85)
    
    route_mask_val = ((preds_val_t1 == 0) & (margin_val < 0.5)) | ((preds_val_t1 == 2) & (~t1_confident_scen2_val))
    
    val_probs_cascade = val_probs_t1.copy()
    
    if np.any(route_mask_val):
        print(f"   -> [Val Mix] Tầng 2 kích hoạt đánh giá lại cho {np.sum(route_mask_val)}/{len(y_val)} mẫu.")
        X_val_sess_routed = X_val_sess[route_mask_val]
        X_val_user_routed = X_val_user[route_mask_val]
        
        logits_val_t2 = model_t2.predict([X_val_sess_routed, X_val_user_routed], batch_size=1024, verbose=0)
        val_probs_t2 = output_to_score(tf.constant(logits_val_t2)).numpy()
        
        p2_benign = val_probs_t2[:, 0]
        p2_scen2 = val_probs_t2[:, 1]
        mass_02_val = val_probs_t1[route_mask_val, 0] + val_probs_t1[route_mask_val, 2]
        
        new_p0 = mass_02_val * p2_benign
        new_p2 = mass_02_val * p2_scen2
                
        val_probs_cascade[route_mask_val, 0] = new_p0
        val_probs_cascade[route_mask_val, 2] = new_p2
    
    fold_auc = roc_auc_score(y_val_ohe, val_probs_cascade, multi_class='ovr', average='macro')
    print(f" 🔥 MACRO AUC FOLD {fold_idx + 1} (VAL CASCADE T1+T2): {fold_auc:.4f}")

    # --------------------------------------------------------------------------
    # -- TẬP TEST R5.2 --
    # --------------------------------------------------------------------------
    r52_probs_cascade_list = []
    
    for f in chunk_files_r52_test: 
        with np.load(f) as data:
            if len(data['y']) == 0: continue
            mask = data['y'] != 4
            if not np.any(mask): continue
            
            batch_sess = data['X_sess'][mask]
            batch_uid = data['user_id'][mask] + OFFSET_TEST_52 

        batch_sess = np.nan_to_num(batch_sess, nan=0.0, posinf=65000.0, neginf=-65000.0, copy=False)
        batch_sess_s = scaler_sess.transform(batch_sess.reshape(-1, F_sess)).astype(np.float32, copy=False).reshape(-1, T_sess, F_sess)
        
        batch_user_raw = np.array([global_user_lookup.get(u, np.zeros(F_user)) for u in batch_uid], dtype=np.float32)
        batch_user_raw = np.nan_to_num(batch_user_raw, nan=0.0, posinf=65000.0, neginf=-65000.0, copy=False)
        batch_user_s = scaler_user.transform(batch_user_raw).astype(np.float32, copy=False)
        
        l_sess = session_encoder.predict(batch_sess_s, batch_size=1024, verbose=0)
        l_user = user_encoder.predict(batch_user_s, batch_size=1024, verbose=0)
        scores_t1_raw = np.hstack([l_sess, l_user]).astype(np.float32, copy=False)
        
        logits_batch_t1 = classifier.predict(scores_t1_raw, batch_size=1024, verbose=0)
        probs_batch_t1 = output_to_score(tf.constant(logits_batch_t1)).numpy()
        
        preds_batch_t1 = np.argmax(probs_batch_t1, axis=1)
        margin_batch = probs_batch_t1[:, 0] - probs_batch_t1[:, 2]
        
        t1_confident_scen2_batch = (preds_batch_t1 == 2) & (probs_batch_t1[:, 2] >= 0.85)
        route_mask_batch = ((preds_batch_t1 == 0) & (margin_batch < 0.5)) | ((preds_batch_t1 == 2) & (~t1_confident_scen2_batch))
        
        probs_batch_cascade = probs_batch_t1.copy()
        
        if np.any(route_mask_batch):
            X_batch_sess_routed = batch_sess_s[route_mask_batch]
            X_batch_user_routed = batch_user_s[route_mask_batch]
            
            logits_batch_t2 = model_t2.predict([X_batch_sess_routed, X_batch_user_routed], batch_size=1024, verbose=0)
            probs_batch_t2 = output_to_score(tf.constant(logits_batch_t2)).numpy()
            
            p2_benign = probs_batch_t2[:, 0]
            p2_scen2 = probs_batch_t2[:, 1]
            mass = probs_batch_t1[route_mask_batch, 0] + probs_batch_t1[route_mask_batch, 2]
            
            new_p0 = mass * p2_benign
            new_p2 = mass * p2_scen2
            
            probs_batch_cascade[route_mask_batch, 0] = new_p0
            probs_batch_cascade[route_mask_batch, 2] = new_p2
            
        r52_probs_cascade_list.append(probs_batch_cascade)

    r52_probs_cascade = np.concatenate(r52_probs_cascade_list, axis=0)

    # (Lưu file fold như cũ)
    np.savez(
        os.path.join(TMP_RESULT_DIR, f"fold_{fold_idx}.npz"),
        val_probs = val_probs_cascade,        
        y_val = y_val,                
        r52_probs = r52_probs_cascade, 
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
    print("🚀 PIPELINE (INDEPENDENT EXPERT 118 FEATURES): TRAIN MIX | VAL MIX | TEST R5.2")
    print("="*75)
    
    global_start_time = time.time()
    
    print("📥 Đang nạp dữ liệu Tensor 118 cột...")
    X_dev_42_118, y_dev_42, uid_dev_42, w_dev_42, day_dev_42, _, _ = load_pass1_session(TENSOR_DIR, "dev")
    X_dev_52_118, y_dev_52_full, uid_dev_52, w_dev_52, day_dev_52, _, _ = load_pass1_session(TENSOR_DIR_R52, "dev_r52")
    
    mask_dev_52 = y_dev_52_full != 4
    
    X_mix_sess_118 = np.concatenate([X_dev_42_118, X_dev_52_118[mask_dev_52]], axis=0)
    y_mix_full = np.concatenate([y_dev_42, y_dev_52_full[mask_dev_52]], axis=0)
    w_mix_full = np.concatenate([w_dev_42, w_dev_52[mask_dev_52]], axis=0)
    
    del X_dev_42_118, X_dev_52_118; gc.collect()

    OFFSET_DEV_52 = max(user_mapping_dev.values()) + 1 if user_mapping_dev else 0
    OFFSET_TEST_52 = OFFSET_DEV_52 + max(user_mapping_dev_r52.values()) + 1 if user_mapping_dev_r52 else OFFSET_DEV_52
    
    uid_mix_full = np.concatenate([uid_dev_42, uid_dev_52[mask_dev_52] + OFFSET_DEV_52], axis=0)

    user_lookup_42_dev, feature_cols_ud = load_userday_lookup(f"{TENSOR_DIR}/userday_clean_dev.parquet", user_mapping_dev)
    F_user = len(feature_cols_ud) 
    
    user_lookup_52_dev, _ = load_userday_lookup(PATH_USERDAY_R52_DEV, user_mapping_dev_r52)
    user_lookup_52_test, _ = load_userday_lookup(PATH_USERDAY_R52_TEST, user_mapping_test_r52)
    
    global_user_lookup = {
        **user_lookup_42_dev, 
        **{k + OFFSET_DEV_52: v for k, v in user_lookup_52_dev.items()}, 
        **{k + OFFSET_TEST_52: v for k, v in user_lookup_52_test.items()}
    }
    
    _, y_r52_full, _, _, _, _, chunk_files_r52_test = load_pass1_session(TENSOR_DIR_R52, "test_r52")
    y_r52 = y_r52_full[y_r52_full != 4]

    TMP_RESULT_DIR = "./temp_results_dl_independent_cascade"
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
            target=train_single_fold_ablation, 
            args=(fold_idx, train_weeks, val_weeks, 
                  X_mix_sess_118, y_mix_full, uid_mix_full, w_mix_full, 
                  global_user_lookup, F_user, 
                  chunk_files_r52_test, OFFSET_TEST_52)
        )
        p.start(); p.join()  
        if p.exitcode != 0: break 

    print("\n" + "="*75)
    print("📥 ĐANG THU THẬP KẾT QUẢ TRÊN TẤT CẢ TẬP DỮ LIỆU...")
    val_mix_probs_list, val_mix_y_list, fold_aucs, training_histories = [], [], [], []
    r52_probs_accum = np.zeros((len(y_r52), 4)) 

    for fold_idx in range(len(time_folds)):
        file_path = os.path.join(TMP_RESULT_DIR, f"fold_{fold_idx}.npz")
        if os.path.exists(file_path):
            with np.load(file_path) as data:
                val_mix_probs_list.append(data['val_probs']) 
                val_mix_y_list.append(data['y_val'])
                fold_aucs.append(data['fold_auc'][0])
                r52_probs_accum += (data['r52_probs'] / len(time_folds))

    print(f"🏆 TRUNG BÌNH MACRO AUC SAU {len(fold_aucs)} FOLD (VAL MIX): {np.mean(fold_aucs):.4f}")
    
    val_mix_probs_all = np.vstack(val_mix_probs_list)
    val_mix_y_all = np.concatenate(val_mix_y_list)

    OFFLINE_SAVE_PATH = '/kaggle/working/pipeline_independent_cascade_r52_preds.npz'
    np.savez_compressed(
        OFFLINE_SAVE_PATH,
        oof_probs_all=val_mix_probs_all,
        oof_y_all=val_mix_y_all,
        r52_probs_accum=r52_probs_accum,
        y_r52=y_r52
    )
    print(f"📦 ĐÃ LƯU KẾT QUẢ DỰ ĐOÁN TẠI: {OFFLINE_SAVE_PATH}")

    # ==============================================================================
    # 💾 TÌM NGƯỠNG TỐI ƯU TRÊN TẬP VAL MIX
    # ==============================================================================    
    oof_benign_probs = val_mix_probs_all[(val_mix_y_all == 0)]
    THRESH_S1 = np.percentile(oof_benign_probs[:, 1], 99.995)
    THRESH_S2 = np.percentile(oof_benign_probs[:, 2], 95)  
    THRESH_S3 = np.percentile(oof_benign_probs[:, 3], 99.99)

    print(f"📏 Ngưỡng: Scen1: {THRESH_S1:.2f} | Scen2: {THRESH_S2:.2f} | Scen3: {THRESH_S3:.2f}")
    
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

    # 🎨 XUẤT CÁC BIỂU ĐỒ TRỰC QUAN
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

    plot_confusion_matrix_percent(conf_matrix_r52, classes_names, title="Ma Trận Nhầm Lẫn Test R5.2 - TCN+LSTM 2 Stage")

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
    plot_multiclass_roc(y_r52, r52_probs_accum, 4, classes_names, title="Biểu đồ ROC Test R5.2 - TCN+LSTM")